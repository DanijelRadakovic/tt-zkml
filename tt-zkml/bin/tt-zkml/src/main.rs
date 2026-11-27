//! tt-zkml: load a model, run the forward pass on a Tenstorrent device, generate the output and the proof.
//!
//! ```text
//! tt-zkml example --out model/ --dims 64,128,32          write a random model and an input
//! tt-zkml run     --model model/ --input model/input.f32  run the forward pass, write the trace
//! tt-zkml prove   --model model/ --input model/input.f32  run the forward pass and generate the proof
//! tt-zkml commit  --model model/ --out model.commitment     commit to a model (done once, by whoever holds the weights)
//! tt-zkml verify  --proof proof.bin --commitment model.commitment --input input.f32   check a proof: no device, no weights
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use tt_zkml::modelfile::{self, ModelFile};
use tt_zkml::proof::ModelCommitment;
use tt_zkml::witness::PublicInputs;
use tt_zkml::{device, proof, Device, Dtype, Layer, Model, Trace};

#[derive(Parser)]
#[command(name = "tt-zkml", version, about = "zkML on Tenstorrent: run a quantized network on the device and prove its execution")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write a random example model (model.json and the weights) and an input
    Example(ExampleArgs),
    /// Load a model, run the forward pass, print the output and write the execution trace
    Run(RunArgs),
    /// Run the forward pass and generate the proof of the execution
    Prove(ProveArgs),
    /// Verify a proof (needs no device and no weights: the model's commitment and the input)
    Verify(VerifyArgs),
    /// Commit to a model: writes the architecture and the 32-byte root of the weights, which is all a verifier needs
    Commit(CommitArgs),
}

#[derive(Args)]
struct CommitArgs {
    /// Model directory (model.json and the weights)
    #[arg(long)]
    model: PathBuf,
    /// Where to write the commitment (JSON)
    #[arg(long, default_value = "model.commitment")]
    out: PathBuf,
}

#[derive(Args)]
struct VerifyArgs {
    /// The proof written by `prove`
    #[arg(long, default_value = "proof.bin")]
    proof: PathBuf,
    /// The commitment of the model the proof is about (written by `commit`)
    #[arg(long, conflicts_with = "model")]
    commitment: Option<PathBuf>,
    /// Or the model itself (model.json and the weights): its commitment is computed first
    #[arg(long)]
    model: Option<PathBuf>,
    /// The input the proof is about: little-endian f32
    #[arg(long)]
    input: PathBuf,
}

#[derive(Args)]
struct ExampleArgs {
    /// Directory to write (created if missing)
    #[arg(long)]
    out: PathBuf,
    /// Feature sizes: the input, then the output of every layer (multiples of 32), e.g. 64,128,32
    #[arg(long, value_delimiter = ',', default_value = "64,128,32")]
    dims: Vec<u32>,
    /// f32 or bf16
    #[arg(long, default_value = "bf16")]
    dtype: String,
    /// Rows of the input (a multiple of 32)
    #[arg(long, default_value_t = 32)]
    batch: u32,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Args, Clone)]
struct ForwardArgs {
    /// Model directory (model.json and the weights)
    #[arg(long)]
    model: PathBuf,
    /// Input: little-endian f32, [batch, in_features] row-major
    #[arg(long)]
    input: PathBuf,
    /// Device id
    #[arg(long, default_value_t = 0)]
    device: u32,
    /// Write the execution trace here (output.u8, acc_<layer>.u32, trace.json)
    #[arg(long)]
    trace_dir: Option<PathBuf>,
}

#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    fwd: ForwardArgs,
}

#[derive(Args)]
struct ProveArgs {
    #[command(flatten)]
    fwd: ForwardArgs,
    /// Where to write the proof
    #[arg(long, default_value = "proof.bin")]
    proof: PathBuf,
}

fn main() -> ExitCode {
    match real_main() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn real_main() -> Result<ExitCode, String> {
    let cli = Cli::parse();
    match cli.command {
        Command::Example(a) => example(a).map(|_| ExitCode::SUCCESS),
        Command::Run(a) => forward(&a.fwd).map(|_| ExitCode::SUCCESS),
        Command::Prove(a) => prove(a),
        Command::Verify(a) => verify(a),
        Command::Commit(a) => commit(a).map(|_| ExitCode::SUCCESS),
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------

/// A small LCG, so that an example needs no dependency.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        lo + ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
    }
}

fn example(a: ExampleArgs) -> Result<(), String> {
    if a.dims.len() < 2 || a.dims.iter().any(|d| *d == 0 || d % 32 != 0) {
        return Err("--dims needs at least two sizes, each a non-zero multiple of 32".into());
    }
    if a.batch == 0 || a.batch % 32 != 0 {
        return Err("--batch must be a non-zero multiple of 32".into());
    }
    let dtype = modelfile::parse_dtype(&a.dtype)?;
    let mut rng = Lcg(a.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let n_layers = a.dims.len() - 1;
    let mut layers = Vec::new();
    let mut weights = Vec::new();
    for l in 0..n_layers {
        let (i, o) = (a.dims[l], a.dims[l + 1]);
        layers.push(Layer { in_features: i, out_features: o, relu: l + 1 < n_layers });
        weights.push((0..i as usize * o as usize).map(|_| rng.next(-1.0, 1.0)).collect());
    }
    modelfile::save_model(&a.out, &ModelFile { dtype, layers, weights })?;
    let x: Vec<f32> = (0..a.batch as usize * a.dims[0] as usize).map(|_| rng.next(-2.0, 2.0)).collect();
    modelfile::write_f32(&a.out.join("input.f32"), &x)?;
    println!("wrote {} ({} layers, dims {:?}, {}) and input.f32 ({} x {})", a.out.display(), n_layers, a.dims, a.dtype, a.batch, a.dims[0]);
    Ok(())
}

/// Loads the model, runs the forward pass on the device and returns the trace (and prints what happened).
fn forward(a: &ForwardArgs) -> Result<(ModelFile, Vec<f32>, Trace), String> {
    let model_file = modelfile::load_model(&a.model)?;
    let x = modelfile::read_f32(&a.input)?;
    let in0 = model_file.layers[0].in_features as usize;
    if x.is_empty() || x.len() % in0 != 0 {
        return Err(format!("{}: {} values are not a whole number of rows of {} features", a.input.display(), x.len(), in0));
    }
    let batch = (x.len() / in0) as u32;
    device::check_abi().map_err(|e| e.to_string())?;
    println!("tt-zkml {} (ABI {}); model: {} layers, {}, batch {}", device::version(), device::abi_version(), model_file.layers.len(), modelfile::dtype_name(model_file.dtype), batch);

    // The device, the model and everything on them live in this function: one thread, closed before the process exits.
    let t0 = Instant::now();
    let dev = Device::open(a.device).map_err(|e| e.to_string())?;
    let mut model = Model::new(&dev, &model_file.layers, model_file.dtype).map_err(|e| e.to_string())?;
    println!("opened device {} in {:.2} s", a.device, t0.elapsed().as_secs_f64());

    let t1 = Instant::now();
    model.load_parameters(&model_file.weights).map_err(|e| e.to_string())?;
    println!("loaded the parameters in {:.1} ms", t1.elapsed().as_secs_f64() * 1e3);

    let t2 = Instant::now();
    let trace = model.forward(&x, batch).map_err(|e| e.to_string())?;
    println!("forward pass in {:.1} ms (includes the first-call compile)", t2.elapsed().as_secs_f64() * 1e3);
    drop(model);
    drop(dev);

    print_trace(&trace);
    if let Some(dir) = &a.trace_dir {
        write_trace(dir, &trace)?;
        println!("trace written to {}", dir.display());
    }
    Ok((model_file, x, trace))
}

fn print_trace(t: &Trace) {
    println!("scaler_x = {}", t.scaler_x);
    for (i, l) in t.layers.iter().enumerate() {
        println!("layer {i}: out_features {}, scaler_w = {}, scaler_out = {}, acc {} values", l.out_features, l.scaler_w, l.scaler_out, l.acc.len());
    }
    let last = t.layers.last().map(|l| l.out_features as usize).unwrap_or(0);
    let vals = t.output_values();
    println!("output [{} x {}], first row: {:?}", t.batch, last, &vals[..last.min(vals.len()).min(16)]);
}

fn write_trace(dir: &Path, t: &Trace) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    fs::write(dir.join("output.u8"), &t.output).map_err(|e| e.to_string())?;
    let mut layers = Vec::new();
    for (i, l) in t.layers.iter().enumerate() {
        let mut bytes = Vec::with_capacity(l.acc.len() * 4);
        for a in &l.acc {
            bytes.extend_from_slice(&a.to_le_bytes());
        }
        fs::write(dir.join(format!("acc_{i}.u32")), bytes).map_err(|e| e.to_string())?;
        layers.push(format!(r#"{{"out_features": {}, "scaler_w": {}, "scaler_out": {}, "acc": "acc_{i}.u32"}}"#, l.out_features, l.scaler_w, l.scaler_out));
    }
    let json = format!(
        r#"{{"batch": {}, "in_features": {}, "scaler_x": {}, "output": "output.u8", "layers": [{}]}}"#,
        t.batch,
        t.in_features,
        t.scaler_x,
        layers.join(", ")
    );
    fs::write(dir.join("trace.json"), json).map_err(|e| e.to_string())
}

fn prove(a: ProveArgs) -> Result<ExitCode, String> {
    let (model, x, trace) = forward(&a.fwd)?;
    // The CPU rebuilds q_x / q_w from the trace, converts everything to M31 and proves the constraints.
    let t = Instant::now();
    let proof = proof::prove(&model, &x, &trace).map_err(|e| e.to_string())?;
    println!("proof generated in {:.2} s ({} KiB)", t.elapsed().as_secs_f64(), proof.bytes.len() / 1024);
    fs::write(&a.proof, proof.to_bytes()).map_err(|e| format!("{}: {e}", a.proof.display()))?;
    println!("proof written to {}", a.proof.display());
    // what a verifier would hold: the commitment of the model (not the weights) and the input
    let commitment = ModelCommitment::new(&model).map_err(|e| e.to_string())?;
    println!("model commitment id: {}", hex(&commitment.id()));
    let t = Instant::now();
    proof::verify(&proof, &PublicInputs { model: commitment, input: x }).map_err(|e| e.to_string())?;
    println!("proof verified in {:.2} s", t.elapsed().as_secs_f64());
    Ok(ExitCode::SUCCESS)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn commit(a: CommitArgs) -> Result<(), String> {
    let model = modelfile::load_model(&a.model)?;
    let t = Instant::now();
    let commitment = ModelCommitment::new(&model).map_err(|e| e.to_string())?;
    fs::write(&a.out, commitment.to_json()).map_err(|e| format!("{}: {e}", a.out.display()))?;
    println!("model commitment (id {}; {} layers, {}) written to {} in {:.2} s", hex(&commitment.id()), commitment.layers.len(), modelfile::dtype_name(commitment.dtype), a.out.display(), t.elapsed().as_secs_f64());
    Ok(())
}

fn verify(a: VerifyArgs) -> Result<ExitCode, String> {
    let bytes = fs::read(&a.proof).map_err(|e| format!("{}: {e}", a.proof.display()))?;
    let proof = proof::Proof::from_bytes(&bytes).map_err(|e| e.to_string())?;
    let commitment: ModelCommitment = match (&a.commitment, &a.model) {
        (Some(path), _) => {
            let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            ModelCommitment::from_json(&text).map_err(|e| format!("{}: {e}", path.display()))?
        }
        (None, Some(dir)) => ModelCommitment::new(&modelfile::load_model(dir)?).map_err(|e| e.to_string())?,
        (None, None) => return Err("give --commitment (the model's commitment) or --model (the model itself)".into()),
    };
    let input = modelfile::read_f32(&a.input)?;
    let t = Instant::now();
    match proof::verify(&proof, &PublicInputs { model: commitment, input }) {
        Ok(()) => {
            println!("proof valid ({} layers, output {} values), verified in {:.2} s", proof.specs.len(), proof.output.len(), t.elapsed().as_secs_f64());
            Ok(ExitCode::SUCCESS)
        }
        Err(e) => {
            eprintln!("{e}");
            Ok(ExitCode::from(1))
        }
    }
}

// keep the type in the `use` list used (the library re-exports it for callers that build layers by hand)
#[allow(dead_code)]
fn _types(_: Dtype) {}
