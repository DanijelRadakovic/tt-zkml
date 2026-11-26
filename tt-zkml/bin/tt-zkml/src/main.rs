//! tt-zkml: load a model, run the forward pass on a Tenstorrent device, generate the output and the proof.
//!
//! ```text
//! tt-zkml example --out model/ --dims 64,128,32          write a random model and an input
//! tt-zkml run     --model model/ --input model/input.f32  run the forward pass, write the trace
//! tt-zkml prove   --model model/ --input model/input.f32  run the forward pass and generate the proof
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use tt_zkml::modelfile::{self, ModelFile};
use tt_zkml::{device, prover, Device, Dtype, Layer, Model, Trace};

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
fn forward(a: &ForwardArgs) -> Result<Trace, String> {
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
    Ok(trace)
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
    let trace = forward(&a.fwd)?;
    match prover::prove(&trace) {
        Ok(proof) => {
            fs::write(&a.proof, &proof.bytes).map_err(|e| format!("{}: {e}", a.proof.display()))?;
            println!("proof written to {}", a.proof.display());
            Ok(ExitCode::SUCCESS)
        }
        Err(e) => {
            // The forward pass worked (the trace above is real); the proof is the part that does not exist yet.
            eprintln!("{e}");
            Ok(ExitCode::from(2))
        }
    }
}

// keep the type in the `use` list used (the library re-exports it for callers that build layers by hand)
#[allow(dead_code)]
fn _types(_: Dtype) {}
