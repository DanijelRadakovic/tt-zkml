// The whole path on a device: create a network, run it on the device, rebuild the witness from the device trace, generate the proof and verify it; then
// forged traces, forged inputs and forged proofs, which must all be refused.
//
// A test program without the libtest harness (`harness = false` in Cargo.toml): the device context of tt-metal is thread-affine, so everything runs on the
// main thread, and the device is opened once.
use stwo::core::fields::qm31::SecureField;
use tt_zkml::modelfile::ModelFile;
use tt_zkml::proof::{self, Proof};
use tt_zkml::witness::{effective, PublicInputs};
use tt_zkml::{device, Device, Dtype, Layer, Model, Trace};

fn main() {
    let device = Device::open(0).unwrap();
    for dtype in [Dtype::F32, Dtype::Bf16] {
        let net = Network::new(dtype, &[64, 128, 64, 32], 32, 7);
        let (trace, proof) = honest_proof(&device, &net);
        forged_traces(&net, &trace);
        forged_proofs(&net, &proof);
        println!("{dtype:?}: ok");
    }
    // a network whose first layer has no ReLU, and a single layer
    for dims in [vec![32, 64], vec![64, 32, 32]] {
        let net = Network::new(Dtype::Bf16, &dims, 32, 11);
        honest_proof(&device, &net);
    }
    drop(device);
    println!("all tests passed");
}

// A small LCG: no dependency for the test data.
fn lcg(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + ((s >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
        })
        .collect()
}

/// A network with its input: the model as a verifier knows it (float weights) and the float input.
struct Network {
    model: ModelFile,
    input: Vec<f32>,
    batch: u32,
}

impl Network {
    fn new(dtype: Dtype, dims: &[u32], batch: u32, seed: u64) -> Self {
        let n = dims.len() - 1;
        let layers: Vec<Layer> = (0..n).map(|l| Layer { in_features: dims[l], out_features: dims[l + 1], relu: l + 1 < n }).collect();
        let weights = layers.iter().enumerate().map(|(l, ly)| lcg(seed + l as u64, (ly.in_features * ly.out_features) as usize, -1.0, 1.0)).collect();
        let input = lcg(seed + 100, (batch * dims[0]) as usize, -2.0, 2.0);
        Self { model: ModelFile { dtype, layers, weights }, input, batch }
    }

    fn public(&self) -> PublicInputs {
        PublicInputs::new(&self.model, &self.input)
    }

    fn run(&self, device: &Device) -> Trace {
        let mut model = Model::new(device, &self.model.layers, self.model.dtype).unwrap();
        model.load_parameters(&self.model.weights).unwrap();
        model.forward(&self.input, self.batch).unwrap()
    }
}

/// The float network on the CPU (the floats the device sees): `relu?(x @ w)` per layer.
fn float_forward(net: &Network) -> Vec<f32> {
    let mut x = effective(&net.input, net.model.dtype);
    for (l, layer) in net.model.layers.iter().enumerate() {
        let (k, n) = (layer.in_features as usize, layer.out_features as usize);
        let w = effective(&net.model.weights[l], net.model.dtype);
        let mut y = vec![0f32; net.batch as usize * n];
        for i in 0..net.batch as usize {
            for kk in 0..k {
                for j in 0..n {
                    y[i * n + j] += x[i * k + kk] * w[kk * n + j];
                }
            }
        }
        if layer.relu {
            y.iter_mut().for_each(|v| *v = v.max(0.0));
        }
        x = y;
    }
    x
}

/// Runs the network on the device, builds the proof from the trace and verifies it.
fn honest_proof(device: &Device, net: &Network) -> (Trace, Proof) {
    let trace = net.run(device);
    let proof = proof::prove(&net.model, &net.input, &trace).unwrap();
    // the statement is the device's: its output and its scalers
    assert_eq!(proof.output, trace.output_values());
    assert_eq!(proof.specs.len(), trace.layers.len());
    for (spec, lt) in proof.specs.iter().zip(&trace.layers) {
        assert_eq!(2f32.powi(spec.kw), lt.scaler_w);
        assert_eq!(2f32.powi(spec.ko), lt.scaler_out);
        assert_eq!(2f32.powi(spec.kx), trace.scaler_x);
    }
    // the CPU's choice of the scalers is the device's: pot_k of the absolute maximum of the floats the device sees
    let absmax = |v: &[f32]| effective(v, net.model.dtype).iter().fold(0.0f32, |a, x| a.max(x.abs()));
    assert_eq!(proof.specs[0].kx, device::pot_k(absmax(&net.input), 8).unwrap());
    for (l, spec) in proof.specs.iter().enumerate() {
        assert_eq!(spec.kw, device::pot_k(absmax(&net.model.weights[l]), 8).unwrap());
    }
    proof::verify(&proof, &net.public()).unwrap();
    // the dequantized output: the same from the trace and from the proof, and close to the float network on the CPU
    assert_eq!(proof.dequantized_output(), trace.dequantized_output());
    let (got, want) = (trace.dequantized_output(), float_forward(net));
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    let worst = got.iter().zip(&want).fold(0f32, |a, (g, w)| a.max((g - w).abs()));
    assert!(worst <= 0.1 * scale, "the dequantized output is off by {worst}, the largest output is {scale}");
    // the proof round-trips through its serialization
    let again = Proof::from_bytes(&proof.to_bytes()).unwrap();
    proof::verify(&again, &net.public()).unwrap();
    (trace, proof)
}

/// Forged device traces and forged inputs: the witness cannot be rebuilt, so there is no proof.
fn forged_traces(net: &Network, trace: &Trace) {
    let refused = |model: &ModelFile, input: &[f32], forged: &Trace, what: &str| {
        assert!(proof::prove(model, input, forged).is_err(), "a proof was generated for {what}");
    };
    // a wrong accumulator, in the first and in the last layer
    for l in [0, trace.layers.len() - 1] {
        let mut forged = trace.clone();
        forged.layers[l].acc[3] = forged.layers[l].acc[3].wrapping_add(1);
        refused(&net.model, &net.input, &forged, "a wrong accumulator");
    }
    // a wrong output
    let mut forged = trace.clone();
    forged.output[0] ^= 1;
    refused(&net.model, &net.input, &forged, "a wrong output");
    // wrong scalers: every one of them
    let mut forged = trace.clone();
    forged.scaler_x *= 2.0;
    refused(&net.model, &net.input, &forged, "a wrong scaler_x");
    for l in 0..trace.layers.len() {
        let mut forged = trace.clone();
        forged.layers[l].scaler_w *= 0.5;
        refused(&net.model, &net.input, &forged, "a wrong scaler_w");
        let mut forged = trace.clone();
        forged.layers[l].scaler_out *= 2.0;
        refused(&net.model, &net.input, &forged, "a wrong scaler_out");
    }
    // a scaler that is not a power of two
    let mut forged = trace.clone();
    forged.scaler_x = 3.0;
    refused(&net.model, &net.input, &forged, "a scaler that is not a power of two");
    // the trace of this execution, but another input, or other weights, or another model
    // (a change below the resolution of the quantization would be another valid execution of the same trace, so the change is large)
    let mut input = net.input.clone();
    input[0] += 0.5;
    refused(&net.model, &input, trace, "another input");
    let mut model = net.model.clone();
    model.weights[0][0] = -model.weights[0][0] - 0.5;
    refused(&model, &net.input, trace, "other weights");
    let mut model = net.model.clone();
    model.layers[0].relu = !model.layers[0].relu;
    refused(&model, &net.input, trace, "another activation");
    refused(&net.model, &net.input[1..], trace, "an input of the wrong size");
}

/// Forged proofs and forged public inputs: the verifier rejects them.
fn forged_proofs(net: &Network, proof: &Proof) {
    let public = net.public();
    let rejected = |forged: &Proof, public: &PublicInputs, what: &str| assert!(proof::verify(forged, public).is_err(), "the verifier accepted {what}");

    // the proof itself
    let mut forged = proof.clone();
    let mid = forged.bytes.len() / 2;
    forged.bytes[mid] = if forged.bytes[mid] == b'1' { b'2' } else { b'1' };
    rejected(&forged, &public, "a tampered proof");
    let mut forged = proof.clone();
    forged.bytes.truncate(forged.bytes.len() / 2);
    rejected(&forged, &public, "a truncated proof");
    let mut forged = proof.clone();
    forged.claimed_sums[0] += claimed_one();
    rejected(&forged, &public, "a forged claimed sum");

    // the public statement
    let mut forged = proof.clone();
    forged.output[1] = if forged.output[1] == 5 { 6 } else { 5 };
    rejected(&forged, &public, "another output");
    for edit in [
        |p: &mut Proof| p.specs[0].ko -= 1,
        |p: &mut Proof| p.specs[1].relu = !p.specs[1].relu,
        |p: &mut Proof| p.specs[0].kw += 1,
        |p: &mut Proof| p.specs.iter_mut().for_each(|s| s.kx -= 1),
        |p: &mut Proof| p.specs[0].m *= 2,
    ] {
        let mut forged = proof.clone();
        edit(&mut forged);
        rejected(&forged, &public, "another statement");
    }
    let mut forged = proof.clone();
    forged.specs.pop();
    forged.claimed_sums.pop();
    rejected(&forged, &public, "a truncated network");

    // the public floats: the proof is about this model and this input only
    let mut other = public.clone();
    other.input[7] += 0.05;
    rejected(proof, &other, "another input");
    let mut other = public.clone();
    // the smallest change the device can see: the lowest mantissa bit of an f32, of a bf16 (bits below it do not survive the rounding)
    let lowest_bit = if net.model.dtype == Dtype::Bf16 { 1 << 16 } else { 1 };
    other.weights[0][9] = f32::from_bits(other.weights[0][9].to_bits() ^ lowest_bit);
    rejected(proof, &other, "weights that differ by one bit");
    // a change below the dtype's precision is the same model for the device, and the proof still holds
    if net.model.dtype == Dtype::Bf16 {
        let mut same = public.clone();
        same.weights[0][9] = f32::from_bits(same.weights[0][9].to_bits() ^ 1);
        if effective(&same.weights[0], Dtype::Bf16) == effective(&public.weights[0], Dtype::Bf16) {
            proof::verify(proof, &same).unwrap();
        }
    }
    let mut other = public.clone();
    other.weights.swap(1, 2);
    rejected(proof, &other, "swapped weights");
    let mut other = public.clone();
    other.weights.pop();
    rejected(proof, &other, "a model with fewer layers");
    let mut other = public;
    other.dtype = if net.model.dtype == Dtype::F32 { Dtype::Bf16 } else { Dtype::F32 };
    // the other dtype rounds the floats differently: the same floats do not match unless every one of them is exactly representable
    if effective(&other.input, other.dtype) != effective(&net.input, net.model.dtype) {
        rejected(proof, &other, "the floats of another dtype");
    }
}

fn claimed_one() -> SecureField {
    SecureField::from_u32_unchecked(1, 0, 0, 0)
}
