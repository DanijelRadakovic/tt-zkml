// Run a quantized network on the device, prove the execution and verify the proof.
//
//     cargo run --release --example quickstart
use tt_zkml::modelfile::ModelFile;
use tt_zkml::proof::{ModelCommitment, VerifyingKey};
use tt_zkml::witness::PublicInputs;
use tt_zkml::{device, proof, Device, Dtype, Layer, Model};

fn main() {
    // A model: two fully connected layers (64 -> 128 with ReLU -> 32), float weights `[in, out]` row-major, and a float input `[batch, 64]`.
    let layers = vec![Layer { in_features: 64, out_features: 128, relu: true }, Layer { in_features: 128, out_features: 32, relu: false }];
    let weights = vec![pseudo(1, 64 * 128), pseudo(2, 128 * 32)];
    let input = pseudo(3, 32 * 64);
    let model_file = ModelFile { dtype: Dtype::Bf16, layers, weights };

    // 1. Run it on the device. The execution trace holds, per layer, the int32 accumulators and the power-of-two scalers, and the output.
    device::check_abi().unwrap();
    let device = Device::open(0).unwrap();
    let mut model = Model::new(&device, &model_file.layers, model_file.dtype).unwrap();
    model.load_parameters(&model_file.weights).unwrap();
    let trace = model.forward(&input, 32).unwrap();
    println!("output (int8), first row: {:?}", &trace.output_values()[..8]);
    println!("output (float), first row: {:?}", &trace.dequantized_output()[..8]);

    // 2. Prove it: the CPU rebuilds the quantized tensors from the floats and the trace, converts them to M31 and proves the constraints.
    let proof = proof::prove(&model_file, &input, &trace).unwrap();
    println!("proof: {} KiB, {} layers", proof.bytes.len() / 1024, proof.specs.len());

    // 3. Verify it: no device and no weights needed. The verifier holds the model's COMMITMENT (the architecture and a 32-byte root, computed once
    //    by whoever holds the weights) and the float input; the output is part of the proof.
    let commitment = ModelCommitment::new(&model_file).unwrap();
    println!("model commitment id: {}", commitment.id().iter().map(|b| format!("{b:02x}")).collect::<String>());
    proof::verify(&proof, &PublicInputs { model: commitment.clone(), input: input.clone() }).unwrap();
    println!("proof verified: the committed model on this input gives this output");
    assert_eq!(proof.dequantized_output(), trace.dequantized_output()); // computable from the proof alone

    // Many proofs of one model and batch size: prepare a key once, then each verification costs nothing proportional to the model.
    let key = VerifyingKey::new(&commitment, 32).unwrap();
    proof::verify_with_key(&proof, &key, &input).unwrap();

    // The proof is plain bytes: store it, send it, verify it elsewhere.
    let bytes = proof.to_bytes();
    proof::verify_with_key(&proof::Proof::from_bytes(&bytes).unwrap(), &key, &input).unwrap();
}

fn pseudo(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}
