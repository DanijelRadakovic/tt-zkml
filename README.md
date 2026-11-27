# tt-zkml

Zero-knowledge proofs of quantized neural-network inference executed on Tenstorrent hardware: the network runs on a Tenstorrent device, the
execution trace is exported, and a stwo proof of it is generated in Rust.

Modules (each one is self-contained, with its own build and README, and can move to its own repository):

- [`tt-zkml-core/`](tt-zkml-core/README.md): C++. Everything that touches the Tenstorrent device: kernels, device operations, the quantized network
  execution, benchmarks and docs. Its public C++ API (`src/api`) runs a model and returns the execution trace.
- [`abi/`](abi/README.md): the C ABI of the core (`libtt_zkml_abi.so`) and its Python bindings. The one interface other languages use to reach the device code.
- [`tt-zkml/`](tt-zkml/README.md): Rust workspace. `lib/tt-zkml-sys` (raw bindings to the C ABI), `lib/tt-zkml` (safe API, model files,
  AIR constraints, stwo prover) and `bin/tt-zkml` (the `tt-zkml` CLI: load a model, run the forward pass, generate the output and the proof).

## Example

Needs a built tt-metal and a Tenstorrent device. Build the C ABI first (the Rust side links it), then use either the CLI or the Rust library.

```bash
export TT_METAL_HOME=/path/to/tt-metal TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
(cd abi && cmake --preset default && cmake --build .build/default --target tt_zkml_abi)
cd tt-zkml                       # the Rust workspace (nightly toolchain, pinned by rust-toolchain.toml)
```

### CLI

```bash
# 1. a random example model (model.json + weights) and an input; layers 64 -> 128 (ReLU) -> 32, batch 32, bf16 on the device
cargo run --release --bin tt-zkml -- example --out /tmp/demo --dims 64,128,32 --batch 32 --dtype bf16

# 2. run it on the device and print the execution trace (optionally write it with --trace-dir)
cargo run --release --bin tt-zkml -- run --model /tmp/demo --input /tmp/demo/input.f32

# 3. run it, generate the proof of the execution and verify it
cargo run --release --bin tt-zkml -- prove --model /tmp/demo --input /tmp/demo/input.f32 --proof /tmp/demo/proof.bin
#   output [32 x 32], first row: [8, -25, 18, -18, -6, 14, 1, 7, ...]
#   proof generated in 0.11 s (1611 KiB)
#   proof verified in 0.01 s

# 4. verify it later, elsewhere, without a device: the verifier needs the proof and the float model and input the proof is about
cargo run --release --bin tt-zkml -- verify --proof /tmp/demo/proof.bin --model /tmp/demo --input /tmp/demo/input.f32
#   proof valid (2 layers, output 1024 values), verified in 0.01 s      (exit code 1 and a message if it is not valid)
```

A model is a directory with `model.json` (`{"dtype": "bf16", "layers": [{"in": 64, "out": 128, "relu": true, "weights": "w0.f32"}, ...]}`) and the weights
as little-endian f32 (`[in, out]` row-major); the input is a little-endian f32 file (`[batch, in]` row-major). All sizes are multiples of 32.

### Rust library

The same flow as a program, `tt-zkml/lib/tt-zkml/examples/quickstart.rs` (`cargo run --release --example quickstart`):

```rust
use tt_zkml::modelfile::ModelFile;
use tt_zkml::witness::PublicInputs;
use tt_zkml::{device, proof, Device, Dtype, Layer, Model};

// A model: float weights `[in, out]` row-major per layer, and a float input `[batch, in]`.
let layers = vec![Layer { in_features: 64, out_features: 128, relu: true }, Layer { in_features: 128, out_features: 32, relu: false }];
let model_file = ModelFile { dtype: Dtype::Bf16, layers, weights };   // weights: Vec<Vec<f32>>, input: Vec<f32> (batch 32)

// 1. Run it on the device: the trace holds the output, and the int32 accumulators and the scalers of every layer.
device::check_abi()?;
let device = Device::open(0)?;
let mut model = Model::new(&device, &model_file.layers, model_file.dtype)?;
model.load_parameters(&model_file.weights)?;
let trace = model.forward(&input, 32)?;
let output: Vec<i32> = trace.output_values();   // the network output: [32, 32] int8 values (quantized; trace.output holds the raw device bytes)
let floats: Vec<f32> = trace.dequantized_output();   // the same output as floats, ~ what the unquantized network computes (divided by all the scalers)

// 2. Prove it: the CPU rebuilds the quantized tensors from the floats and the trace, converts them to M31 and proves the constraints.
let proof = proof::prove(&model_file, &input, &trace)?;

// 3. Verify it (no device needed): the verifier supplies the float model and input; the output is part of the proof (proof.output == output,
//    and proof.dequantized_output() == floats: the scalers are in the public statement).
proof::verify(&proof, &PublicInputs::new(&model_file, &input))?;

// The proof is plain bytes (proof.to_bytes() / Proof::from_bytes) and so is easy to store or send.
```

The device context is thread-affine: open the device, create and run models and drop them on one thread (the `main` thread in the CLI and the tests).
The Python bindings of the C ABI (`abi/python`) cover the device part (steps 1) only; the proof is Rust.

<!-- TODO: use the tt-metal library from the Rust code for proving (generate the proof on the device, not only on the CPU). -->
