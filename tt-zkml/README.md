# tt-zkml (Rust)

```
lib/tt-zkml-sys/   raw bindings to the C ABI (abi/include/tt_zkml/tt_zkml.h); links libtt_zkml_abi.so
lib/tt-zkml/       library: device (safe Device / Model / Trace), modelfile (model directory format), air (constraints), prover (stwo)
bin/tt-zkml/       the CLI (package tt-zkml-cli, binary tt-zkml)
```

Build the C++ side first (`cmake --preset default && cmake --build .build/default --target tt_zkml_abi` in `abi/`), then `cargo test`.
The library directory is `<repo>/abi/.build/default/lib`, or `TT_ZKML_ABI_LIB_DIR`; `TT_METAL_HOME` gives the RPATH of the tt-metal libraries.

```bash
tt-zkml example --out model/ --dims 64,128,32 --batch 64   # model.json, w0.f32 ..., input.f32 (random)
tt-zkml run     --model model/ --input model/input.f32 [--trace-dir trace/]
tt-zkml prove   --model model/ --input model/input.f32 [--proof proof.bin]   # exit code 2 until the prover exists
```

Model format: a directory with `model.json` (`{"dtype": "bf16", "layers": [{"in": 64, "out": 128, "relu": true, "weights": "w0.f32"}, ...]}`) and the
weights as little-endian f32, `[in, out]` row-major. Input: little-endian f32, `[batch, in]` row-major. The trace directory has `output.u8`,
`acc_<layer>.u32` (little-endian raw sign-magnitude int32 bits) and `trace.json` (scalers and file names).

## Proof status

`air.rs`: the constraint of the int8 matmul of a layer (`acc = sum_k x_k * w_k`, one row per output element, `2K + 1` columns), checked against the
device's accumulator on the host before proving, plus the range check of `x` and `w` (LogUp lookup of `value + 127` into a preprocessed 256-row table, the
verifier recomputes the table's commitment). `prover.rs`: `prove_layers` / `verify` over stwo (Blake2s, ~80 bit), one component per layer and one for
the table in one proof. Tests: `cargo test -p tt-zkml --lib`. Not done: the requantization + ReLU, the link between layers, the public inputs,
and `prove(&Trace)` (the device trace has to carry `q_x`/`q_w`, or the host re-derives them). Needs nightly Rust (`rust-toolchain.toml`).
