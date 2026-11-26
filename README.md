# tt-zkml

Zero-knowledge proofs of quantized neural-network inference executed on Tenstorrent hardware: the network runs on a Tenstorrent device, the
execution trace is exported, and a stwo proof of it is generated in Rust.

Modules (each one is self-contained, with its own build and README, and can move to its own repository):

- [`tt-zkml-core/`](tt-zkml-core/README.md): C++. Everything that touches the Tenstorrent device: kernels, device operations, the quantized network
  execution, benchmarks and docs. Its public C++ API (`src/api`) runs a model and returns the execution trace.
- [`abi/`](abi/README.md): the C ABI of the core (`libtt_zkml_abi.so`) and its Python bindings. The one interface other languages use to reach the device code.
- [`tt-zkml/`](tt-zkml/README.md): Rust workspace. `lib/tt-zkml-sys` (raw bindings to the C ABI), `lib/tt-zkml` (safe API, model files,
  AIR constraints, stwo prover) and `bin/tt-zkml` (the `tt-zkml` CLI: load a model, run the forward pass, generate the output and the proof).

<!-- TODO: add an example of running the `tt-zkml` CLI on a model (example -> run -> prove). -->
<!-- TODO: use the tt-metal library from the Rust code for proving (generate the proof on the device, not only on the CPU). -->
