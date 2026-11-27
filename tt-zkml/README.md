# tt-zkml (Rust)

```
lib/tt-zkml-sys/   raw bindings to the C ABI (abi/include/tt_zkml/tt_zkml.h); links libtt_zkml_abi.so
lib/tt-zkml/       library: device (safe Device / Model / Trace), modelfile (model directory format), air (constraints), witness, proof (prove / verify with stwo)
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

## Proof

```bash
tt-zkml commit --model model/ --out model.commitment                                # once, by whoever holds the weights
tt-zkml prove  --model model/ --input model/input.f32 [--proof proof.bin]           # device forward pass, then the proof (and its verification)
tt-zkml verify --proof proof.bin --commitment model.commitment --input input.f32    # no device, no weights
```

**What the verifier holds.** Not the model: a `ModelCommitment` (the architecture: layer sizes, ReLU flags, dtype; and a 32-byte Merkle root of the
weights' floats), the float input, and the proof with its public statement (the output and the scaler exponents). The commitment is computed once from
the weights and depends on nothing else (not on the batch size or the input). The proof contains five commitment trees: the model (its root must be the
commitment), the trace, the lookup trace, the constants of the statement (index columns, flags, multiplicities, tables: the verifier knows their root
from the shapes alone; a `VerifyingKey` caches it per model and batch size) and the composition polynomial. Verification then costs nothing proportional
to the model: about 0.04 s for a 256-512-256-64 network at batch 64, with a 2.2 MB proof and a 591-byte commitment.

**Not hidden in the cryptographic sense.** The proof is not zero knowledge (stwo does not blind the trace). The verifier is never given the weights, but
every proof opens the committed columns at about 64 positions and one random point; these are evaluations of fixed polynomials, so a number of proofs of
the order of (number of weights of the largest layer) / 65 determines them. Treat the commitment as "the verifier does not need the model", not as
"the model is secret". Real hiding needs zero-knowledge blinding of the model tree, which is not implemented.

The device returns, per layer, the int32 accumulator and the power-of-two scalers. The CPU (`witness.rs`) rebuilds `q_x0` and `q_w` from the float
input / weights and `scaler_x` / `scaler_w`, and every next `q_x` from the previous accumulator, checks that they reproduce the device's accumulators and
output, converts everything to M31 columns (`air/trace.rs`) and proves the constraints of `air/`:

- `Tensor` (input `X_0` and weights `W_l`): every element takes its float `(sign, m, e')` and proves `q = round_half_even(float * 2^k)` (variable shift
  from the power-of-two table, ties to even), that no scaled float exceeds 127 and that `k` is the largest exponent that fits; `q` is range checked to
  [-127, 127] and yielded once per use by the matmul. For the input the float is the verifier's public one (a lookup); for the weights it is the committed
  model: the row equals the model's columns, and since nobody but the committer knows them, the constraints make every committed triple well formed
  themselves (sign a bit, significand limb checked). bf16 models: the floats are the bf16-rounded values.
- `Matmul(l)`: `acc = sum_k x_k * w_k`, one row per output element; `x_k`, `w_k` and `acc` are tied by LogUp lookups to the tensors.
- `Requant(l)`: ReLU, then `out = round_half_even(acc / 2^s)` (`acc = out * 2^s + rem`, `|rem| <= 2^(s-1)`, ties to even) or `acc * 2^ko`, and that the
  exponent is the largest one (one row has `|acc| * 2^(ko+1) > 127`); its output is the next layer's `X`, so the layers are linked.
- Tables: the range table `0 ..= 254` (values and 7-bit limbs) and the powers of two.
- Public statement: the layer shapes, ReLU flags and exponents, the network output, the model commitment and (known to the verifier, not in the proof)
  the float input: the verifier adds the input and the output to the LogUp sum and compares the model tree's root with the commitment, so a proof is about
  exactly that committed model on that input.

Soundness: the constraints were audited (a manual review, an independent adversarial review, regression tests for every finding and mutation testing of
those tests); see `docs/audit.md` for the findings (3 critical in the first version, all fixed), the argument for each component and the limits.
Supported ranges: scaler exponents in [-122, 127] (every finite f32), `K <= 8192`, at most 64 layers; the verifier takes the layers and activations from its own model.

The circuits are documented in full (every column, constraint and lookup, with worked examples) in `lib/tt-zkml/docs/zk-circuits.pdf`
(source `zk-circuits.typ`, build `typst compile docs/zk-circuits.typ docs/zk-circuits.pdf` from `lib/tt-zkml/`; the numbers of the examples come from
`cargo run --release --example circuit_examples`).

Tests: `cargo test` (the library tests need no device and include forged witnesses and proofs; `tests/proof.rs` is the end-to-end test: it builds a
network, runs it on the device, rebuilds the witness from the trace, proves and verifies, then checks that forged traces, inputs, proofs and statements are
refused; `tests/abi.rs` tests the device API; `tests/audit.rs` has a regression test for every finding of the soundness audit, written as a malicious
prover that rewrites the columns of a component; `tests/forgery_fuzz.rs` is a randomized forgery test: random networks, random forged witnesses and columns,
and it fails if the verifier ever accepts a statement other than the true one; `FUZZ_ITERS=2000 FUZZ_SEED=7 cargo test --release --test forgery_fuzz` runs it longer).
`tests/row_exhaustive.rs` runs the real constraint code on single rows for every choice a malicious prover has, over small domains, and requires that
every accepted row is a true one (`ROW_FULL=1 cargo test --release --test row_exhaustive` for the full domain).
The `forgery` cargo feature (enabled by `cargo test`) only exposes the internals these tests need. Needs nightly Rust (`rust-toolchain.toml`).
