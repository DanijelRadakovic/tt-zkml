# stwo-cpu-simd: CPU and SIMD reference of the quantization and its prover witness

> **Note:** the device numbers in the comparison tables below were measured before the fused SFPU quantizer kernels and the batched readers and writers. The device side is now faster, for example quantize + M31 at 256 tiles: 0.166 ms (8 bits) against 0.247 ms in the table (`../quant_tensor_trace_bench/results/device.md`). The CPU numbers are unchanged.


Rust crate that computes on the CPU what `pot_quantize_per_tensor` and `pot_quant_m31` produce on the device, and the
rest of the witness a prover needs, which the **host** derives (the device does not):

```
Q = 2^(bits - 1) - 1            (QuantRange::Full; QuantRange::M31 caps 32 bits at 2^30 - 1, see the root README)
k = max{k <= 127 : absmax(x) * 2^k <= Q}
q = clamp(round_half_even(x * 2^k), -Q, Q)              device: pot_quantize_per_tensor
m31 = q mod (2^31 - 1)                                  device: pot_quant_m31
remainder = x * 2^k - q         exact in fp32, |remainder| <= 1/2                      host only
argmax    = index of the first element (in tile order) with |x| = absmax, into the     host only
            row-major tile-padded tensor
```

It has three jobs: the oracle the device is checked against, the baseline for "how long does a CPU take for the same job",
and the place where the host-side witness (remainder, argmax) is derived and timed. The decision behind the last one:
the remainder and the argmax follow from `(x, q, k)`, which the host has anyway, they cost 0.1 to 0.4 ns per element there
(the `+witness` columns below), and computing them on the device cost 4.4 to 11 times the quantization itself.

## Contents

| Path | What |
|---|---|
| `src/reference.rs` | Scalar oracle in f64 / i64, written for clarity. |
| `src/simd.rs` | `std::simd` (16 lanes of f32), single thread or a rayon pool. `k` comes from the exponent bits as on the device (not from the oracle's search), rounding is the `(\|y\| + 2^23) - 2^23` trick, the residues go into stwo's `BaseColumn` of `PackedM31` through `m31_column`. |
| `src/range.rs` | `QuantRange` (`Full`, `M31`), `Q`, the M31 map and its inverse, the tile-order rank (same rules as `pot_range.hpp`). |
| `src/bin/bench.rs` | The benchmark. |
| `src/bin/dump.rs` | Writes reference cases (inputs, `q`, remainder, residues, `k`, argmax) for the device test, which compares `q`, `k` and the residues. |
| `tests/agree.rs` | Oracle against SIMD on 7 shapes x 5 magnitudes x 3 widths, with 1 and 4 threads; `k` by search against `k` by exponent bits on 60 000 values; the prover's invariants; the tile-order argmax; the M31 column. |
| `compare.py` | Joins `results/cpu_simd.csv` with the device benchmark into `results/comparison.md`. |

stwo is used as a git dependency on the `starkware-libs/proving` repository (the successor of `starkware-libs/stwo`),
pinned to commit `c87ea10`. It needs nightly Rust (`portable_simd`): `rust-toolchain.toml` pins `nightly-2026-01-15`, the
same as stwo. `.cargo/config.toml` builds with `-C target-cpu=native`, so stwo's SIMD backend uses AVX-512 where the CPU has it.

## Run

```bash
cd tools/stwo-cpu-simd
cargo test --release
cargo run --release --bin bench -- --csv results/cpu_simd.csv [--tiles 16,64,256,1024,4096] [--threads N]
python3 compare.py                      # needs ../quant_tensor_trace_bench/results/device.csv

# bit-exact check of the device against this crate (on the machine with the device); writes both ranges:
cargo run --release --bin dump -- /some/dir
ZKML_RUST_REF_DIR=/some/dir .build/default/test_pot_quant_m31 --gtest_filter='*CrossCheckWithRust'
```

## Results

Machine: AMD Ryzen 7 9700X (8 cores, 16 threads, AVX-512), idle, one run. Input: `32 x 32*tiles` bf16-exact values,
uniform in [-8, 8), quantized with `QuantRange::M31` (it differs from `Full` only at 32 bits and costs the same). Each time
is the median of at least 5 runs. Before a size is timed, the benchmark asserts that the oracle, `simd-1` and `simd-16`
return identical results.

### CPU alone

Milliseconds for the whole tensor of `tiles` tiles (1 tile = 1024 elements). `q+m31` is what the device does (absmax, `k`,
quantize, M31 residues); `+witness` adds the remainder and the argmax. `simd-1` is one thread, `simd-16` is 16 threads (the
argmax scan is not parallel, so `+witness` scales less well). 'column build' (packing the residues into stwo's
`BaseColumn`) is separate. The scalar oracle always includes the witness and is skipped at 4096 tiles.

| bits | tiles | scalar +witness ms | simd-1 q+m31 ms | simd-16 q+m31 ms | simd-1 +witness ms | simd-16 +witness ms | column build ms |
|---|---|---|---|---|---|---|---|
| 8 | 16 | 0.032 | 0.012 | 0.016 | 0.019 | 0.026 | 0.003 |
| 8 | 64 | 0.153 | 0.081 | 0.083 | 0.106 | 0.128 | 0.010 |
| 8 | 256 | 0.652 | 0.351 | 0.146 | 0.447 | 0.237 | 0.043 |
| 8 | 1024 | 2.715 | 1.399 | 0.529 | 1.841 | 0.805 | 0.175 |
| 8 | 4096 | - | 7.426 | 2.355 | 10.339 | 3.995 | 0.798 |
| 16 | 16 | 0.025 | 0.012 | 0.016 | 0.013 | 0.017 | 0.003 |
| 16 | 64 | 0.103 | 0.048 | 0.046 | 0.053 | 0.054 | 0.011 |
| 16 | 256 | 0.424 | 0.201 | 0.053 | 0.222 | 0.084 | 0.048 |
| 16 | 1024 | 1.724 | 0.924 | 0.279 | 1.067 | 0.512 | 0.218 |
| 16 | 4096 | - | 7.519 | 2.331 | 10.398 | 3.982 | 0.962 |
| 32 | 16 | 0.026 | 0.012 | 0.017 | 0.013 | 0.020 | 0.003 |
| 32 | 64 | 0.104 | 0.049 | 0.045 | 0.053 | 0.046 | 0.010 |
| 32 | 256 | 0.429 | 0.204 | 0.054 | 0.225 | 0.105 | 0.048 |
| 32 | 1024 | 1.727 | 0.916 | 0.277 | 1.070 | 0.510 | 0.233 |
| 32 | 4096 | - | 7.472 | 2.310 | 10.275 | 3.992 | 1.023 |

### CPU against the device: which is faster

Both sides do the same job on the same tensor: the values `x` in, the quantized values `q`, the scaler and the M31 residues
out. The device runs `pot_quantize_per_tensor` followed by `pot_quant_m31` on one Tensix core (of 110), with `x` already
in device memory; the CPU has `x` in host memory. Every time is for the whole tensor; the number in parentheses is
microseconds per tile (the time divided by the number of tiles), so the sizes can be compared. Lower is faster. 'fastest'
names the lowest of the first three columns.

#### 8 bits

| tiles | CPU 1 thread | CPU 16 threads | device | fastest | device vs CPU 1 thread | CPU + remainder and argmax: 1 thread / 16 threads | copy M31 to host |
|---|---|---|---|---|---|---|---|
| 16 | 0.012 ms (0.75 us/tile) | 0.016 ms (1.01 us/tile) | 0.022 ms (1.37 us/tile) | CPU 1 thread | 1.8x slower | 0.019 / 0.026 ms | 0.025 ms |
| 64 | 0.081 ms (1.26 us/tile) | 0.083 ms (1.29 us/tile) | 0.067 ms (1.04 us/tile) | device | 1.2x faster | 0.106 / 0.128 ms | 0.078 ms |
| 256 | 0.351 ms (1.37 us/tile) | 0.146 ms (0.57 us/tile) | 0.247 ms (0.96 us/tile) | CPU 16 threads | 1.4x faster | 0.448 / 0.237 ms | 0.220 ms |
| 1024 | 1.399 ms (1.37 us/tile) | 0.529 ms (0.52 us/tile) | 0.967 ms (0.94 us/tile) | CPU 16 threads | 1.4x faster | 1.841 / 0.805 ms | 1.981 ms |
| 4096 | 7.426 ms (1.81 us/tile) | 2.355 ms (0.57 us/tile) | 3.846 ms (0.94 us/tile) | CPU 16 threads | 1.9x faster | 10.339 / 3.995 ms | 7.974 ms |

#### 16 bits

| tiles | CPU 1 thread | CPU 16 threads | device | fastest | device vs CPU 1 thread | CPU + remainder and argmax: 1 thread / 16 threads | copy M31 to host |
|---|---|---|---|---|---|---|---|
| 16 | 0.012 ms (0.75 us/tile) | 0.016 ms (0.97 us/tile) | 0.034 ms (2.11 us/tile) | CPU 1 thread | 2.8x slower | 0.013 / 0.017 ms | 0.024 ms |
| 64 | 0.048 ms (0.75 us/tile) | 0.046 ms (0.72 us/tile) | 0.114 ms (1.78 us/tile) | CPU 16 threads | 2.4x slower | 0.053 / 0.054 ms | 0.081 ms |
| 256 | 0.201 ms (0.79 us/tile) | 0.053 ms (0.21 us/tile) | 0.436 ms (1.70 us/tile) | CPU 16 threads | 2.2x slower | 0.222 / 0.084 ms | 0.223 ms |
| 1024 | 0.924 ms (0.90 us/tile) | 0.279 ms (0.27 us/tile) | 1.722 ms (1.68 us/tile) | CPU 16 threads | 1.9x slower | 1.067 / 0.512 ms | 0.857 ms |
| 4096 | 7.519 ms (1.84 us/tile) | 2.331 ms (0.57 us/tile) | 6.867 ms (1.68 us/tile) | CPU 16 threads | 1.1x faster | 10.398 / 3.982 ms | 8.094 ms |

#### 32 bits

| tiles | CPU 1 thread | CPU 16 threads | device | fastest | device vs CPU 1 thread | CPU + remainder and argmax: 1 thread / 16 threads | copy M31 to host |
|---|---|---|---|---|---|---|---|
| 16 | 0.012 ms (0.76 us/tile) | 0.017 ms (1.04 us/tile) | 0.033 ms (2.06 us/tile) | CPU 1 thread | 2.7x slower | 0.013 / 0.021 ms | 0.023 ms |
| 64 | 0.049 ms (0.76 us/tile) | 0.045 ms (0.70 us/tile) | 0.111 ms (1.74 us/tile) | CPU 16 threads | 2.3x slower | 0.053 / 0.046 ms | 0.077 ms |
| 256 | 0.204 ms (0.80 us/tile) | 0.054 ms (0.21 us/tile) | 0.423 ms (1.65 us/tile) | CPU 16 threads | 2.1x slower | 0.225 / 0.105 ms | 0.213 ms |
| 1024 | 0.916 ms (0.89 us/tile) | 0.277 ms (0.27 us/tile) | 1.672 ms (1.63 us/tile) | CPU 16 threads | 1.8x slower | 1.070 / 0.510 ms | 0.860 ms |
| 4096 | 7.472 ms (1.82 us/tile) | 2.310 ms (0.56 us/tile) | 6.666 ms (1.63 us/tile) | CPU 16 threads | 1.1x faster | 10.275 / 3.992 ms | 7.928 ms |

How to read it:
- **8 bits:** the device takes 0.9 to 1.4 us per tile. It is faster than one CPU thread from 64 tiles on (1.2 to 1.9 times),
  and 1.7 to 1.8 times slower than 16 threads from 256 tiles on. It is 1.8 times slower at 16 tiles.
- **16 and 32 bits:** the device takes 1.6 to 2.1 us per tile. One CPU thread takes 0.75 to 0.9 us up to 1024 tiles, so the device
  is 1.8 to 2.8 times slower. At 4096 tiles the CPU thread slows to 1.8 us per tile (probably memory-bound), and the device is
  1.1 times faster. 16 threads take 0.2 to 0.3 us per tile at 256 to 1024 tiles (6 to 8 times faster than the device) and
  0.57 at 4096 tiles (2.9 times faster).
- **Small tensors (16 tiles):** one CPU thread is fastest, and 16 threads are slower than one because starting them costs more
  than the work.
- **Deriving the rest of the witness on the host** (the remainder and the argmax) adds about 15% at 16 and 32 bits and 30% at
  8 bits for one thread (1024 tiles: 0.92 to 1.07 ms, 1.40 to 1.84 ms). With 16 threads it nearly doubles the time, because the
  argmax scan is sequential; that can be parallelized.
- **Copying instead of computing on the device:** copying only the M31 tensor to the host takes 8 ms at 4096 tiles (last
  column), about as much as the device needs to produce it at 16 and 32 bits (6.7 ms) and twice as much at 8 bits (3.8 ms).
- Not measured: power of the CPU run, and the CPU cost of building the full AIR trace (this crate stops at the M31
  columns of `q`). The device multi-core version is not built, so the 109 idle cores are the main headroom.
