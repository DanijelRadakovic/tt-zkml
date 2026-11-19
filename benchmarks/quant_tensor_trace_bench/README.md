# Quantizer and pot_quant_m31 on the device: wall-clock benchmark

Times `pot_quantize_per_tensor` and `pot_quant_m31` on one Tensix core, with the input already in device memory. The CPU
counterpart is `tools/stwo-cpu-simd`; the two are joined in `tools/stwo-cpu-simd/results/comparison.md`. (The directory
is called `trace_bench` because it started as a benchmark of a device-side trace, which was dropped: the host derives the
rest of the prover witness, see the root `README.md`.)

```bash
.build/default/zkml_quant_trace_bench                       # tiles 16 64 256 1024 4096, bits 8 16 32
.build/default/zkml_quant_trace_bench --tiles 256 --csv out.csv
```

## What is measured

BFLOAT16 input of `32 x 32*tiles` elements (1024 per tile), INT32 output, `QuantRange::M31`. (The `quantize` numbers in the
table below were measured before the fused SFPU kernels, when 16 and 32 bits used the FPU multiply and composed ops; the quantizer is
now faster, see `benchmarks/quant_tensor_fpu_vs_sfpu/results/lreg7_scaler.md`.) Per call: n calls back to back with the program cache warm, then one
`Finish` on the command queue; the table shows the median of 5 such runs divided by n. The time is wall clock, so it includes
the host-side dispatch of the op (about 10 us), which matters at 16 tiles.

| Column | Operation |
|---|---|
| quantize | `pot_quantize_per_tensor`: `q` and the scaler |
| to_m31 | `pot_quant_m31(q)` |
| quantize+m31 | the two together: the device's time in the CPU comparison |
| readback of m31 | `to_vector<uint32_t>()` of the M31 tensor: what copying the result to the host costs |

## Results

`results/device.md` (`device.csv` has the same numbers per op, with Melem/s and ns per tile). Machine: Blackhole p100a,
one core at 1350 MHz, tt-metal as installed in the IRD container. Every time is in milliseconds for the whole tensor of
`tiles` tiles (1 tile = 1024 elements, so 4096 tiles are 4 million elements); divide by the tile count for the time per tile.

| bits | tiles | quantize ms | to_m31 ms | quantize+m31 ms | readback of m31 ms |
| 8 | 16 | 0.012 | 0.010 | 0.022 | 0.025 |
| 8 | 64 | 0.036 | 0.031 | 0.067 | 0.078 |
| 8 | 256 | 0.133 | 0.113 | 0.247 | 0.220 |
| 8 | 1024 | 0.524 | 0.443 | 0.967 | 1.981 |
| 8 | 4096 | 2.083 | 1.763 | 3.846 | 7.974 |
| 16 | 16 | 0.023 | 0.010 | 0.034 | 0.024 |
| 16 | 64 | 0.083 | 0.031 | 0.114 | 0.081 |
| 16 | 256 | 0.322 | 0.113 | 0.436 | 0.223 |
| 16 | 1024 | 1.279 | 0.444 | 1.722 | 0.857 |
| 16 | 4096 | 5.104 | 1.763 | 6.867 | 8.094 |
| 32 | 16 | 0.023 | 0.010 | 0.033 | 0.023 |
| 32 | 64 | 0.080 | 0.031 | 0.111 | 0.077 |
| 32 | 256 | 0.310 | 0.114 | 0.423 | 0.213 |
| 32 | 1024 | 1.228 | 0.443 | 1.672 | 0.860 |
| 32 | 4096 | 4.903 | 1.763 | 6.666 | 7.928 |

## Reading the numbers

- `quantize` takes 0.51 ms per 1024 tiles at 8 bits and 1.2 to 1.3 ms per 1024 tiles at 16 and
  32 bits; the composed round, clamp and convert ops are the difference (see `tools/quant_tensor_fpu_vs_sfpu`).
- `pot_quant_m31` is a plain read-modify-write stream: 0.43 us per tile (0.44 ms per 1024 tiles), the same at every width.
- **Copying to the host** costs 8 ms for 4096 tiles (16 MB): 2.1 times the whole quantize+m31 at 8 bits (3.8 ms) and
  1.2 times at 16 and 32 bits (6.7 ms). The M31 tensor is 4 bytes per element at every width. So producing the residues on the
  device pays when the next step also runs there.
