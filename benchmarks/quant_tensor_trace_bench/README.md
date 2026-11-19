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
| 8 | 16 | 0.008 | 0.007 | 0.014 | 0.024 |
| 8 | 64 | 0.024 | 0.021 | 0.045 | 0.078 |
| 8 | 256 | 0.088 | 0.078 | 0.166 | 0.224 |
| 8 | 1024 | 0.346 | 0.307 | 0.652 | 1.935 |
| 8 | 4096 | 1.375 | 1.221 | 2.595 | 4.838 |
| 16 | 16 | 0.008 | 0.007 | 0.014 | 0.024 |
| 16 | 64 | 0.024 | 0.021 | 0.045 | 0.075 |
| 16 | 256 | 0.088 | 0.078 | 0.166 | 0.215 |
| 16 | 1024 | 0.346 | 0.307 | 0.652 | 0.868 |
| 16 | 4096 | 1.375 | 1.221 | 2.595 | 8.043 |
| 32 | 16 | 0.015 | 0.007 | 0.022 | 0.024 |
| 32 | 64 | 0.053 | 0.021 | 0.074 | 0.078 |
| 32 | 256 | 0.204 | 0.078 | 0.282 | 0.315 |
| 32 | 1024 | 0.808 | 0.307 | 1.114 | 1.087 |
| 32 | 4096 | 3.224 | 1.221 | 4.444 | 7.550 |

## Reading the numbers

- `quantize` takes 0.35 ms per 1024 tiles at 8 and 16 bits and 0.81 ms at 32 bits (fused SFPU sweeps with the scaler in
  LReg7, batched writer; `benchmarks/quant_tensor_fpu_vs_sfpu/results/lreg7_scaler.md`). It was 0.52, 1.28 and 1.23 ms before.
- `pot_quant_m31` takes 0.31 ms per 1024 tiles (0.30 us per tile), the same at every width. It was 0.44 ms with one DRAM
  round trip (a barrier) per tile in the reader and in the writer; reader and writer now move a compute block (4 tiles)
  behind one barrier. Batching only the writer changed nothing (0.443 ms): the reader's per-tile barrier was the next
  limit, so both are batched. Larger batches (8 tiles, ring of 16) gave no further gain (0.304 ms): at about 8 KB per tile
  moved (4 KB read, 4 KB written) the one core is bound by its DRAM bandwidth, not by the SFPU.
- **Copying to the host** costs 4 to 8 ms for 4096 tiles (16 MB; it varies between runs), 1.7 to 3.2 times the whole quantize+m31
  at 8 bits (2.6 ms). The M31 tensor is 4 bytes per element at every width. So producing the residues on the
  device pays when the next step also runs there.
