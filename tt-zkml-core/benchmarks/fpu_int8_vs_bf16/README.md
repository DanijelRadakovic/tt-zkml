# int8 against bf16 matmul on the FPU

Which way of feeding quantized values to the Tensix matrix engine (FPU) is fast, and which is exact? The quantized
tensors hold integers in `[-127, 127]`. The FPU can multiply them in two ways:

| path | CB data format of A and B | FPU mode | accumulator in `Dst` | output CB |
|---|---|---|---|---|
| **int8** | `Int8` | integer math (`ALU_ACC_CTRL_INT8_math_enabled`) | Int32 | `Int32` |
| **bf16** | `Float16_b` (the integers stored as bf16 values) | floating-point math | FP32 | `Float32` |

For each path, math fidelity (`LoFi` to `HiFi4`), K and data pattern the benchmark measures the throughput and checks
the result against an exact int64 matmul on the host. It is the experiment behind the choice of the matmul path in
`docs/zkml_layer_execution.pdf` (the prover compares `acc` with the exact sum in the field, so a rounded `acc` fails the proof).

## Run

```bash
cmake --preset default && cd .build/default && ninja zkml_fpu_int8_vs_bf16_bench && cd ../..
.build/default/zkml_fpu_int8_vs_bf16_bench                                   # everything, about 1 minute
.build/default/zkml_fpu_int8_vs_bf16_bench --kt 32,128 --fidelity hifi4 --mode block --verbose
.build/default/zkml_fpu_int8_vs_bf16_bench --csv benchmarks/fpu_int8_vs_bf16/results/fpu_int8_vs_bf16.csv
```

| Option | Meaning | Default |
|---|---|---|
| `--kt` | K in tiles of 32 (K = 32 * kt) | `1,4,16,32,64,128` (K = 32 to 4096) |
| `--fidelity` | `lofi,hifi2,hifi3,hifi4` | all four |
| `--mode` | `tiles` = `matmul_tiles`, `block` = `matmul_block` (what ttnn uses) | both |
| `--twos` | write the int8 operands as two's complement bytes (the quantizer's output encoding) instead of sign-magnitude | off |
| `--verbose` | on a mismatch, print the first device outputs next to the exact ones | off |
| `--csv` | write every row to a CSV file | off |

Needs `TT_METAL_HOME` / `TT_METAL_RUNTIME_ROOT` and a device, like the other benchmarks. No profiler is needed. The compute
kernel has a `DeviceZoneScopedN("MATMUL")` zone, so `TT_METAL_DEVICE_PROFILER=1` also gives cycles if wanted.

## What is measured

- **One core, one output block** of 2 x 2 tiles (64 x 64 outputs) with the whole K accumulated in `Dst`. A (2 x KT tiles) and B
  (KT x 2 tiles) are copied to L1 once and the block is recomputed `reps` times, with the pack of the result in every repetition.
  So the number is the matrix engine plus the unpacker for operands that are already in L1. DRAM, the NoC and the other 109
  cores are not part of it.
- **Time:** wall clock around the workload at `reps` and `4 * reps` (about 20 ms and 80 ms); the difference divided by
  `3 * reps` removes the launch, the DRAM-to-L1 copy and the write-back. Median of 5. Reported as ns per 32x32x32 tile matmul and
  as GMAC/s (`64 * 64 * K` multiply-adds per block).
- **Exactness:** after one repetition the 64 x 64 output is compared with the exact matmul. Data patterns:

| pattern | A | B | why |
|---|---|---|---|
| `random` | uniform in [-127, 127] | uniform in [-127, 127] | what 8-bit data looks like |
| `medium` | [-45, 45] | [-45, 45] | products below 2^11 |
| `small` | [-31, 31] | [-31, 31] | products below 2^10 |
| `narrow` | [-31, 31] | [-15, 15] | the range for which the int8 mode needs one fidelity phase (ISA docs) |
| `constant` | 127 | 127 | every product is 127 * 127; a degenerate case (all partial sums are multiples of 16 * 16129), not a worst case |

- **Layouts built on the host:** tiles of 4 faces of 16 x 16; `Int8` tiles are sign-magnitude bytes (`0x80 | magnitude`), as the
  unpacker expects (ISA docs, `Unpackers/FormatConversion.md`); bf16 is the top half of the fp32 bits.
  The output of the int8 path is tried as two's complement and as sign-magnitude, and the better match is reported in the CSV.

## Results

Full tables: `results/fpu_int8_vs_bf16.md` (from `results/fpu_int8_vs_bf16.csv`, 480 rows). Machine: Blackhole p100a, one core at
1350 MHz, tt-metal as installed in the IRD container.

Time per 32x32x32 tile matmul at K >= 512 (`matmul_block`); int8 and bf16 are the same within 2%:

| fidelity | ns per tile | GMAC/s per core | exact for |
|---|---|---|---|
| LoFi | 13.0 | 2525 | bf16: only A, B in [-31, 31]; int8: never |
| HiFi2 | 25.6 | 1281 | bf16: A, B in [-45, 45]; int8: never |
| HiFi3 | 37.5 | 876 | same as HiFi2 |
| HiFi4 | 49.3 | 665 | **int8: all of [-127, 127]**; bf16: A, B in [-45, 45] |

`matmul_tiles` is slower than `matmul_block` only at LoFi (21.1 ns against 13.0 ns); at higher fidelity both are bound by the
matrix engine.

### Which path for which K (full int8 range, A and B in [-127, 127], `block` mode)

Time of the whole 64x64 output block in ns (K = inner dimension), and how many of its 4096 outputs were wrong on random data. "INT8 math on" is the `int8` path (Int8 buffers); "off" is the `bf16` path (Float16_b buffers).

| K | INT8 math on, HiFi4 | INT8 math off (bf16), HiFi4 | off, HiFi2 | off, LoFi |
|---|---|---|---|---|
| 32 | 207 ns, exact | 207 ns, **3690 wrong** | 140 ns, **3690 wrong** | 138 ns, **4092 wrong** |
| 128 | 798 ns, exact | 798 ns, **3900 wrong** | 418 ns, **3900 wrong** | 217 ns, **4094 wrong** |
| 512 | 3160 ns, exact | 3162 ns, **4017 wrong** | 1644 ns, **4017 wrong** | 839 ns, **4096 wrong** |
| 1024 | 6314 ns, exact | 6314 ns, **4020 wrong** | 3279 ns, **4020 wrong** | 1668 ns, **4096 wrong** |
| 2048 | 12622 ns, exact | 12618 ns, **4028 wrong** | 6549 ns, **4028 wrong** | 3326 ns, **4095 wrong** |
| 4096 | 25226 ns, exact | 25225 ns, **4067 wrong** | 13097 ns, **4067 wrong** | 6649 ns, **4095 wrong** |

The result for every K is the same: only INT8 math on with HiFi4 is exact for data in [-127, 127], and it costs the same time as the (wrong) bf16 HiFi4 run. The choice depends on how many bits the operands have, not on K:

| operand range | cheapest exact path | ns per tile (K >= 512) |
|---|---|---|
| A, B in [-127, 127] (8 bit) | INT8 math on, HiFi4 | 49.3 |
| A, B in [-45, 45] (about 6 bit) | INT8 math off (bf16), HiFi2 | 25.6 |
| A, B in [-31, 31] (5 bit) | INT8 math off (bf16), LoFi | 13.0 |

## What it shows

1. **int8 is exact only at HiFi4.** At LoFi, HiFi2 and HiFi3 every int8 run was wrong, also for the `small` and `narrow` data. HiFi4
   (4 phases) gave 0 wrong outputs of 4096 in every run (K up to 4096, all patterns).
2. **bf16 is not exact for the full int8 range, at any fidelity.** With random A, B in [-127, 127] and HiFi2 to HiFi4, 3690 of 4096 outputs differ
   already at K = 32 (4067 at K = 4096), by up to 15 (153 at K = 4096). The cause is not the number of fidelity phases (the
   `constant` data with 127 is exact from HiFi2) and not the 2^24 limit of the fp32 accumulator. The ISA documentation
   (`MatrixUnitArithmetic.md`) says a lane sums its 8 products in fixed point aligned to the largest exponent and keeps only 10
   fractional bits, so small products lose their low bits when a large product is in the same lane. With products below 2^11
   (`medium`, `small`) the results were exact for every K up to 4096. The exact threshold between 45 and 127 was not measured.
3. **The two paths run the same instructions, so they take the same time.** There is no throughput reason to prefer bf16 at full
   fidelity. bf16 could use fewer phases (down to LoFi, 3.8 times faster than HiFi4) only if the quantization is limited to about 5
   bits (`[-31, 31]`).
4. **The int8 path's Int32 output is in sign-magnitude in L1** (matched in all 12 hifi4 `random` runs; `constant` data is
   non-negative, so it matches both). A ttnn `INT32` tensor is two's complement, so a conversion is needed before the result is used as one.
5. **At about 13 ns for LoFi the loop runs at about one matrix instruction per cycle** (16 per tile at 1350 MHz is 11.9 ns, plus overhead),
   so the numbers are the matrix engine's, not the unpacker's. If all 110 cores kept this rate, which was not measured, HiFi4 would give
   about 73 TMAC/s, 146 int8 TOPS. Data movement and the cross-core reduction for the requantize would reduce that.

## The LLK questions

- **What turns on INT8 math?** No compute-kernel flag does. `_llk_math_hw_configure_` (`tt_llk_blackhole/llk_lib/llk_math_common.h`,
  line 37) sets `ALU_ACC_CTRL_INT8_math_enabled` when the SrcA or SrcB data format is `Int8` or `Int32` (`is_int8_or_int32_format`
  in `ckernel_defs.h`). It is called by `compute_kernel_hw_startup`, so the formats of the input CBs decide. The reconfig functions
  assert `fp32_dest_acc_en`, so the kernel needs `ComputeConfig{.fp32_dest_acc_en = true}`.
- **Does the unpacker feed int8 from L1 straight into SrcA and SrcB?** Yes: `Int8` (sign-magnitude, +-127) becomes the "Integer 8"
  type in SrcA and SrcB, with no conversion step in the kernel. In the JIT (`jit_build/data_format.cpp`) the unpack format of an
  `Int8` CB stays `Int8`. The hardware could also convert `INT8` to BF16 on unpack (`Unpackers/FormatConversion.md`), but the
  standard configuration does not select it, and I did not try to override it.
- **Does the quantizer's int8 output match the FPU's `Int8` input? No.** `pot_quantize_per_tensor` with INT8 output writes
  two's complement bytes through the `UInt8` packer (documented in `quantize.cpp`; the tests read it back as `int8_t`). The FPU
  expects sign-magnitude. `--twos` feeds the benchmark with the quantizer's encoding: at HiFi4 every one of the 4096 outputs was
  wrong for the `random`, `medium`, `small` and `narrow` data, and only `constant` (all +127) stayed exact, because non-negative
  values have the same byte in both encodings. A negative `v` is `256 + v` in two's complement and `128 + |v|` in sign-magnitude
  (-1 is `0xFF` against `0x81`; -40 is `0xD8` against `0xA8`), and the FPU reads `0xD8` as -88. So the quantizer's int8
  output cannot be used as a matmul operand as it is. Options: pack the quantizer's int8 output as `Int8` (sign-magnitude), or convert
  two's complement to sign-magnitude in an extra pass. Nothing was changed in the quantizer.

## Limits

One core, operands resident in L1, one 2 x 2 block, no operand streaming and no cross-core traffic; K above 4096 and non-square blocks were
not run. `Dst` holds the 4 output tiles in FP32 mode; larger blocks would need more of it. Power was not measured. The matmul
is not yet wired into any operation.

## Files

| File | Purpose |
|---|---|
| `bench.cpp` | host: data, tilize, program setup, timing, exact check |
| `kernels/compute.cpp` | the matmul loop (`matmul_tiles` or `matmul_block`), same code for both paths |
| `kernels/reader.cpp`, `kernels/writer.cpp` | DRAM to L1 once, and the result back |
| `results/` | CSV and the generated tables |
