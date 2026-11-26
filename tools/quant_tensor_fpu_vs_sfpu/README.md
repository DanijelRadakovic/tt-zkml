# quant_tensor_fpu_vs_sfpu

**Input type.** All measurements use a BFLOAT16 input tile, on both units. The FPU cannot take FLOAT32 (it keeps
19 bits of an operand), so there is no FPU result to compare an fp32 SFPU run with; no FLOAT32 row was measured.

**Question.** `pot_quantize_per_tensor` runs everything on the SFPU (the vector unit). A BFLOAT16 input has only
16 bits, and the FPU (the matrix unit) keeps 19 bits of an operand, so it should be able to do part of the job
without losing precision, and the output can stay fp32 in the DST. Is the FPU faster, and does it use less power?

**Answer (this card, one Blackhole p100a, one core).**

| | |
|---|---|
| **Multiply by the scaler, 16 and 32 bits** | The FPU is **about 10 % faster** end to end (5.5 % at 16 tiles, 10 – 10.7 % at 256 tiles and more; wall time agrees) and uses **8.5 – 9.5 % less energy**. The result is bit-identical to the SFPU. |
| **Multiply, 8 bits** | The FPU is **2.3 – 2.6x slower** at 1024 tiles and uses 2.1 – 2.4x more energy, because the SFPU has one fused op (`quant_tile`) that the FPU variant has to replace with three or four separate SFPU passes. `Auto` keeps the SFPU here. |
| **Finding the max (pass 1)** | **Not moved to the FPU.** The FPU max pool is very cheap (29 cycles per tile against 231 for the SFPU absmax), but the FPU has no abs and no min, so it can only give `max(x)`, not `max(abs(x))`. Getting the other half needs a DST to SrcA move that tt-metal does not expose. Details and measurements in "What the FPU can and cannot do". |

The gain is small because the multiply is a small part of a pass-2 tile (about 200 of 1650 cycles, see the
breakdown). Pass 1 (SFPU, 230 cycles per tile) and the SFPU round / clamp / convert passes plus the writer
dominate.

## Contents

| File | What it is |
|---|---|
| `bench.cpp` | Host program, three modes: `profile` (device cycles), `wall` (host time), `power` (sustained multi-core run with the power sensor). Runs the **real kernels** of the op, with the same CBs as the program factory. Checks every run against the host reference. |
| `run_all.sh` | Runs the matrix: `cycles`, `wall`, `power` or `all`. |
| `analyze.py` | Device-profiler logs to `results/cycles.csv` and `cycles.md`. |
| `summarize.py` | `cycles.csv`, `wall.txt`, `power.txt` to `results/summary.md`. |
| `probe/` | A small probe for the FPU primitives (max pool, `mul_tiles`): what they compute and what they cost. Used for "What the FPU can and cannot do". |
| `results/raw/prof_*.csv` | **The raw device-profiler logs**, one per configuration (11 runs each). |
| `results/wall.txt`, `results/power.txt` | Raw one-line-per-run output of the wall and power runs. |
| `results/primitives.txt` | Raw output of the probe. |
| `results/cycles.csv`, `results/summary.md` | Reduced numbers (the tables below). |

## What changed in the op

- The compute kernel (`.../kernels/compute/quantize.cpp`) has a compile-time arg `use_fpu`. With it, pass 2 does
  `mul_tiles(c_0, c_1, i, 0, i)` (SrcA = the bf16 input, SrcB = the scaler tile, a power of two) instead of
  `copy_tile` + scaler copy + `mul_binary_tile`. The product is exact in the fp32 DST. The scaler CB `c_1` is then
  unpacked to SrcB (not to DST), no DST slot is kept for the scaler, so a block is 4 tiles instead of 3 and `c_16` has 8 pages.
  The rest of pass 2 (round, clamp, convert) stays on the SFPU, always as separate ops. For INT8 output the byte has
  to be masked (`& 0xFF`): the UInt8 packer **saturates**, so a negative int32 would come out as 255.
  (`quant_int8_tile` leaves the byte, 256 + q for a negative q, in the DST for the same reason.)
- The op got a knob `ComputeUnit compute_unit` (`Auto`, `Sfpu`, `Fpu`; in `operations/pot_quantization/compute_unit.hpp`).
  `Auto` is the FPU for BFLOAT16 at 16 or 32 bits and the SFPU otherwise. `Fpu` on a FLOAT32 input throws (the FPU
  would cut the input to 19 bits).
- The kernel has profiler zones `PASS1`, `SCALER`, `PASS2` (no code unless `TT_METAL_DEVICE_PROFILER=1`).
- Tests (`test_pot_quantize_per_tensor`) run BFLOAT16 on both units for every width, tile count and batch sweep.

## How to run

Everything needs the device. On a lab machine the tools live in the IRD container, so run these inside it.

```bash
export TT_METAL_HOME=/proj_sw/user_dev/ctr-dradakovic/tt-metal
cmake --build .build/default --target zkml_quant_fpu_vs_sfpu_bench zkml_fpu_probe
tools/quant_tensor_fpu_vs_sfpu/run_all.sh all       # cycles ~35 min, wall ~7 min, power ~15 min
python3 -I tools/quant_tensor_fpu_vs_sfpu/analyze.py   tools/quant_tensor_fpu_vs_sfpu/results
python3 -I tools/quant_tensor_fpu_vs_sfpu/summarize.py tools/quant_tensor_fpu_vs_sfpu/results
```

One configuration by hand:

```bash
rm -f $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
TT_METAL_DEVICE_PROFILER=1 .build/default/zkml_quant_fpu_vs_sfpu_bench profile bf16 fpu 16 int32 256 11
#                                                                      mode   in   unit bits out tiles iters
.build/default/zkml_quant_fpu_vs_sfpu_bench wall  bf16 sfpu 16 int32 256 50
.build/default/zkml_quant_fpu_vs_sfpu_bench power bf16 fpu  16 int32 1024 16 15     # ... tiles cores seconds
```

## How it is measured

**Cycles (device).** The device profiler timestamps the kernels on the chip's own counter (1350 MHz here), so
dispatch and launch latency are not in the numbers. `bench profile` runs the program 11 times back to back on core
(0,0); run 0 (JIT, cold caches) is dropped and the median of the other 10 is used (the spread of the kernel time
is 0.09 % in the median configuration and at most 1.2 %; column `spread_pct` in `cycles.csv`, so the differences of 5 % and more are real). `kernel` is the earliest `*-KERNEL` start to the latest end
on the core; `pass1`, `scaler`, `pass2` are the kernel's own zones on the math RISC; `reader` and `writer` are the
data-movement RISCs. The whole matrix (4 widths/outputs x 4 sizes x 2 units) is in `cycles.csv`.

**Wall time (host).** The same program without the profiler. *Latency* is enqueue + finish of one run (what a
caller waits for), *stream* is the time per run when 50 runs are enqueued back to back. Both include dispatch
(about 10 µs), so for 16 tiles they are dominated by it.

**Power and energy.** The card reports its board power through hwmon (`/sys/class/hwmon/*/power1_input`, name
`blackhole`). Facts that shape the method, all measured here:
- The reading has 1 W resolution and updates about every 100 ms; consecutive readings alternate by 1 W.
- One Tensix core draws far less than 1 W, so a single-core run is invisible. `bench power` therefore runs the same
  program on **16 cores at once** (each with its own input and output buffers; not DRAM-bound: 526 µs per run on 16
  cores against 522 µs on one core, 8-bit INT8 on the SFPU) for 15 s and samples the sensor every 10 ms.
- An open device with no workload already draws ~64 W (firmware, clocks; 16 W when no process has the device open),
  so the baseline is measured too: 6 s of idle with the device open, right before the load, in the same process.
- Units alternate every round (SFPU, FPU, SFPU, ...), 3 rounds, and the table gives mean ± sd over the rounds.

`energy total` = `load W` x `time per run` / cores: what a run costs including the board baseline.
`energy dynamic` = `(load W - idle W)` x `time per run` / cores: what the work adds on top of the baseline.
Both are per core-run (one single-core quantization of 1024 tiles).

## Results

**Input tile type: every row is BFLOAT16 input on BOTH units** (the SFPU column is the SFPU processing the same bf16 tiles, not fp32). No FLOAT32 rows were run: the FPU cannot take a FLOAT32 input, so there is no FPU counterpart to compare with.

**Columns.**
- `input`: element type of the tensor read from DRAM (BFLOAT16, 2 bytes per element, on both units).
- `bits`: the `bit_width` of the operation = the width of the quantized integers. `Q = 2^(bits-1) - 1` (127, 32767,
  2147483647 for 8, 16, 32). The scaler `2^k` is the largest power of two with `absmax * 2^k <= Q`, so every output
  value lies in `[-Q, Q]`. It is neither the input size nor the output container size.
- `out`: dtype of the quantized output tensor. INT32 (4 bytes per element) works for any `bits`; INT8 (1 byte per
  element, so a 1 KB output tile instead of 4 KB) only for `bits = 8`. For example `bits 32, out int32` quantizes to
  +-2147483647, `bits 8, out int32` quantizes to +-127 but stores it in 4-byte integers.
- `tiles`: tensor size in 32x32 tiles (1024 elements each).
- unit (SFPU / FPU): which math unit multiplies by `2^k` in pass 2. The absmax of pass 1 is on the SFPU in both.
- cycles: device cycles at 1350 MHz; `pass 1` / `pass 2` are per tile.

In the `FPU vs SFPU` columns **negative = the FPU is faster / uses less**.

### Device cycles (one core, device profiler, median of 10 runs)

| input | bits | out | tiles | SFPU cycles | FPU cycles | FPU vs SFPU | SFPU µs | FPU µs | pass 1 cyc/tile (both) | pass 2 cyc/tile SFPU → FPU |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 8 | int32 | 16 | 14,412 | 29,761 | +106.5% | 10.7 | 22.0 | 302 | 452 → 1404 |
| bf16 | 8 | int32 | 64 | 47,524 | 107,296 | +125.8% | 35.2 | 79.5 | 247 | 434 → 1391 |
| bf16 | 8 | int32 | 256 | 180,378 | 417,407 | +131.4% | 133.6 | 309.2 | 233 | 454 → 1388 |
| bf16 | 8 | int32 | 1024 | 711,944 | 1,657,750 | +132.8% | 527.4 | 1228.0 | 229 | 462 → 1387 |
| bf16 | 8 | int8 | 16 | 14,110 | 31,800 | +125.4% | 10.5 | 23.6 | 302 | 479 → 1544 |
| bf16 | 8 | int8 | 64 | 46,992 | 115,960 | +146.8% | 34.8 | 85.9 | 246 | 462 → 1530 |
| bf16 | 8 | int8 | 256 | 178,443 | 452,721 | +153.7% | 132.2 | 335.3 | 233 | 458 → 1527 |
| bf16 | 8 | int8 | 1024 | 704,308 | 1,799,622 | +155.5% | 521.7 | 1333.1 | 229 | 457 → 1526 |
| bf16 | 16 | int32 | 16 | 32,618 | 30,830 | -5.5% | 24.2 | 22.8 | 302 | 1676 → 1470 |
| bf16 | 16 | int32 | 64 | 122,636 | 111,545 | -9.0% | 90.8 | 82.6 | 246 | 1655 → 1457 |
| bf16 | 16 | int32 | 256 | 482,853 | 434,294 | -10.1% | 357.7 | 321.7 | 233 | 1650 → 1454 |
| bf16 | 16 | int32 | 1024 | 1,923,510 | 1,725,362 | -10.3% | 1424.8 | 1278.0 | 229 | 1648 → 1453 |
| bf16 | 32 | int32 | 16 | 31,561 | 29,764 | -5.7% | 23.4 | 22.0 | 302 | 1611 → 1404 |
| bf16 | 32 | int32 | 64 | 118,470 | 107,306 | -9.4% | 87.8 | 79.5 | 246 | 1590 → 1391 |
| bf16 | 32 | int32 | 256 | 466,106 | 417,381 | -10.5% | 345.3 | 309.2 | 233 | 1584 → 1388 |
| bf16 | 32 | int32 | 1024 | 1,856,662 | 1,657,726 | -10.7% | 1375.3 | 1227.9 | 229 | 1583 → 1387 |

### Host wall time (one core, no profiler)

| input | bits | out | tiles | SFPU latency µs | FPU latency µs | FPU vs SFPU | SFPU stream µs/run | FPU stream µs/run | FPU vs SFPU |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | 8 | int32 | 16 | 20.4 | 36.2 | +77.5% | 11.5 | 23.0 | +100.0% |
| bf16 | 8 | int32 | 64 | 45.0 | 90.1 | +100.2% | 36.0 | 80.2 | +122.8% |
| bf16 | 8 | int32 | 256 | 145.4 | 324.7 | +123.3% | 134.4 | 310.0 | +130.7% |
| bf16 | 8 | int32 | 1024 | 543.8 | 1240.9 | +128.2% | 528.2 | 1228.6 | +132.6% |
| bf16 | 8 | int8 | 16 | 20.1 | 33.4 | +66.2% | 11.2 | 24.3 | +117.0% |
| bf16 | 8 | int8 | 64 | 44.8 | 97.2 | +117.0% | 35.6 | 86.7 | +143.5% |
| bf16 | 8 | int8 | 256 | 144.6 | 349.7 | +141.8% | 132.9 | 336.2 | +153.0% |
| bf16 | 8 | int8 | 1024 | 533.7 | 1351.2 | +153.2% | 522.4 | 1333.9 | +155.3% |
| bf16 | 16 | int32 | 16 | 35.3 | 32.9 | -6.8% | 25.0 | 23.6 | -5.6% |
| bf16 | 16 | int32 | 64 | 102.7 | 94.2 | -8.3% | 91.6 | 83.4 | -9.0% |
| bf16 | 16 | int32 | 256 | 370.3 | 335.6 | -9.4% | 358.3 | 322.6 | -10.0% |
| bf16 | 16 | int32 | 1024 | 1442.2 | 1291.6 | -10.4% | 1425.1 | 1279.1 | -10.2% |
| bf16 | 32 | int32 | 16 | 34.8 | 32.1 | -7.8% | 24.2 | 22.8 | -5.8% |
| bf16 | 32 | int32 | 64 | 99.5 | 91.7 | -7.8% | 88.5 | 80.3 | -9.3% |
| bf16 | 32 | int32 | 256 | 355.4 | 323.9 | -8.9% | 345.8 | 309.9 | -10.4% |
| bf16 | 32 | int32 | 1024 | 1392.1 | 1244.7 | -10.6% | 1375.3 | 1228.8 | -10.7% |

### Power and energy (16 cores at once, 1024 tiles per core, mean ± sd over rounds)

| input | bits | out | unit | rounds | µs / run | idle W | load W | Δ W (load − idle) | energy total mJ / core-run | energy dynamic mJ / core-run |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 8 | int32 | SFPU | 3 | 550.7 ± 0.2 | 64.2 ± 0.2 | 74.5 ± 0.6 | 10.3 ± 0.4 | 2.563 ± 0.020 | 0.353 ± 0.012 |
| bf16 | 8 | int32 | FPU | 3 | 1233.2 ± 0.1 | 64.1 ± 0.1 | 69.5 ± 0.4 | 5.4 ± 0.3 | 5.357 ± 0.031 | 0.416 ± 0.021 |
| bf16 | 8 | int32 | **FPU vs SFPU** | | +123.9% | | | -47.4% | +109.1% | +17.9% |
| bf16 | 8 | int8 | SFPU | 3 | 526.0 ± 0.7 | 64.9 ± 0.7 | 72.6 ± 1.0 | 7.7 ± 0.3 | 2.387 ± 0.033 | 0.253 ± 0.011 |
| bf16 | 8 | int8 | FPU | 3 | 1337.7 ± 0.4 | 64.6 ± 0.1 | 68.4 ± 0.2 | 3.8 ± 0.2 | 5.721 ± 0.012 | 0.321 ± 0.020 |
| bf16 | 8 | int8 | **FPU vs SFPU** | | +154.3% | | | -50.1% | +139.7% | +26.9% |
| bf16 | 16 | int32 | SFPU | 3 | 1428.4 ± 0.1 | 63.9 ± 0.5 | 69.1 ± 0.5 | 5.1 ± 0.1 | 6.167 ± 0.046 | 0.458 ± 0.009 |
| bf16 | 16 | int32 | FPU | 3 | 1284.4 ± 1.0 | 64.3 ± 0.1 | 70.3 ± 0.3 | 6.0 ± 0.2 | 5.644 ± 0.018 | 0.480 ± 0.015 |
| bf16 | 16 | int32 | **FPU vs SFPU** | | -10.1% | | | +16.5% | -8.5% | +4.7% |
| bf16 | 32 | int32 | SFPU | 3 | 1379.4 ± 0.7 | 64.8 ± 0.3 | 69.9 ± 0.4 | 5.1 ± 0.2 | 6.030 ± 0.041 | 0.444 ± 0.016 |
| bf16 | 32 | int32 | FPU | 3 | 1234.3 ± 1.3 | 65.5 ± 0.5 | 70.7 ± 0.2 | 5.2 ± 0.3 | 5.455 ± 0.023 | 0.399 ± 0.024 |
| bf16 | 32 | int32 | **FPU vs SFPU** | | -10.5% | | | +0.6% | -9.5% | -10.0% |

### Reading the results

- **Cycles and wall time agree.** For 16 and 32 bits the FPU saves 5.5 % (16 tiles, where the fixed cost matters) to
  10.7 % (1024 tiles) of the kernel time. For 8 bits it takes 2.1x (16 tiles) to 2.3x (INT32) / 2.6x (INT8) at 1024 tiles.
- **Where the time goes** (cycles per tile at 1024 tiles, 16 bits): pass 1 = 229 (the same for both units; mostly
  the SFPU absmax, 211 of 231 in the probe), pass 2 = 1648 on the SFPU and 1453 on the FPU. The FPU path removes `copy_tile` + scaler copy +
  SFPU multiply (about 200 cycles) and pays for nothing new; what remains in pass 2 is round, clamp and convert on
  the SFPU plus the wait for the writer (one NoC write + barrier per tile; the writer RISC takes 1.7 – 1.9k cycles per tile in total,
  including its wait for compute).
- **8 bits is the other way round.** The SFPU does multiply, round, clamp and convert in one fused op (pass 2 =
  460 cycles per tile, writer-bound). With the FPU multiply the fused op cannot be used (it multiplies by the scale
  itself), so the kernel runs round, clamp, convert, and for INT8 a mask: pass 2 = 1387 – 1526 cycles per tile.
- **Power.** Under load the 16 cores add 5 – 10 W on top of the ~64 W baseline. The FPU variant does not draw
  visibly less than the SFPU variant while it works: the difference in the dynamic part is within what 3 rounds
  of a 1 W sensor can resolve (the `energy dynamic` change is +4.7 % at 16 bits and -10.0 % at 32 bits). What
  does show is the baseline: the board burns ~64 W the whole time, so a run that takes 10 % less time uses
  8.5 – 9.5 % less energy. For 8 bits the SFPU variant draws more power (10 W against 5 W above idle) but
  finishes 2.3x sooner, so it uses less than half the energy.

## What the FPU can and cannot do

The probe (`probe/`, raw output in `results/primitives.txt`) measures the primitives with 16 bf16 tiles resident in
L1, so only compute is timed:

| Step (per tile) | Cycles | Notes |
|---|---|---|
| SFPU absmax, with `copy_tile` (pass 1 today) | 231 | `copy_tile` alone is 45, the SFPU absmax alone is 211 |
| FPU max pool, column (`reduce_tile<MAX, REDUCE_COL>`) | 29.5 | accumulates across tiles in one DST tile |
| FPU max pool, scalar (`REDUCE_SCALAR`) | 53 | same |
| FPU multiply (`mul_tiles` with the scaler tile) | 88 | blocks of 3 tiles, includes the DST handshake |
| `copy_tile` + scaler copy + SFPU `mul_binary_tile` | 237 | what pass 2 did before |

So the FPU is 2.7x cheaper for the multiply and about 8x cheaper for a max. The max pool was tried first:

- **It is exact and it accumulates.** Over 4 tiles, mode 0 returns the per-column maximum of all tiles, equal to the
  host answer, also for all-negative data (so the DST needs no initial value).
- **It cannot give `max(abs(x))`.** `GMPOOL` is a max over SrcA only: there is no min and no abs on the FPU
  (tt-metal's own `static_assert` for `PoolType::MIN` says to use the SFPU). The scaler tile does not help: a
  scaler of -1 returns exactly the same result as +1 (modes 0 and 1 in `primitives.txt`), so `max(-x)` is not
  available that way.
- **`-x` is easy to make, but it ends up in the DST** (`mul_tiles` by -1 is exact), and `GMPOOL` reads SrcA. Moving it
  back (a DST to SrcA move, `MOVD2A`) is not exposed by the compute API for pooling. The alternative, packing `-x`
  to a CB and unpacking it again, gives the DST section away on every pack (`tile_regs_commit`), so the running
  maximum held in the DST is lost; my estimate for the round trip (about 150 – 200 cycles per tile in total) leaves little against 231.
  Reinterpreting the bf16 bits as UInt16 for an integer pool was tried too: the unpacker converts them to an
  unrelated float encoding (see `probe/probe.cpp`, last argument).

**What would change it** (not built; the numbers are estimates, not measurements):
1. A custom math step that moves a DST face to SrcA and then pools it (the FPU absmax would need two reduces and a
   multiply per tile and re-initialise the FPU every few tiles because the DST holds only four fp32 tiles; my
   estimate is about 150 – 180 cycles per tile against 231, so a gain of 25 % at best, for hand-written Tensix
   instruction sequences).
2. A faster SFPU absmax (the loop costs about 6.6 cycles per SFPU row; `SFPSWAP` could do max in fewer
   instructions). This helps both units.
3. A fused SFPU op for 16 / 32 bits (round, clamp, convert in one sweep): pass 2 is 1450 – 1650 cycles per tile
   now, against 460 for the fused 8-bit op.
4. A writer that batches its NoC writes (one write and one barrier per tile now).
