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

## Device cycles (one core, device profiler, median of 10 runs)

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

## Host wall time (one core, no profiler)

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

## Power and energy (16 cores at once, 1024 tiles per core, mean ± sd over rounds)

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
