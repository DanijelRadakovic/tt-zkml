# Results: int8 against bf16 matmul on the FPU

Generated from `fpu_int8_vs_bf16.csv` (`zkml_fpu_int8_vs_bf16_bench`, one core of a Blackhole p100a at 1350 MHz, one 2x2-tile output block, A and B resident in L1, wall clock with the launch overhead removed by differencing two repetition counts).

## Time per 32x32x32 tile matmul (ns), random data

int8 and bf16 give the same time for the same fidelity (the same instructions run), so one table; the two paths differ by at most 2% (at K=32, where the launch of the block dominates).

| mode | fidelity | K=32 | K=128 | K=512 | K=1024 | K=2048 | K=4096 | GMAC/s at K=4096 |
|---|---|---|---|---|---|---|---|---|
| block | lofi | 35.2 | 13.6 | 13.1 | 13.0 | 13.0 | 13.0 | 2525 |
| block | hifi2 | 34.9 | 26.2 | 25.7 | 25.6 | 25.6 | 25.6 | 1281 |
| block | hifi3 | 39.8 | 38.0 | 37.6 | 37.5 | 37.4 | 37.4 | 876 |
| block | hifi4 | 51.7 | 49.9 | 49.4 | 49.3 | 49.3 | 49.3 | 665 |
| tiles | lofi | 34.4 | 21.7 | 21.3 | 21.2 | 21.1 | 21.1 | 1552 |
| tiles | hifi2 | 35.1 | 26.5 | 26.1 | 26.0 | 25.9 | 25.9 | 1263 |
| tiles | hifi3 | 40.2 | 38.4 | 37.9 | 37.8 | 37.8 | 37.8 | 867 |
| tiles | hifi4 | 52.0 | 50.2 | 49.8 | 49.7 | 49.6 | 49.6 | 661 |

`block` is `matmul_block` (what ttnn uses), `tiles` is `matmul_tiles`.

## Exactness: outputs that differ from the exact integer matmul (of 4096), `block` mode

Each cell is K=32 / K=4096. `0 / 0` means every output was exact at both ends (the other K values in the CSV agree).

| path | fidelity | random: A, B in [-127, 127] | medium: A, B in [-45, 45] | small: A, B in [-31, 31] | narrow: A in [-31, 31], B in [-15, 15] | constant: A = B = 127 |
|---|---|---|---|---|---|---|
| int8 | lofi | 4096 / 4096 | 4095 / 4096 | 4093 / 4096 | 4093 / 4096 | 4096 / 4096 |
| int8 | hifi2 | 4096 / 4096 | 4095 / 4096 | 4095 / 4096 | 4096 / 4096 | 4096 / 4096 |
| int8 | hifi3 | 4096 / 4096 | 4095 / 4096 | 4095 / 4096 | 4096 / 4096 | 4096 / 4096 |
| int8 | hifi4 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| bf16 | lofi | 4092 / 4095 | 4057 / 4094 | 0 / 0 | 0 / 0 | 4096 / 4096 |
| bf16 | hifi2 | 3690 / 4067 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| bf16 | hifi3 | 3690 / 4067 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| bf16 | hifi4 | 3690 / 4067 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Largest error of the bf16 path (hifi4, random data), in units of the exact integer result: K=32: 15, K=128: 33, K=512: 62, K=1024: 78, K=2048: 106, K=4096: 153.

Encoding of the int8 path's Int32 output in L1 (random data, hifi4): sign-magnitude (12 runs).

## Which path for which K (full int8 range, A and B in [-127, 127], `block` mode)

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
