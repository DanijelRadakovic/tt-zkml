# SFPU quantizer with the scaler in LReg7 (own fused kernels) against tt-metal's quant kernel and composed ops

Device-profiler cycles on one core (`bench profile <in> <unit> <bits> <out> <tiles> 11`, median of 10 runs, 1350 MHz). **before** = the op with tt-metal's kernels: `quant_tile` / `quant_int8_tile` at 8 bits, and at 16 and 32 bits the composed ops (multiply, round, clamp, typecast, one pass each) with the scaler tile in DST; **after** = `quant_sfpu.hpp`: one fused sweep per tile that multiplies by the scaler in LReg7, rounds half to even and converts to int32 (no `c_1` with the SFPU multiply, no scaler copy per block, blocks of 4 tiles instead of 3, no clamp). Before and after were measured on the same card in one session (the 16/32-bit rows with the old code rebuilt from a `git stash`). Pass 1 (absmax, about 229 to 249 cycles per tile for bf16 and 328 to 336 for fp32 input) is unchanged. The op tests pass, including `RoundingBoundaries` (exact ties and the limits of the rounding methods at 16 and 32 bits).

## 8 bits (`quant8`: SFPLOAD, SFPMAD, SFPNOP, SFPIADD [, SFPXOR], SFPSTORE)

| input | out | tiles | pass 2 cycles/tile before | after | change | whole kernel cycles/tile before | after | change |
|---|---|---|---|---|---|---|---|---|
| bf16 | int32 | 1024 | 455.0 | 452.7 | -0% | 688.4 | 688.5 | +0% |
| bf16 | int32 | 256 | 448.3 | 437.4 | -2% | 697.8 | 696.0 | -0% |
| bf16 | int32 | 64 | 434.5 | 376.5 | -13% | 735.0 | 726.6 | -1% |
| bf16 | int8 | 1024 | 457.0 | 396.9 | -13% | 687.8 | 631.7 | -8% |
| bf16 | int8 | 256 | 458.1 | 385.0 | -16% | 696.9 | 640.1 | -8% |
| bf16 | int8 | 64 | 462.3 | 337.7 | -27% | 733.5 | 674.0 | -8% |
| f32 | int32 | 1024 | 511.5 | 453.8 | -11% | 842.0 | 788.5 | -6% |
| f32 | int32 | 256 | 512.2 | 440.9 | -14% | 852.4 | 798.1 | -6% |
| f32 | int32 | 64 | 514.6 | 389.7 | -24% | 892.9 | 836.5 | -6% |
| f32 | int8 | 1024 | 537.9 | 398.2 | -26% | 868.1 | 732.1 | -16% |
| f32 | int8 | 256 | 538.5 | 388.7 | -28% | 877.6 | 742.4 | -15% |
| f32 | int8 | 64 | 541.1 | 360.2 | -33% | 915.1 | 783.7 | -14% |

The gain is largest where pass 2 is bound by the SFPU (Int8 output, 1 KB tiles). With Int32 output and bf16 input the loop is bound by the 4 KB output tile (pack and writer), so the faster SFPU kernel changes little at 1024 tiles.

## 16 and 32 bits (Int32 output)

16 bits uses the same sweep as 8 bits (`|x * 2^k| < 2^22`). 32 bits uses `quant32` (SFPI code: rounding by the magic-number add below 2^23, by the significand shift from 2^23 on, sign last).

| input | unit | bits | tiles | pass 2 cycles/tile before | after | change | whole kernel cycles/tile before | after | change |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | fpu | 16 | 1024 | 1453 | 453 | -69% | 1685 | 689 | -59% |
| bf16 | fpu | 16 | 256 | 1454 | 440 | -70% | 1696 | 698 | -59% |
| bf16 | fpu | 16 | 64 | 1457 | 387 | -73% | 1742 | 734 | -58% |
| bf16 | sfpu | 16 | 1024 | 1648 | 453 | -73% | 1878 | 688 | -63% |
| bf16 | sfpu | 16 | 256 | 1649 | 437 | -73% | 1886 | 696 | -63% |
| bf16 | sfpu | 16 | 64 | 1655 | 376 | -77% | 1916 | 726 | -62% |
| f32 | sfpu | 16 | 1024 | 1719 | 454 | -74% | 2048 | 788 | -62% |
| f32 | sfpu | 16 | 256 | 1720 | 441 | -74% | 2056 | 798 | -61% |
| f32 | sfpu | 16 | 64 | 1724 | 390 | -77% | 2088 | 836 | -60% |
| bf16 | fpu | 32 | 1024 | 1387 | 870 | -37% | 1619 | 1102 | -32% |
| bf16 | fpu | 32 | 256 | 1388 | 871 | -37% | 1630 | 1113 | -32% |
| bf16 | fpu | 32 | 64 | 1391 | 874 | -37% | 1676 | 1159 | -31% |
| bf16 | sfpu | 32 | 1024 | 1583 | 834 | -47% | 1813 | 1066 | -41% |
| bf16 | sfpu | 32 | 256 | 1584 | 835 | -47% | 1820 | 1077 | -41% |
| bf16 | sfpu | 32 | 64 | 1589 | 836 | -47% | 1850 | 1120 | -39% |
| f32 | sfpu | 32 | 1024 | 1654 | 940 | -43% | 1983 | 1271 | -36% |
| f32 | sfpu | 32 | 256 | 1655 | 940 | -43% | 1991 | 1282 | -36% |
| f32 | sfpu | 32 | 64 | 1659 | 942 | -43% | 2022 | 1329 | -34% |

`unit` is the multiply: `sfpu` = copy_tile and a multiply by LReg7 inside the sweep, `fpu` = `mul_tiles` (scaler through SrcB), after which the sweep only rounds and converts. With the fused sweeps the FPU multiply no longer pays: at 16 bits both take the same time (about 453 cycles per tile, bound by the 4 KB output tile), at 32 bits the SFPU multiply is faster (834 against 870 for bf16 input at 1024 tiles).


## Writer: batched writes (one flush per batch instead of one write barrier per tile)

The writer of the op waited for one tile, wrote it and waited for the write barrier (the data has arrived in DRAM) before the next tile. It now waits for a compute block (4 tiles, never across the end of the `c_16` ring), issues the 4 writes back to back, waits once until they have left L1 (`async_writes_flushed`) and pops the block; the only full `async_write_barrier` is the last statement of the kernel. Measured with the fused SFPU kernels before and after the writer change (`bench profile`, one core, median of 10 runs). Pass 1 and the compute kernels are the same.

| bits | input | out | tiles | pass 2 cycles/tile before | after | change | whole kernel cycles/tile before | after | change |
|---|---|---|---|---|---|---|---|---|---|
| 8 | bf16 | int32 | 1024 | 453 | 225 | -50% | 688 | 455 | -34% |
| 8 | bf16 | int32 | 256 | 437 | 225 | -49% | 696 | 463 | -34% |
| 8 | bf16 | int32 | 64 | 376 | 226 | -40% | 727 | 491 | -32% |
| 8 | bf16 | int8 | 1024 | 397 | 258 | -35% | 632 | 488 | -23% |
| 8 | bf16 | int8 | 256 | 385 | 258 | -33% | 640 | 494 | -23% |
| 8 | bf16 | int8 | 64 | 338 | 259 | -23% | 674 | 521 | -23% |
| 8 | f32 | int32 | 1024 | 454 | 336 | -26% | 789 | 666 | -16% |
| 8 | f32 | int32 | 256 | 441 | 336 | -24% | 798 | 674 | -16% |
| 8 | f32 | int32 | 64 | 390 | 337 | -14% | 836 | 706 | -16% |
| 8 | f32 | int8 | 1024 | 398 | 359 | -10% | 732 | 689 | -6% |
| 8 | f32 | int8 | 256 | 389 | 359 | -8% | 742 | 696 | -6% |
| 8 | f32 | int8 | 64 | 360 | 360 | +0% | 784 | 726 | -7% |
| 16 | bf16 | int32 | 1024 | 453 | 225 | -50% | 688 | 455 | -34% |
| 16 | bf16 | int32 | 256 | 437 | 225 | -49% | 696 | 463 | -34% |
| 16 | bf16 | int32 | 64 | 376 | 226 | -40% | 726 | 492 | -32% |
| 16 | f32 | int32 | 1024 | 454 | 336 | -26% | 788 | 666 | -16% |
| 16 | f32 | int32 | 256 | 441 | 336 | -24% | 798 | 674 | -16% |
| 16 | f32 | int32 | 64 | 390 | 337 | -14% | 836 | 705 | -16% |
| 32 | bf16 | int32 | 1024 | 834 | 834 | -0% | 1066 | 1065 | -0% |
| 32 | bf16 | int32 | 256 | 835 | 835 | -0% | 1077 | 1072 | -0% |
| 32 | bf16 | int32 | 64 | 836 | 836 | -0% | 1120 | 1101 | -2% |
| 32 | f32 | int32 | 1024 | 940 | 945 | +1% | 1271 | 1275 | +0% |
| 32 | f32 | int32 | 256 | 940 | 945 | +1% | 1282 | 1283 | +0% |
| 32 | f32 | int32 | 64 | 942 | 946 | +0% | 1329 | 1314 | -1% |

The per-tile barrier, not the SFPU, was what bounded pass 2 at 8 and 16 bits: with bf16 input and Int32 output pass 2 went from 453 to 225 cycles per tile (the earlier explanation in this file, that the loop was bound by the 4 KB output tile, was only half right: it was the round trip of the barrier for every tile). 32 bits is bound by the SFPU sweep and does not change. The whole kernel at 1024 tiles takes 455 cycles per tile for 8 and 16 bits with bf16 input and Int32 output, down from 688.
