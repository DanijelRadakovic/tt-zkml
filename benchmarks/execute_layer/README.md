# execute_layer: cycles and time of one quantized layer

`zkml::prim::execute_layer(x, w, activation)` on one Tensix core: quantize `x` and `w` to 8-bit sign-magnitude bytes (the quantizer's
two passes with the scaler in LReg7), multiply them on the FPU at HiFi4 into an exact int32 accumulator, apply the activation
on it, and requantize to 8 bits. This benchmark times the four stages with the device profiler (zones `QUANT_X`, `QUANT_W`,
`MATMUL`, `REQUANT` of the compute kernel) and the whole call with the host clock.

## Run

```bash
cmake --preset default && cd .build/default && ninja zkml_execute_layer_bench && cd ../..
TT_METAL_HOME=... benchmarks/execute_layer/run_all.sh                    # about 2 minutes, writes results/
TT_METAL_DEVICE_PROFILER=1 .build/default/zkml_execute_layer_bench profile 4 16 4 relu 8      # then analyze.py <log> 4 16 4
.build/default/zkml_execute_layer_bench wall 4 16 4 relu 50
```

The sizes are in tiles of 32: `Mt Kt Nt` means `x` is `32 Mt x 32 Kt` and `w` is `32 Kt x 32 Nt`. Run it on the machine with the device
(inside the IRD container). The correctness of the op is tested by `test_execute_layer` (every output byte and the three scalers
against a host model of the whole pipeline).

## Results (one core at 1350 MHz, ReLU; `results/cycles.txt`, `results/wall.txt`)

Cycles of the slowest compute RISC of each zone, median of 7 runs (the first is dropped). The zones include the time the compute
kernel waits for the reader (`QUANT_X` for the tiles of x, `QUANT_W` for those of w, which are read while x is quantized).

| x (M x K) | w (K x N) | quantize x | quantize w | matmul + relu | requantize | sum (us) | wall per call (us) | cycles per tile matmul |
|---|---|---|---|---|---|---|---|---|
| 32 x 32 | 32 x 32 | 1747 | 1125 | 524 | 948 | 3.2 | 12.4 | 523.5 |
| 64 x 128 | 128 x 64 | 7538 | 6510 | 2864 | 2643 | 14.5 | 18.6 | 179.0 |
| 128 x 256 | 256 x 128 | 26496 | 24937 | 14950 | 8089 | 55.2 | 56.3 | 116.8 |
| 128 x 512 | 512 x 128 | 51130 | 49454 | 23525 | 8081 | 97.9 | 99.6 | 91.9 |
| 64 x 1024 | 1024 x 64 | 51092 | 49426 | 10362 | 2802 | 84.2 | 86.0 | 81.0 |
| 32 x 256 | 256 x 256 | 7536 | 49432 | 7602 | 4570 | 51.2 | 52.2 | 118.8 |
| 256 x 256 | 256 x 32 | 50998 | 6544 | 7600 | 4563 | 51.6 | 53.3 | 118.8 |

- **The quantization dominates**: quantize x and w together are about 70 to 75% of a call at the larger sizes (about 800 cycles per fp32 tile: pass 1,
  pass 2 with the sign-magnitude sweep, and the DRAM read). Weights that are quantized once and reused would remove half of it;
  this version quantizes both on every call.
- **The FPU matmul runs at 69 to 72 cycles per 32x32x32 tile** (HiFi4, `matmul_tiles`) when nothing else is done, the rate measured by
  `benchmarks/fpu_int8_vs_bf16`. With the activation and the absmax it is 81 to 119 cycles per tile.

## Does the activation overlap the matmul? Not in this version

The matmul works on blocks of 4 output tiles. After the K loop of a block, an SFPU sweep applies the activation on the int32
accumulator (ReLU: zero the negative values), converts it to fp32 and folds it into the running absmax in LReg7; then the block is packed
while the FPU starts the next one. What overlaps is the packer, the unpacker and the next block's FPU work. The SFPU sweep itself does not
overlap the FPU: removing it (a temporary edit of the kernel, outputs then wrong, same shapes) changed the `MATMUL` zone like this:

| Mt Kt Nt | blocks | `MATMUL` with the sweep | without | difference per block | per accumulator tile |
|---|---|---|---|---|---|
| 4 16 4 | 4 | 23525 | 17606 | 1480 | 370 |
| 2 32 2 | 1 | 10356 | 8879 | 1477 | 369 |
| 1 8 8 | 2 | 7596 | 4642 | 1477 | 369 |

So the sweep costs 370 cycles per accumulator tile every time and is fully visible. It is 35% of the matmul at K = 512 and 17% at K = 1024.
The reason is how the math thread works: the SFPU code is issued instruction by instruction by the same RISC-V core that issues the FPU
work (about 11 instructions per 4 rows x 8 lanes, 32 of those per tile, one per cycle), so while it issues the sweep it does not issue the
next block's matmul. The hardware allows the FPU and the SFPU to work at the same time on different halves of DST (ISA documentation,
`Dst.md`), but not with this structure. Ways to get the overlap, not done:

1. Issue the sweep of block i in small pieces between the matmul calls of block i + 1 (the FPU instructions run 66 cycles each), which
   needs two DST sections held at once (full-sync DST, 8 tiles) and loses the packer overlap.
2. Record the sweep in the replay buffer so one instruction issues many (as tt-metal's quant kernels do); this shortens the issue time
   but the SFPU executes about one instruction per cycle too, so it helps only together with 1.
3. Apply the activation after the requantize, on the int8 bytes (less work per tile; ReLU is then not exact on the accumulator's absmax).

## Limits

One core; FLOAT32 inputs (an `Int8` circular buffer cannot share a program with a bf16 one); `K <= 1024` (the accumulator is converted to
fp32 exactly only below 2^24); `x`, `w`, the accumulator and the outputs must fit in L1 (the op throws otherwise); the activation is ReLU or
none. Power was not measured.
