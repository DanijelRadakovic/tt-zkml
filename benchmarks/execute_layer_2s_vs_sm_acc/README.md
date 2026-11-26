# execute_layer: two's-complement against sign-magnitude accumulator

The FPU's int32 accumulator is sign-magnitude. The first version of `execute_layer` converted it to two's complement on the
SFPU before packing it (`acc` output) and then went through fp32 for the requantization. This benchmark measures what the raw
sign-magnitude `acc` and the integer requantizer (the production code now) save, using the same program as the op (same CBs,
reader, writer) with the accumulator handling as a compile-time variant of the compute kernel (`kernels/compute.cpp`).

| variant | after the matmul (per block) | requantization |
|---|---|---|
| `twos` (old op) | SM -> two's complement in place, float absmax of the activated accumulator | int32 -> fp32 (ReLU), then the fp32 quantize sweep |
| `sm_float` | accumulator untouched (raw SM), float absmax (SM -> fp32 convert per element) | SM -> fp32 (ReLU), then the fp32 quantize sweep |
| `sm_int_sfpi` | accumulator untouched, integer absmax in SFPI code with branches | integer scaler, integer round-half-even in SFPI code with branches |
| `sm_int_sfpi_absmax` | as `sm_int_sfpi` | as `sm_int` |
| `sm_int` (production) | accumulator untouched, integer absmax as 2-3 hand-written instructions (SFPSWAP max) | integer scaler, hand-written branch-free requantizer (11 / 13 instructions per 4 rows x 8 lanes) |

`sm_float` is exact only for `K <= 1024` (fp32 holds the accumulator below 2^24); the integer variants have no such limit
(K <= 66560). `acc` of `twos` is two's complement, the others return the raw sign-magnitude bits.

## The requantizer

With `K24 = 24 + k` in LReg7 and `mag` the magnitude: `z = mag << K24` puts q in bits 24..30 and the fraction in bits 0..23 (the
sign bit shifts out, no mask needed), then `q = (z + 0x7FFFFF + ((z >> 24) & 1)) >> 24` is round half to even. Without ReLU the
sign returns as bit 7 if `q != 0` (SFPSETCC / SFPOR / SFPENCC, no "negative zero"), with ReLU a negative accumulator is zeroed
first. The absmax is one SFPSWAP (mod1 = 9, VD = max) per 4 rows x 8 lanes against the sign-magnitude bits: negatives never win
against the running max, so ReLU needs no mask.

## Run

Inside the IRD container, from the repo root (`cmake --preset default && cd .build/default && ninja zkml_execute_layer_2s_vs_sm_acc_bench`):

    TT_METAL_HOME=/proj_sw/user_dev/ctr-dradakovic/tt-metal benchmarks/execute_layer_2s_vs_sm_acc/run_all.sh

If the profiler aborts with "Source location hashes are colliding", delete `$TT_METAL_HOME/generated/profiler/.logs/zone_src_locations.log` (stale after the kernel changed).

Results: `results/check.txt` (all variants give identical `out` bytes, identical decoded `acc` and the same `scaler_out` for every
shape), `results/cycles.txt` (zone cycles: median of 8 runs, slowest compute RISC), `results/wall.txt` (host time of enqueue +
finish per run, program cached). Random inputs in [-2, 2].

## Results (ReLU; cycles of MATMUL + REQUANT, the zones that differ; QUANT_X / QUANT_W are identical)

| Mt,Kt,Nt | twos | sm_float | sm_int_sfpi | sm_int_sfpi_absmax | **sm_int** |
|---|---|---|---|---|---|
| 1,1,1 | 2176 | 1453 | 1655 | 1562 | **1131** |
| 2,4,2 | 7771 | 5780 | 6655 | 5795 | **4179** |
| 4,8,4 | 31743 | 25456 | 28986 | 24467 | **19174** |
| 4,16,4 | 40340 | 34031 | 37555 | 33042 | **27745** |
| 2,32,2 | 15411 | 13271 | 14144 | 13454 | **11677** |
| 1,8,8 | 16617 | 13048 | 14778 | 12853 | **9894** |
| 8,8,1 | 16615 | 13054 | 14750 | 12854 | **9887** |

Wall time per run, 4,8,4 / 4,16,4 (ReLU): twos 62.6 / 105.7 us, sm_float 58.9 / 101.2, sm_int_sfpi 61.1 / 104.1, **sm_int 53.8 / 96.6**.

Per zone at 4,8,4 (relu): MATMUL (matmul + the per-block sweep) 19330 / 14434 / 15462 / 15463 / **10773**, REQUANT 12413 / 11022 /
13524 / 9004 / **8401** (twos / sm_float / sm_int_sfpi / sm_int_sfpi_absmax / sm_int). The matmul itself is about 128 tile
matmuls x 66.6 = 8500 cycles of the 10773.

## Findings

- Dropping the two's complement conversion alone (`sm_float`) saves about 25% of the matmul-zone sweep.
- Written as SFPI code the integer requantizer and absmax were slower than the float ones (`sm_int_sfpi`: 28986 against 25456).
- Written as hand-written instructions they are the fastest: `sm_int` takes 19174 cycles against 31743 for the old op (-40%), 8-10% of
  a whole call (about 9 us of 62 at 4,8,4, 9 us of 106 at 4,16,4), and has no K limit. The absmax sweep (2-3 instructions) is now
  close to free next to the matmul, the requantization is 8401 against 11022 cycles for the float one.
- Remaining cost: the matmul itself (about 8500 of the 10773 cycles of the MATMUL zone) and the quantization of x and W (QUANT_X, QUANT_W).

## Quantization of x and W (QUANT_X, QUANT_W)

All variants share this part. `results/cycles_before_quant_opt.txt` and `wall_before_quant_opt.txt` are the numbers before it was optimized.
`QX_P1` / `QX_P2` (last two columns of `cycles.txt`) are pass 1 (absmax, scaler) and pass 2 (quantize, pack) of x.

| Mt,Kt,Nt (ReLU, `sm_int`) | QUANT_X before | after | QUANT_W before | after | call (wall) before | after |
|---|---|---|---|---|---|---|
| 2,4,2 | 7575 | 6298 | 6516 | 5138 | 15.3 us | 12.8 us |
| 4,8,4 | 26577 | 21211 | 24954 | 19326 | 53.8 us | 45.6 us |
| 4,16,4 | 51141 | 40246 | 49452 | 38196 | 96.6 us | 79.9 us |
| 2,32,2 | 51139 | 40064 | 49434 | 38206 | 84.7 us | 68.1 us |

What changed (all outputs bit-identical to before, `check.txt`, and the op tests, including the new tie test, pass):
- **Pass 2 writes the Int8 CB directly.** The quantize kernel ends in one SFPCAST to sign-magnitude and the Int8 packer writes sign + 7
  magnitude bits (`sm_int8pack` of `benchmarks/sign_mag_fpu_compatibillity`; fp32 input only), instead of abs / shift / and / or
  (4 instructions) for the UInt8 packer; the FPU reads the same CB, so the aliased UInt8/Int8 CB pair is gone. The loop is software
  pipelined over LREG0/LREG1, which removes the SFPNOP: 5 instructions per 4 rows x 8 lanes instead of 9.
- **Pass 1 absmax in hand-written instructions**: SFPSWAP (mod1 9 / 1) against a running max and min kept in LReg7 / LReg6, two rows per
  3 swaps, no SFPABS: 2.5 instructions per 4 rows x 8 lanes instead of about 6 of SFPI code. (`pot_quantize_per_tensor` shares this.)

Where the rest goes (x, 32 tiles, `ZKML_QMODE` experiments in `kernels/compute.cpp`): pass 1 is 10.2k cycles of which copy_tile alone is 5.9k
(184 per tile) and the SFPU alone 4.8k; pass 2 is 11.0k of which copy + pack is 5.1k. copy_tile of an fp32 tile and the SFPU instructions
run on the same math thread and add up, so the SFPU work is what is left to cut. Tried and not possible on Blackhole: SFPCAST from fp32 to
sign-magnitude with rounding (`FP32_TO_SM32_RNE` exists only on Quasar), which would save the magic-number add.

## SFPLOADMACRO for pass 2 of the quantization (`run_quant_lm.sh`, `results/quant_lm.txt`)

`ZKML_QLM=0` is the pipelined hand-written kernel (`quant_sm8`, 5 instructions per 4 rows x 8 lanes), `ZKML_QLM=1` the SFPLOADMACRO
kernel (`quant_sm8_lm`). All accumulator variants still agree (`check`); the op tests compare the LM kernel with the host model.

`SFPLOADMACRO` loads a row and schedules up to four more instructions, one per SFPU sub-unit (Simple, MAD, Round, Store), for later
cycles. Here one macro per row schedules the SFPMAD (x * 2^k + magic, MAD unit), the SFPIADD (minus the magic bits, Simple unit) and the
SFPSTORE (Store unit); the SFPCAST to sign-magnitude is the one instruction issued normally, in the slot between two macros. That is 2 issue
slots per row instead of 5, and the loop runs at about 2 cycles per row (the Simple unit does 2 instructions per row).

| Mt,Kt,Nt (ReLU, `sm_int`) | QX_P2 (cycles) TTI -> LM | QUANT_X | QUANT_W | call (wall) |
|---|---|---|---|---|
| 2,4,2 | 2855 -> 2178 | 6262 -> 5178 | 5160 -> 4082 | 12.8 -> 11.4 us |
| 4,8,4 | 10953 -> 7085 (-35%) | 21285 -> 17076 (-20%) | 19448 -> 15225 (-22%) | 46.1 -> 39.5 us (-14%) |
| 4,16,4 | 21697 -> 13600 | 40407 -> 31917 | 38446 -> 30034 | 80.3 -> 68.3 us |
| 2,32,2 | 21705 -> 13597 | 40322 -> 31820 | 38474 -> 30032 | 68.3 -> 56.3 us |

What the LM kernel needs (tt-isa-documentation, `SFPLOADMACRO.md`): the macro config (`SFPCONFIG`: the sequence bytes, two instruction
templates captured by issuing an instruction with VD = 12 + i, the delay kind), 4 row registers (an SFPSWAP-like write-back or a late store
must not meet the next load), no regular SFPU instruction on a sub-unit on a cycle where a scheduled one arrives (the regular one is
silently dropped), SFPNOPs before the first macro (SFPLOADMACRO does not stall for an FPU write to Dst, as with the bf16 copy_tile) and
after the last (scheduled instructions only advance when instructions are issued).

### What did not work: SFPLOADMACRO for pass 1 (the absmax)

The macro form (load + scheduled SFPABS, the SFPSWAP issued normally) gave wrong results: SFPSWAP keeps the Simple sub-unit busy for 2
cycles, and the scheduled SFPABS that lands on the cycle after a swap is lost (all negative values were missed). The swaps are also the
cost: a software-pipelined pair kernel (3 swaps per 2 rows, the loads of the next pair in the gaps, 6 issue slots per pair) measured
148 cycles per tile for the SFPU part, the same as the earlier version, i.e. about 3.1 cycles per SFPSWAP, so pass 1 is bound by the
swaps and not by the issue slots. The shipped absmax is the pipelined pair kernel (whole tile in one call).

### bf16 input now uses the same kernel

For bf16 input the q buffers are UInt8 buffers (the JIT does not allow an Int8 one next to a bf16 one). The packer is now configured for
Int8 by hand (`pack_reconfig_int8_view`, like the unpacker's Int8 view for the matmul), so the bf16 path uses the same LM kernel and
Int8 packer as fp32 instead of the 8-instruction UInt8 kernel. `benchmarks/execute_layer` with `ZKML_DTYPE=bf16`, 4,8,4 ReLU:
QUANT_X 17032 -> 9982 cycles, call 40.4 -> 29.8 us (fp32 39.8 us).

## SFPLOADMACRO for the requantize sweep: analysis, not implemented

The requantize sweep is the other SFPU-bound stage (REQUANT zone, 16 tiles at 4,8,4). I measured how much of it is the sweep
(`results/requant_sfpu_cost.txt`, `ZKML_QMODE=4` removes the sweep and keeps copy_tile + pack): 5037 of the 8403 cycles with ReLU
(315 per tile, 9.8 cycles per 4 rows x 8 lanes for 11 instructions) and 5905 of 9441 without (11.5 cycles for 13 instructions). That
is about one cycle per instruction: issue-bound, as quantize pass 2 was. But SFPLOADMACRO cannot reach the same kind of gain here:

- **The work is almost all on the Simple sub-unit.** Per row the integer algorithm needs 8 (ReLU) or 10 (no activation) Simple
  instructions (shifts, AND, IADDs, SETCC/ENCC, ...). The Simple unit does one instruction per cycle, so that is the floor, whatever
  the issue slots: 8 against 9.8 cycles with ReLU, 10 against 11.5 without. A macro can only take the load, the store and at most
  ONE Simple instruction off the issue stream (one instruction per sub-unit column). The MAD column (float multiply-add) and the Round
  column (SFPSTOCHRND, SFPSHFT2) have nothing to do in an exact integer shift-and-round.
- **The chain is longer than a macro can schedule.** The delay of a scheduled instruction is 3 bits, so it executes at most 8 cycles
  after the macro. A row's chain is 9 to 11 Simple instructions long, so its scheduled store (and any scheduled last instruction)
  would have to run before the data is final.
- **The CC instructions do not mix with overlapping rows.** The ReLU is done with SFPSETCC / conditional load / SFPENCC (lane flags
  are global state), and the "no negative zero" gate uses them too; with several rows in flight at once the flags of one row would
  disable lanes of another row's scheduled instructions. An arithmetic version (mask = arithmetic shift, XOR, add) needs the same
  number of instructions.
- **SFPSTOCHRND (Round column) as the final shift-and-round** does shift right with rounding, but rounds ties away from zero, and the
  correction for ties to even needs `mag - 1 + bit` as an input (a borrow at mag = 0 turns the SM value into a huge magnitude);
  its round-to-zero mode is wrong when 23 bits are discarded and all of them are ones (documented hardware bug), which happens for
  an accumulator with 30 magnitude bits. Neither is exact.

Upper bound of what a perfect schedule could win: about 1.5 cycles per row, 50 per tile, 0.8k cycles of the 8.4k of the zone, 1.5% of a
call. That is not worth the risk of a kernel that is hard to get right, so the requantizer stays the hand-written 11 / 13 instruction
kernel (`requant_tile`). A real gain would need fewer Simple instructions, i.e. a different algorithm (for example finding a
one-instruction exact "round half to even shift"), not a different way of issuing them.
