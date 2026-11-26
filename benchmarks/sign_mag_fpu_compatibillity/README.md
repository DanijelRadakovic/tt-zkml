# Sign-magnitude int8 output of the SFPU quantizer: what works with the FPU, and at what cost

`pot_quantize_per_tensor` with INT8 output leaves two's complement bytes in L1. The FPU's `Int8` format is sign-magnitude
(see `benchmarks/fpu_int8_vs_bf16`). This benchmark answers, by measurement, whether the SFPU quantize step can write
sign-magnitude bytes directly, what that costs in cycles and time, and which input formats and circular-buffer formats it can be combined with
(`results/compat.md`).

| variant | what it does | packer |
|---|---|---|
| `twos_lib` | tt-metal's `quant_int8_tile`: what the op uses now. Byte = `(n + 128) ^ 0x80` | `UInt8` |
| `twos_own` | the same kernel copied into `kernels/variants.hpp` (a control: it must equal `twos_lib`) | `UInt8` |
| `sm_int8pack` | clamp to +-127, `SFPCAST` to sign-magnitude (1 instruction instead of the XOR); the `Int8` packer writes sign + 7 magnitude bits | `Int8` |
| `sm_uint8pack` | clamp to +-127, byte = `abs(n) \| ((n >>> 24) & 0x80)` (4 instructions: abs, shift, and, or) | `UInt8` |

`n = clamp(RNE(x * scale))` is the same in all of them (the replay-buffer kernel of tt-metal's `ckernel_sfpu_quant.h`, with the
lower clamp -128 for the two's complement variants and -127 for the sign-magnitude ones, as `[-Q, Q]` needs no -128).

## Run

```bash
cmake --preset default && cd .build/default && ninja zkml_sign_mag_fpu_compat_bench && cd ../..
TT_METAL_HOME=... benchmarks/sign_mag_fpu_compatibillity/run_all.sh          # check + compat + cycles + wall, about 6 minutes
benchmarks/sign_mag_fpu_compatibillity/run_all.sh compat                     # only what builds and what is correct (1 minute)

# single runs
.build/default/zkml_sign_mag_fpu_compat_bench check 48 [variant] [--input fp32|bf16|fp16] [--extra-cb <format>]
TT_METAL_DEVICE_PROFILER=1 .build/default/zkml_sign_mag_fpu_compat_bench profile sm_uint8pack full 192 32 11   # then analyze.py
.build/default/zkml_sign_mag_fpu_compat_bench wall sm_int8pack sfpu 192 32
```

Run it on the machine with the device (inside the IRD container); `TT_METAL_HOME` must be set. `profile` needs
`TT_METAL_DEVICE_PROFILER=1` and writes `$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv` (`run_all.sh` copies it to
`results/raw/`).

## What is measured

`kernels/compute.cpp` is pass 2 of the 8-bit SFPU quantizer on one core, with the same sequence and DST layout as
`quantize.cpp`: fp32 DST, blocks of 3 tiles plus the scaler in slot 3; per block `copy_tile` of the input and of the scaler,
the quantize call per tile, then `pack_tile`. The data stays in L1 and the loop is repeated `reps` times.

- **`full`**: copy + quantize + pack. This is the cost of pass 2 as it runs in the op (the absmax pass and the scaler are not part of it).
- **`sfpu`**: only the quantize calls, on whatever is in DST. This isolates the SFPU instructions.
- **Cycles**: the device profiler, zone `LOOP` on the three compute threads (unpack, math, pack), median of 10 runs after one
  warm-up run, divided by `tiles * reps`. 1 tile = 1024 elements; clock 1350 MHz. "Slowest" is the thread that bounds the loop.
- **Time**: host wall clock without the profiler, at `reps` and `4 * reps`; the difference divided by `3 * reps * tiles` removes launch and
  copy overhead. Median of 7.
- **Correctness**: the input is chosen so that `x * 32 = m / 2` for every `m` from -256 to 256: every half-integer from -128 to 128 (ties,
  rounded to even; the whole int8 range; values beyond +-127 for the clamp). Each output byte (49,152 of them, 48 tiles) is compared with the
  host answer for its encoding (`twos`: clamp -128..127, byte `q & 0xFF`; sign-magnitude: clamp -127..127, byte `0x80 | -q` for negative q).
- **Input**: fp32 by default; `--input bf16` is the op's default input type; `--input fp16` is Float16.
- **`--extra-cb <format>`** adds one unused circular buffer (c_5) of that format, to ask the JIT which formats may share a program with an `Int8` buffer.

## Results

Files: `results/check.md`, `results/compat.md`, `results/cycles.md` and `cycles.csv` (device profiler), `results/wall.txt`, `results/raw/` (the raw profiler
logs), `results/jit_int8_with_bf16_input.txt`. Machine: Blackhole p100a, one core at 1350 MHz.

### What builds and what is correct (`results/compat.md`, one process per case)

Input format of the quantizer against the output encoding (48 tiles, every output byte compared with the host answer):

| input CB | `twos_lib` (UInt8 packer) | `sm_int8pack` (Int8 packer) | `sm_uint8pack` (UInt8 packer) |
|---|---|---|---|
| fp32 | builds, all bytes correct | builds, all bytes correct | builds, all bytes correct |
| bf16 | builds, all bytes correct | **does not build** | builds, all bytes correct |
| fp16 | builds, 382 wrong bytes | builds, 382 wrong bytes | builds, 382 wrong bytes |

Other circular-buffer formats next to an `Int8` buffer (the `sm_int8pack` program with fp32 input and one unused extra CB):

| extra CB | result |
|---|---|
| fp32, fp16, bfp8_a, int8, uint8, uint16, int32, uint32 | builds, all bytes correct |
| bf16, bfp8_b, bfp4_b | **does not build**: JIT, "All input data-formats must have the same exponent format" |
| int16 | does not build: "No valid conversion from fp32 dest to output format = Int16" |
| tf32 | does not build: "TF32 unsupported atm" (host side) |

So the answer to "can a kernel have a bf16 buffer and an Int8 buffer" is no. The JIT checks the data formats of all circular buffers of the program
(the reader and writer kernels failed with the same message as the compute kernel), and `Int8` counts as an A-exponent format like `Float16` and `Bfp8_a`,
while `Float16_b`, `Bfp8_b` and `Bfp4_b` are B-exponent formats. `UInt8`, `Float32`, `Int32`, `UInt16` and `UInt32` are exempt from the check, which is why
the two's complement output goes through a `UInt8` buffer. (A `Float16_b` buffer and a `UInt8` buffer in the same program work: `twos_lib` and `sm_uint8pack` with bf16 input above.)

**The fp16 input is wrong in all three variants, so it is not the sign-magnitude conversion that fails.** The 382 wrong bytes come from the inputs
`m = -7, -3, 1, 5`, i.e. `x * 32 = -3.5, -1.5, 0.5, 2.5`, exact ties. They are rounded up instead of to even (`-3.5` gives -3, `0.5` gives 1), and only for
`|m| <= 7` (`|x| < 0.125`); the ties in that range for which round-half-to-even also rounds up (`m = -5, -1, 3, 7`), and all ties with `|m| >= 9`, are correct. The same code with fp32 or bf16 input is exact.
The cause was not investigated. For a quantizer whose rounding must be exactly round-half-to-even this makes fp16 input unusable until it is understood.

### Correctness

Every variant produced exactly the expected bytes, 0 wrong of 49,152, for fp32 input (all four) and for bf16 input (all but
`sm_int8pack`, which does not build; fp16 is in the table above). That includes `sm_int8pack`: the `Int8` packer writes sign + 7 magnitude bits of the sign-magnitude value
that `SFPCAST` leaves in DST.

### Cycles and time per tile (192 tiles; 48 tiles gives the same numbers within 0.3%)

Slowest compute thread, cycles per tile (ns per tile in brackets); device profiler. The wall-clock numbers are in `results/wall.txt`
and agree within 1% (for example `twos_lib`, bf16, `full`: 327.7 ns by wall clock against 328.9 ns by cycles).

| input | what | `twos_lib` | `twos_own` | `sm_int8pack` | `sm_uint8pack` |
|---|---|---|---|---|---|
| fp32 | `sfpu` only | 369.7 (273.8 ns) | 369.7 | 369.7 (+0.0%) | 465.7 (344.9 ns, **+26.0%**) |
| fp32 | `full` pass 2 | 540.6 (400.5 ns) | 540.6 | 540.6 (+0.0%) | 635.1 (470.5 ns, **+17.5%**) |
| bf16 | `sfpu` only | 369.7 (273.8 ns) | 369.7 | does not build | 465.7 (344.9 ns, **+26.0%**) |
| bf16 | `full` pass 2 | 444.0 (328.9 ns) | 444.0 | does not build | 539.9 (399.9 ns, **+21.6%**) |

## What it shows

1. **`sm_int8pack` costs nothing**: the same 369.7 cycles per tile for the quantize calls and the same total as the library kernel, because
   `SFPCAST` replaces the XOR one for one.
2. **`sm_uint8pack` costs 96 cycles per tile** (465.7 against 369.7, 3 more instructions in each of the 8 loop iterations of a tile),
   which is 17.5% (fp32 input) to 21.6% (bf16 input) of pass 2 in the full loop (not of the whole operation: the absmax pass is not in this number).
3. **`twos_own` equals `twos_lib` to the decimal**, so the copy in `variants.hpp` adds no overhead and the differences above come only
   from the changed instructions.
4. **The `Int8` pack format cannot be used next to a bf16 input today.** With `--input bf16`, `sm_int8pack` does not build: the JIT
   (`tt_metal/jit_build/data_format.cpp:89`) rejects the program with "All input data-formats must have the same exponent
   format", because `Int8` counts as an A-exponent format and `Float16_b` as a B-exponent one (`UInt8` and `Float32` are exempt; see
   `results/jit_int8_with_bf16_input.txt`). `sm_int8pack` works with fp32 input and with fp16 input (which builds, but see the tie error above), or if the check is changed.
   That is probably why the library packs int8 output through the `UInt8` packer. With bf16 input the working sign-magnitude route is `sm_uint8pack`.
5. **The clamp differs by one value.** The two's complement variants can produce -128 (`0x80`) when `x * 2^k < -127.5`; the sign-magnitude
   ones saturate at -127 (`0xFF`), since sign-magnitude has no -128. PoT quantization never reaches either (`|x| * 2^k <= Q = 127`).

## Limits

- One core, data resident in L1; no DRAM, NoC or other cores. Pass 2 only: the absmax pass, the scaler and the dispatch of the op are not
  included, so the percentages are for the loop, not for the whole quantizer.
- 8-bit output only, `QuantRange::Full` and `M31` are the same here. The scale is a fixed 2^5 and the zero point 0.
- The variants were run with the library's SFPU `INT32_2S_COMP` store mode on a raw 32-bit value; the correct bytes show it does not convert on Blackhole
  (as the ISA documentation says), but no other store mode was tried.
- Power was not measured.
- Cycles and wall time were measured for fp32 and bf16 input only; fp16 input gives wrong bytes (see above), so its speed was not measured.
- The extra-CB test uses an unused buffer; the JIT counts it, but no data flows through it.

## Files

| File | Purpose |
|---|---|
| `bench.cpp` | host: data, program setup, the `check`, `profile` and `wall` commands |
| `kernels/compute.cpp` | pass-2 loop with the variant selected at compile time |
| `kernels/variants.hpp` | the three own SFPU kernels (`twos_own`, `sm_int8pack`, `sm_uint8pack`) |
| `kernels/reader.cpp`, `kernels/writer.cpp` | DRAM to L1 once, and the result back |
| `run_all.sh` | the whole matrix: `check`, `compat`, `cycles`, `wall` |
| `analyze.py` | device-profiler logs to `results/cycles.csv` and `cycles.md` |
| `results/` | tables, raw profiler logs, the JIT error text |
