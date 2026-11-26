#set page(paper: "a4", margin: (x: 1.7cm, y: 1.8cm), numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show raw: set text(size: 8.5pt)
#show raw.where(block: true): it => block(fill: luma(247), stroke: 0.4pt + luma(200), inset: 6pt, radius: 3pt, width: 100%, it)

// ---------- helpers ----------
#let note(t) = block(fill: rgb("#fff7e0"), stroke: 0.5pt + rgb("#d9b44a"), inset: 7pt, radius: 3pt, width: 100%, t)
#let warn(t) = block(fill: rgb("#fdecec"), stroke: 0.5pt + rgb("#d98a8a"), inset: 7pt, radius: 3pt, width: 100%, t)
#let good(t) = block(fill: rgb("#eef7ee"), stroke: 0.5pt + rgb("#8ab98a"), inset: 7pt, radius: 3pt, width: 100%, t)
#let blk(t, fill: luma(245), w: auto, h: auto, sz: 8pt) = box(stroke: 0.6pt, inset: 5pt, radius: 3pt, fill: fill, width: w, height: h, align(center + horizon, text(size: sz, t)))
#let arrd = text(size: 12pt)[#sym.arrow.b]
#let arrr = text(size: 12pt)[#sym.arrow.r]
#let isa = text(fill: rgb("#1f5f9f"), size: 7.5pt, weight: "bold")[\[ISA\]]
#let tm = text(fill: rgb("#7a4b00"), size: 7.5pt, weight: "bold")[\[tt-metal\]]
#let ms = text(fill: rgb("#1c6b2a"), size: 7.5pt, weight: "bold")[\[measured\]]
#let tbl(cols, ..args) = table(columns: cols, inset: 4pt, stroke: 0.4pt + luma(170), ..args)
#let cell(t, c) = table.cell(fill: c, t)

#align(center)[
  #text(size: 18pt, weight: "bold")[The Vector Unit (SFPU) on Blackhole] \
  #v(2pt)
  #text(size: 11pt)[A working reference: lanes and registers, how instructions reach the unit, and `SFPLOADMACRO`]
]
#v(6pt)

This document is a reference for writing SFPU kernels by hand (raw `TTI_SFPxxx` instructions, as in tt-metal's `ckernel_sfpu_*.h`), as opposed to going through the `sfpi` compiler. It is general: it is not tied to one operation, though it uses real kernels (tt-metal's and ours) as worked examples. Statements are tagged by where they come from:

#tbl((auto, 1fr),
  [#isa], [The Tenstorrent ISA documentation, `github.com/tenstorrent/tt-isa-documentation`, Blackhole A0 pages under `BlackholeA0/TensixTile/` (and the shared Wormhole pages they point to). This is the authority on what the hardware does.],
  [#tm], [tt-metal / tt-llk sources (`tt_metal/hw/ckernels/blackhole/...`, `tt_metal/tt-llk/tt_llk_blackhole/...`, `runtime/sfpi/include`): how software uses it.],
  [#ms], [Something we measured on a Blackhole p100a (one core, 1.35 GHz, device profiler cycles) while building a quantized-matmul layer. Measurements are marked so they are not mistaken for architecture facts.]
)

#outline(indent: 1.2em, depth: 2)
#pagebreak()

= The SFPU: what it is and how it is laid out

== Where it sits in the Tensix tile

A Tensix tile contains five small RISC-V cores (the "baby RISCVs") and a *Tensix coprocessor*. The coprocessor does not fetch instructions: the RISC-V cores _push_ 32-bit Tensix instructions into it. The Vector Unit (SFPU) is one of nine backend execution units of that coprocessor. #isa

#v(4pt)
#align(center)[
#block(width: 100%, stroke: 0.6pt + luma(120), radius: 5pt, inset: 8pt, fill: luma(252))[
  #grid(columns: (1.05fr, 0.15fr, 1.4fr, 0.15fr, 1.5fr), column-gutter: 3pt, align: center + horizon,
    [#text(weight: "bold", size: 8.5pt)[baby RISC-V cores] \
     #v(2pt)
     #blk[RISCV T0 \ (unpack)]
     #v(2pt)
     #blk(fill: rgb("#dbeafe"))[RISCV T1 \ (math, pushes SFPU code)]
     #v(2pt)
     #blk[RISCV T2 \ (pack)]
     #v(2pt)
     #blk[RISCV B / NC \ (dataflow, boot)]],
    [#arrr \ #text(size: 6.5pt)[`sw` / \ `.ttinsn`]],
    [#text(weight: "bold", size: 8.5pt)[frontend, 1 copy per thread] \
     #v(2pt)
     #blk(w: 100%)[Tensix T0 : FIFO #arrr MOP #arrr Replay #arrr Wait Gate]
     #v(2pt)
     #blk(w: 100%, fill: rgb("#dbeafe"))[Tensix T1 : FIFO #arrr MOP #arrr Replay #arrr Wait Gate]
     #v(2pt)
     #blk(w: 100%)[Tensix T2 : FIFO #arrr MOP #arrr Replay #arrr Wait Gate]
     #v(2pt)
     #text(size: 7pt)[strictly in order; one instruction per thread per cycle leaves the gate]],
    [#arrr],
    [#text(weight: "bold", size: 8.5pt)[backend (mostly shared by the 3 threads)] \
     #v(2pt)
     #grid(columns: (1fr, 1fr), gutter: 3pt,
       blk[Unpackers], blk[Matrix Unit (FPU)],
       blk[Packers], blk(fill: rgb("#fde68a"))[*Vector Unit (SFPU)*],
       blk[Scalar Unit (ThCon)], blk[Sync Unit],
       blk[Configuration Unit], blk[Mover],
       grid.cell(colspan: 2, blk[Miscellaneous Unit]))
     #v(3pt)
     #blk(w: 100%, fill: rgb("#f3e8ff"))[shared state: `Dst` (the big register file), `LReg` (SFPU registers), SrcA/SrcB]]
  )
]
]
#text(size: 8pt)[_Figure 1. Block diagram (own drawing, simplified from `Diagrams/Out/TensixFrontend_BH.svg` and `TensixBackend.svg` of the ISA repository)._]

Two facts from this picture matter for everything below:
+ *The SFPU is shared.* There is one `LReg` file and one `Dst` for the whole tile, not one per thread. Which thread uses them when is up to software. By convention T0 unpacks (L1 #sym.arrow.r `Dst` / SrcA / SrcB), T1 does the math (the FPU and the SFPU) and T2 packs (`Dst` #sym.arrow.r L1), but "any thread can access any backend execution unit". #isa
+ *The SFPU can only accept one instruction per cycle from the outside world*, summed over all threads. By default four fifths of it is idle (section 1.6). #isa

== 32 lanes of 32 bits

The SFPU is a SIMD engine of *32 lanes*, each lane 32 bits wide, at 1.35 GHz on Blackhole (1 GHz on Wormhole). Every vector instruction works on all 32 lanes at once. A single `SFPMAD` is 32 multiplies plus 32 adds, 64 FP32 operations per cycle: 0.0864 TFLOP/s per SFPU. #isa

Lanes are usually viewed as a *4 x 8 grid*. This makes no difference to lanewise instructions, but cross-lane instructions move data horizontally or vertically in this grid:

#align(center)[
#grid(columns: (auto, auto), column-gutter: 18pt, align: horizon,
  table(columns: 8, inset: 3.5pt, stroke: 0.4pt + luma(170), align: center,
    ..range(32).map(i => [#text(size: 7.5pt)[#i]])),
  [#text(size: 8.5pt)[
    Row 0 of the grid is lanes 0 to 7, row 1 is 8 to 15, \
    row 2 is 16 to 23, row 3 is 24 to 31. \
    *Lane `l` = grid row `l / 8`, grid column `l % 8`.*]]
)
]
#text(size: 8pt)[_Figure 2. Lane numbering (ISA, `LReg.md`)._]

Per-lane state exists too: *lane flags* (predication, section 1.5), a flag stack, a PRNG, and the lane config. All of these are what makes `v_if` / conditional code possible without branches.

== The register file: `LReg`

The registers visible to SFPU instructions are the global `LReg[17][32]` (17 registers of 32 lanes of 32 bits). The 17 are not uniform: #isa

#tbl((auto, 1fr, auto),
  [*Register*], [*Content / how it can be used*], [*Writable by*],
  [`LReg0` ... `LReg7`], [Eight general registers, 32 lanes x 32 bits. All instruction operands refer to these (plus the constants below). `LReg7` is special only by convention: ours keeps a running value across calls, and the indirect modes of `SFPMAD` read `LReg7`.], [any instruction],
  [`LReg8`], [Constant: `0.8373` (as fp32) in all lanes.], [read-only],
  [`LReg9`], [Constant: zero (same bit pattern for every data type). tt-metal calls it `LCONST_0`.], [read-only],
  [`LReg10`], [Constant: `1.0f`. tt-metal calls it `LCONST_1`.], [read-only],
  [`LReg11` ... `LReg14`], [Four "programmable constants", 8 lanes of 32 bits broadcast to 32 lanes. They are written *only* with `SFPCONFIG`, which broadcasts 8 lanes of `LReg0`. The `sfpi` compiler reserves `LReg11` for `-1.0f` in every lane; tt-metal's `vConstIntPrgm0..2` / `vConstFloatPrgm0..2` map onto these.], [`SFPCONFIG` only],
  [`LReg15`], [Constant: lane `i` contains the integer `i * 2` (0, 2, 4, ..., 62): a lane-id vector.], [read-only],
  [`LReg16`], [A 32-lane "bonus" register that can only be *written* by instructions scheduled by `SFPLOADMACRO` and only *read* by an `SFPSTORE` scheduled by `SFPLOADMACRO` (section 3).], [macros only],
)

Two traps hide in this table:
- *Operand fields are 4 bits, so 12 to 15 look usable as a destination, but a destination of 12 to 15 does not write a register*: such an instruction is captured as a macro instruction template instead of executing (section 3.3). Constants 8 to 10 and 15 can be sources only.
- Because `LReg11` ... `LReg14` are written by `SFPCONFIG` from `LReg0`, loading a constant into them costs a `SFPLOADI` into `LReg0`, then the `SFPCONFIG`, and then (per the documentation) a spare `SFPNOP`.

=== Data types in `LReg`

A 32-bit datum in `LReg` is one of the following, and software may freely bit-cast between them: #isa

#tbl((auto, 1fr),
  [*FP32*], [1 sign, 8 exponent, 23 mantissa bits. LReg arithmetic is closer to IEEE 754 than the FPU's.],
  [*uint32*], [Plain unsigned integer.],
  [*int32, two's complement*], [Plain signed integer.],
  [*sign-magnitude int32*], [Bit 31 = sign, bits 0 to 30 = magnitude. This is the format of `Dst`'s "Integer 32" and of the FPU's integer results. `-0` (`0x80000000`) is a valid, distinct bit pattern.],
)

All integer types agree on non-negative values below 2^31, which is why many tricks work on any of them. Which instructions interpret which type matters: `SFPSWAP`'s min/max and `SFPGT`/`SFPLE` compare as *sign-magnitude* (which is also the correct order for FP32), `SFPIADD`, `SFPSHFT` and `SFPCAST` are integer instructions, `SFPMAD` is fp32.

#note[*Blackhole `SFPLOAD`/`SFPSTORE` with integer 32 are raw.* On Wormhole the 32-bit integer load/store modes converted between sign-magnitude (in `Dst`) and two's complement (in `LReg`). On Blackhole `MOD0_FMT_INT32`, `INT32_ALL`, `INT32_SM` no longer convert: an integer `Dst` value arrives in `LReg` as the raw sign-magnitude bits. Convert explicitly with `SFPCAST` (`SFPCAST_MOD1_SM32_TO_INT32 = 2`, `SFPCAST_MOD1_INT32_TO_SM32 = 3`; both map `-2^31` and `-0` onto each other). #isa]

== `Dst` and how rows land in lanes

The SFPU never touches L1. It exchanges data with the `Dst` register file, which the unpackers fill and the packers drain. `Dst` is 1024 rows x 16 columns of 16-bit cells, or viewed as 32-bit data, 512 rows x 16 columns (a 32-bit datum is stored as two 16-bit halves 8 rows apart). #isa

A *tile* is 32 x 32 = 1024 datums made of four 16 x 16 *faces*. In 32-bit mode one tile takes 64 `Dst32b` rows (16 rows per face), so with the usual half-sync double buffering an fp32 `Dst` holds 8 tiles, 4 per half. `SFPLOAD` moves 32 datums, "four consecutive rows x eight columns", in one instruction:

#align(center)[
#grid(columns: (auto, auto), column-gutter: 16pt, align: horizon,
  [#text(size: 8pt, weight: "bold")[one 16 x 16 face, 8 loads]
   #v(2pt)
   #table(columns: 17, inset: 2pt, stroke: 0.4pt + luma(170), align: center,
     [], ..range(16).map(c => [#text(size: 6pt)[c#c]]),
     ..range(16).map(r => ([#text(size: 6pt)[r#r]], ..range(16).map(c => {
       let g = calc.floor(r / 4)
       let par = calc.rem(c, 2)
       let unit = 2 * g + par
       let cols = (rgb("#dbeafe"), rgb("#fde68a"), rgb("#dcfce7"), rgb("#fce7f3"), rgb("#e0e7ff"), rgb("#fef9c3"), rgb("#ffedd5"), rgb("#ccfbf1"))
       cell(text(size: 6pt)[#unit], cols.at(unit))
     }))).flatten())],
  [#text(size: 8.5pt)[
    The number in each cell is the *unit* `d` (0 to 7) \
    of the `SFPLOAD` that reads it: \
    - rows come in groups of four (`d / 2`), \
    - `d` even reads the even columns, odd the odd columns, \
    - the 8 lanes of a grid row are the 8 columns of one parity. \
    A tile has 4 faces, so 32 units: unit `d` of the tile \
    is at address offset `2 d` (see below).]]
)
]
#text(size: 8pt)[_Figure 3. Which `SFPLOAD` reads which cell of a face (derived from the `SFPLOAD` functional model)._]

The address of a load or store is a 10-bit immediate plus offsets. From the `SFPLOAD` functional model: #isa

```text
Addr   = Imm10 + DEST_TARGET_REG_CFG_MATH_Offset      // per-thread config
Addr  += RWC.Dst + DEST_REGW_BASE_Base + (SpOffset & 3)   // the "Dst" read/write counter + base
Row    = (Addr & ~3) + (Lane / 8)      // bits 9..2 pick an aligned group of 4 rows
Column = (Lane & 7) * 2 + ((Addr >> 1) & 1)   // bit 1 picks even/odd columns, bit 0 is unused
```

So Imm10 is in units of *rows*: `+4` is the next group of four rows, `+2` flips to the odd columns, and bit 0 is ignored by `SFPLOAD` (`SFPLOADMACRO` uses it, section 3). For fp32 data one face is 16 rows, hence unit `d` of a tile is at Imm10 `= 2 d`: units 0 and 1 are the even and odd columns of rows 0 to 3, units 2 and 3 are rows 4 to 7, and unit 8 (offset 16) is the first unit of face 1.

Three ways of moving through a tile exist, and all appear in real kernels: #tm
+ `sfpi::dst_reg++` (the SFPI compiler) advances an address counter by 2 per use, the unit stride. The ckernel face functions are written for 8 iterations (one face) and the wrapper calls them for each of the 4 faces, advancing the `Dst` RWC by 16 rows in between (`_llk_math_eltwise_sfpu_inc_dst_face_addr_`).
+ An `AddrMod` (3-bit field of the instruction) selects an increment configured with `addr_mod_t{.dest = {.incr = 2}}.set(ADDR_MOD_6)`, applied after the access. `ADDR_MOD_7` is configured with zero increment by the standard init, so "load with `ADDR_MOD_7` at an explicit offset" is the idiom for random access, and `ADDR_MOD_6` with `incr = 2` the idiom for a store that steps to the next unit.
+ Explicit offsets in an unrolled loop (`TT_SFPLOAD(.., ADDR_MOD_7, 2 * d)`), which lets one call cover a whole tile (`VectorMode::None`) and is what the `SFPLOADMACRO` kernels in section 3 do.

== Lane predication: flags, `ROW_MASK`, the flag stack

Not every lane has to execute an instruction. Whether lane `i` executes is `IsLaneEnabled(i)`: #isa

```c
if (LaneConfig[i & 7].ROW_MASK.Bit[i / 8]) return false;   // masked rows of the 4x8 grid
if (UseLaneFlagsForLaneEnable[i])          return LaneFlags[i];
return true;
```

`LaneFlags` is set by `SFPSETCC` (compare a register with zero: `< 0`, `!= 0`, `>= 0`, `== 0`), `SFPIADD` (sign of the result, unless `CC_NONE`), `SFPGT`/`SFPLE`, `SFPLZ`, `SFPEXEXP`, and `SFPENCC`. `UseLaneFlagsForLaneEnable` is turned on and off with `SFPENCC` (it starts off). `SFPPUSHC`, `SFPCOMPC` (the `else`) and `SFPPOPC` implement a per-lane flag *stack of depth 8* for nested `if`/`else`. Software must not overflow or underflow it.

Typical hand-written pattern (zero the negative lanes of `LReg0`, ReLU style):

```cpp
TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);           // flags = (LReg0 < 0), mod1 0 = LT0
TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);  // only the flagged lanes are written
TTI_SFPENCC(0, 0, 0, 0);                         // back to "all lanes enabled"
```

#warn[*The flags are global state, not per-row state.* A lane that is disabled is disabled for _every_ instruction that executes while the flags are set, including instructions that `SFPLOADMACRO` scheduled for another row. That is the main reason condition-code code and overlapped macro schedules do not mix (section 3.8).]

`SFPLOAD` into a disabled lane does not write the register, and `SFPSTORE` from a disabled lane does not write `Dst` (the `LaneConfig` bits `BLOCK_SFPU_RD_FROM_DEST` / `BLOCK_DEST_WR_FROM_SFPU` additionally switch whole directions off). #isa

== The five sub-units

Inside the SFPU there are five sub-units. Every instruction belongs to one of them; this is what `SFPLOADMACRO` exploits. #isa

#align(center)[
#grid(columns: (1fr, 1fr, 1fr, 1fr, 1fr), column-gutter: 4pt,
  blk(fill: rgb("#dbeafe"), sz: 7.5pt)[*Load* \ `SFPLOAD` \ `SFPLOADI` \ `SFPLOADMACRO` \ `SFPNOP`],
  blk(fill: rgb("#fde68a"), sz: 7.5pt)[*Simple* \ `SFPIADD` `SFPSHFT` \ `SFPAND/OR/XOR/NOT` \ `SFPCAST` `SFPABS` \ `SFPSETCC` `SFPENCC` \ `SFPLZ` `SFPEXEXP` ... \ `SFPSWAP` (2 cycles)],
  blk(fill: rgb("#dcfce7"), sz: 7.5pt)[*MAD* \ `SFPMAD` `SFPADD` \ `SFPMUL` `SFPMULI` \ `SFPADDI` `SFPMUL24` \ `SFPLUT` `SFPLUTFP32`],
  blk(fill: rgb("#fce7f3"), sz: 7.5pt)[*Round* \ `SFPSTOCHRND` \ `SFPSHFT2` \ `SFPNOP`],
  blk(fill: rgb("#e0e7ff"), sz: 7.5pt)[*Store* \ `SFPSTORE`],
)
]
#text(size: 8pt)[_Figure 4. The five sub-units and the instructions each one executes (the columns of the `SFPLOADMACRO` table in the ISA)._]

Notes: (a) The Simple column is the big one: all integer arithmetic, bit manipulation, shifts, conversions, comparisons and lane-flag instructions. (b) The MAD column is floating point only (plus the 24-bit integer multiply `SFPMUL24`). (c) The Round column is small and rarely useful by hand; `SFPSTOCHRND` rounds and clamps (to int8/uint8/int16, bf16/tf32), but its round-to-nearest breaks ties *away from zero* and its round-to-zero mode has a documented hardware bug when the shift discards 23 or more bits. (d) The Store column only has `SFPSTORE`.

The unit can accept *one* instruction per cycle from the instruction stream and, via `SFPLOADMACRO` only, up to five per cycle (one load-style plus one per column). #isa

== Instruction reference (what is needed day to day)

Latency/throughput numbers are from the ISA's `VectorUnit.md`; "IPC" is instructions per cycle for that instruction alone. #isa

#tbl((auto, auto, auto, 1fr),
  [*Instruction*], [*IPC*], [*Latency*], [*Semantics and traps*],
  [`SFPLOAD`], [1], [1], [`VD = Dst[4 rows x 8 cols]`; `Mod0` selects the conversion (`FP32`, `BF16`, `FP16`, `INT32` (raw), `INT8` (sign-magnitude, range +-255 on BH), `UINT16`, ...). Reads of a `Dst` row that is *undefined* (`DstRowValid`, e.g. after `ZEROACC`) are undefined behaviour.],
  [`SFPSTORE`], [1], [1], [`Dst[4 rows x 8 cols] = VD`, same `Mod0` families. On BH the integer modes are raw.],
  [`SFPLOADI`], [1], [1], [Load a 16-bit immediate into `VD`: as bf16, fp16, unsigned/signed short, or into the upper/lower half only (two of these build a 32-bit constant). Half-writes after a `SFPMAD` to the same register are unpredictable (below).],
  [`SFPMAD` / `SFPADD` / `SFPMUL`], [1], [2], [`VD = VA * VB + VC` (fp32, with signs selectable). `SFPMUL` is a `SFPMAD` with a zero addend. The modes `INDIRECT_VA` / `INDIRECT_VD` take the register number from `LReg7`.],
  [`SFPMUL24`], [1], [2], [24-bit integer multiply: `VD = (VA * VB) & 0x7fffff`, or the high part `>> 23`.],
  [`SFPIADD`], [1], [1], [`VD = VC +- VD` (register form) or `VD = VC + sign-extended Imm12` (immediate form, `SFPIADD_MOD1_ARG_IMM`). Sets lane flags from the result's sign unless `CC_NONE`. Always pass `SFPIADD_MOD1_CC_NONE` unless you want the flags.],
  [`SFPAND` `SFPOR` `SFPXOR` `SFPNOT`], [1], [1], [Bitwise. `SFPAND`/`SFPOR` without `USE_VB` use `VD` as the other operand (`VD &= VC`).],
  [`SFPSHFT`], [1], [1], [Left shift, logical right shift, *arithmetic* right shift (BH). Amount from the sign of a register (positive = left, negative = right; low 5 bits) or from `Imm12`. See the mode table below.],
  [`SFPLZ`], [1], [1], [Count leading zeros, or set flags on zero/non-zero.],
  [`SFPCAST`], [1], [1], [`SM32 <-> INT32` (the `-0` / `-2^31` swap), `SM32 -> FP32` (RNE or stochastic), `INT32_ABS`. The `FP32 -> SM32` modes 4/5 exist only on Quasar; they are not available on Blackhole.],
  [`SFPABS`], [1], [1], [`VD = |VC|` for fp32 (`MOD1_FLOAT`) or for int32.],
  [`SFPSWAP`], [<= 1], [2], [See below: min/max, argmin/argmax, plain swap. Keeps the Simple sub-unit busy about 2 cycles.],
  [`SFPSETCC`/`SFPENCC`/ `SFPGT`/`SFPLE`/...], [1], [1], [Lane flags, see section 1.5.],
  [`SFPSTOCHRND`], [1], [1], [Float or int -> narrower type with rounding and clamping, Round sub-unit.],
  [`SFPTRANSP`], [1], [1], [Transpose rows within each column of `LReg0..3` and of `LReg4..7` (cross-lane).],
  [`SFPSHFT2`], [<= 1], [1 or 2], [Lane rotate / shift (`SHFLROR1`, `SHFLSHR1`) and register-chain copies. Cross-lane.],
  [`SFPCONFIG`], [1], [<= 2], [Writes `LReg11..14`, the macro config, or the lane config from `LReg0` (8 lanes broadcast) or an immediate.],
  [`SFPNOP`], [1], [1], [Occupies the Vector Unit for one cycle. See sections 2.4 and 3.7 for why it matters.],
)

=== `SFPSHFT` modes

`TTI_SFPSHFT(imm12, lreg_c, lreg_dest, mod1)`. From the functional model: #isa

#tbl((auto, 1fr),
  [`mod1 = 0`], [`VD = VD shifted by LReg[VC]`: the *amount* is the signed value in `VC` (positive = left by `v & 31`, negative = right by `(-v) & 31`, logical), the value shifted is `VD`.],
  [`mod1 = 1` (`ARG_IMM`)], [The amount is `Imm12` (signed, same convention): `VD = VD shifted by Imm12`.],
  [`mod1 = 2` (`ARITHMETIC`)], [A negative amount shifts right *arithmetically* (sign-extending). New on Blackhole.],
  [`mod1 = 4` (`ARG_IMM_USE_VC`)], [Only with `ARG_IMM`: the value shifted is taken from `VC` instead of `VD`, so `VD = VC shifted by Imm12` leaves `VC` intact. So `mod1 = 5` is "copy and shift by an immediate" in one instruction.],
)

Typical uses: `TTI_SFPSHFT((-24) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG1, 5)` is `LReg1 = LReg0 >>> 24` (copy and logical right shift); `TTI_SFPSHFT(0, p_sfpu::LREG7, p_sfpu::LREG0, 0)` is `LReg0 <<= LReg7` (variable shift by a per-lane register amount).

=== `SFPSWAP` in one place

`TTI_SFPSWAP(0, VC, VD, mod1)` compares `VC` and `VD` lanewise *as sign-magnitude* (fp32 or 32-bit sign-magnitude ints) and exchanges them: #isa

#tbl((auto, 1fr),
  [`mod1 = 0` `SWAP`], [Unconditional swap of `VC` and `VD`.],
  [`mod1 = 1` `VEC_MIN_MAX`], [`VD = min`, `VC = max`, all lanes.],
  [`mod1 = 9`], [`VD = max`, `VC = min`, all lanes. No enum exists for it in `sfpi`, but tt-metal's own kernels use it.],
  [`mod1 = 2..8`], [Sub-vector variants: different lanes get min or max (used for cross-lane sorting networks).],
)

Details that bite: (1) both registers are written, so the `VC` register (usually a scratch row) is clobbered with the min; the result lands *two cycles later*, so a register must not be reloaded in that window. (2) For equal values it may swap anyway (equal positive values swap on max lanes), so do not use it where identity of equal elements matters. (3) The lane config bit `EXCHANGE_SRCB_SRCC` flips min and max and `ENABLE_DEST_INDEX` turns it into argmin/argmax (touching `LReg4..7`): both are off by default, and nothing else should turn them on. (4) The hardware reads `VC`/`VD` in cycle 1 *and again in cycle 2*, and the automatic stall logic does not know about the first read. (5) To compare *integers*, non-negative two's complement values compare correctly because they have the same bits as sign-magnitude; negative two's complement values do not, tt-metal's int32 max/min inverts bits around the swap for that case. #tm

=== Hazards not covered by automatic stalling

The hardware inserts a stall when an instruction reads a register that an in-flight `SFPMAD` writes. That logic has holes. Per the ISA, after an `SFPMAD` that writes a register, you must separate the consumer by at least one cycle (an `SFPNOP`, or an unrelated instruction) when the consumer is: `SFPIADD` (it does not know `SFPIADD` reads `VD`), `SFPSHFT` (same), `SFPAND`/`SFPOR` with `USE_VB`, `SFPSETEXP`/`SFPSETMAN`/`SFPSETSGN` (modifier sense inverted), `SFPCONFIG` reading `LReg0`, `SFPSHFT2` in some modes, `SFPSWAP` (all modes but plain swap), and `SFPLOADI`/`SFPLOAD` partial (hi/lo half) writes. #isa

That list is exactly why you will see `SFPMAD; SFPNOP; SFPIADD` in tt-metal kernels, and why `SFPMAD` followed by a plain `SFPSTORE` of the same register needs no `SFPNOP` (the store is protected by the automatic stall, at the price of a one-cycle bubble).

#pagebreak()

= How instructions reach the SFPU

== The path: RISC-V core, FIFO, frontend, Wait Gate, SFPU

The SFPU does not run a program. A baby RISC-V core (T1, the math core) executes ordinary RISC-V code, and wherever the C++ says `TTI_SFPMAD(...)`, that code *stores a 32-bit instruction word to a special address* (`INSTRN_BUF_BASE`, `0xFFE40000`). The word lands in a FIFO of the matching Tensix thread; the coprocessor executes it some cycles later, asynchronously. #isa

#align(center)[
#block(width: 100%, stroke: 0.6pt + luma(120), radius: 5pt, inset: 8pt, fill: luma(252))[
  #grid(columns: (1fr, 0.12fr, 1fr, 0.12fr, 1.1fr, 0.12fr, 1fr, 0.12fr, 1.1fr), column-gutter: 2pt, align: center + horizon,
    blk(w: 100%, fill: rgb("#dbeafe"))[RISC-V T1 executes \ `.ttinsn W` \ (= `sw W, INSTRN_BUF`)],
    arrr,
    blk(w: 100%)[FIFO \ (32 entries, first stage) \ RISC-V stalls when full],
    arrr,
    blk(w: 100%)[MOP expander, \ Replay expander \ (optional: 1 push #arrr many instructions)],
    arrr,
    blk(w: 100%, fill: rgb("#fee2e2"))[Wait Gate \ in order; may hold \ the instruction (STALLWAIT, \ busy unit, other thread)],
    arrr,
    blk(w: 100%, fill: rgb("#fde68a"))[SFPU \ 1 instruction \ per cycle]
  )
]
]
#text(size: 8pt)[_Figure 5. The path of one SFPU instruction (ISA `PushTensixInstruction.md`, `Tensix Coprocessor README`, `WaitGate.md`)._]

What follows from the ISA text:
- *The RISC-V runs ahead.* After the push, the RISC-V proceeds; the instruction executes "some cycles" later, as FIFOs allow. Do not assume the next RISC-V instruction sees the result. Reading a register that a pushed instruction will write is not a hazard on the RISC-V side, it is simply not ordered. #isa
- *Throughput limits.* The RISC-V can push one instruction per `sw`; the baby RISC-V's `.ttinsn` extension encodes the store of a 32-bit immediate as one RISC-V instruction, and up to four adjacent `.ttinsn` can be *fused* by the instruction cache into one cycle. The coprocessor, however, removes at most one instruction per thread per cycle. So a long straight line of `TTI_` instructions can run at one SFPU instruction per cycle, as long as it is fetched without loops, calls or computed operands in between. #isa
- *The first FIFO has 32 entries* (28 + a 4-wide fused push). When it is full the RISC-V stalls on the next push. This bounds how far ahead the RISC-V can get. #isa
- *The frontend is strictly in order* (FIFO, MOP expander, Replay expander, Wait Gate). Reordering can only start after the Wait Gate, between different backend units. Software uses the Wait Gate (`STALLWAIT`, `SEMWAIT`) to prevent undesirable reordering. #isa

== The Wait Gate and the rest of the frontend

At the end of each thread's frontend sits the *Wait Gate*, which is where "the backend is not ready" turns into a stall. An instruction waits there if: #isa
- the backend unit it needs is busy with a previous instruction;
- another thread wants the same unit in the same cycle (the unit accepts one per cycle, so one thread proceeds);
- the previous instruction of this thread was a Scalar Unit (ThCon) instruction that is still executing;
- a `STALLWAIT` / `SEMWAIT` latched earlier is still unsatisfied *and its block mask catches this instruction*.

`STALLWAIT(BlockMask, ConditionMask)` is the explicit tool. The *block mask* names which instruction classes are held at the gate, the *condition mask* names the conditions that must all be satisfied to release them. The two bits that matter for the SFPU: #isa

#tbl((auto, 1fr),
  [*B8*], [Block thread from starting new Vector Unit (SFPU) instructions. (B6 blocks Matrix Unit / FPU instructions, B2 packer, B3 unpacker.) A `NOP` is only blocked if *all* mask bits are set: Tensix `NOP` slips through a `STALLWAIT`; `SFPNOP` is an SFPU instruction and is held.],
  [*C1* / *C2*], [Wait while the current thread has an instruction in Unpacker 0 / Unpacker 1's pipeline.],
  [*C3*], [Wait while the current thread has an instruction in the packer pipeline.],
  [*C4*], [Wait while the current thread has an instruction in the Matrix Unit (FPU) pipeline.],
  [*C11*], [Wait while the current thread has an instruction in the Vector Unit (SFPU) pipeline.],
)

Idioms: `STALLWAIT(B8, C4)` holds SFPU instructions until the FPU is done, which is exactly what tt-metal's `_llk_math_eltwise_sfpu_start_` does (`TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH)`). Handing a `Dst` half to another thread is `STALLWAIT(B1, C4 [| C11])` followed by `SEMPOST`. Note the documented one-cycle lag: the instruction right after `STALLWAIT` is always subject to the block mask for at least one cycle. #isa #tm

== The MOP and Replay expanders

Both are optional, both sit before the Wait Gate, and both turn one pushed instruction into many: #isa
- *`REPLAY(Index, Count, Exec, Load)`*: records the next `Count` instructions into a per-thread replay buffer and/or replays a recorded sequence (see `REPLAY.md` for the buffer size and the exact `Count` rules, which differ between Wormhole and Blackhole). A replayed sequence of 8 `SFPU` instructions costs the RISC-V one push.
- *`MOP`* (macro-op): a small loop engine with templates, which the LLK layer uses for unpack, FPU and pack loops. (Not to be confused with `SFPLOADMACRO`, which is unrelated.)

Their purpose is to free the RISC-V core from being the instruction source. SFPU kernels in tt-metal use `REPLAY` for fixed straight-line sequences (for example the `quant_int8` kernel loads a replay buffer once and replays it per row group). The cost is that the buffer is shared state per thread and a replay is, like everything in the frontend, in order. Neither expander removes the *one instruction per cycle* limit of the SFPU. #tm

== Issue slots

Everything in this document uses one unit of time: the *issue slot*, the cycle in which the SFPU accepts one instruction from the stream. With back-to-back `TTI_` instructions a kernel executes at one instruction per slot, so *the cost of a kernel is the number of instructions it issues* (plus stalls). The plain version of a typical "scale, round, convert, store" row looks like this ( #ms the cycle numbers are from our own kernels):

#align(center)[
#table(columns: (auto, auto, 1fr, auto), inset: 4pt, stroke: 0.4pt + luma(170), align: (center, left, left, left),
  [*slot*], [*instruction issued*], [*what the hardware does*], [*sub-unit*],
  [0], [`SFPLOAD  a`], [row of `Dst` #sym.arrow.r `LReg a`], [Load],
  [1], [`SFPMAD   a = a*s + m`], [multiply-add, 2 cycles], [MAD],
  [2], [`SFPNOP`], [wait for the MAD result (`SFPIADD` is not auto-stalled)], [--],
  [3], [`SFPIADD  a -= m_bits`], [integer subtract], [Simple],
  [4], [`SFPCAST  a to sign-magnitude`], [convert], [Simple],
  [5], [`SFPSTORE a`], [register #sym.arrow.r `Dst`], [Store],
)
]
#text(size: 8pt)[_Table 1. One row (4 x 8 lanes) of a quantize kernel, 6 slots; four of the five sub-units idle in every slot._]

A tile has 32 such rows, so the kernel costs about 32 x 6 = 192 slots per tile no matter how fast the individual instructions are. Reordering the instructions of consecutive rows to remove the `SFPNOP` (software pipelining: do the load and multiply of row `r + 1` while row `r` waits for its multiply-add result) gets 5 slots per row. `SFPLOADMACRO` (section 3) gets 2.

== The ways instructions are written in tt-metal

#tbl((auto, 1fr),
  [`TTI_SFPxxx(args)`], [*Immediate* form: the whole instruction word is a compile-time constant, emitted as one `.ttinsn` (one RISC-V instruction, fusable). Arguments must be constant expressions. This is the fast form.],
  [`TT_SFPxxx(args)`], [*Runtime* form: the word is built from run-time values and stored with `sw`. If the arguments are compile-time constants after inlining and unrolling, the compiler folds it into the same code as `TTI_`; if they are genuinely runtime (a loop counter), each instruction costs several RISC-V instructions to build, and the push rate can fall below one per cycle.],
  [`sfpi::vFloat` / `dst_reg` / `v_if`], [The `sfpi` compiler front end. It allocates `LReg`s, inserts the `SFPNOP`s and the flag handling and emits `SFPLOAD`/`SFPSTORE` for `dst_reg[i]`. Convenient, but it does not use macros, and it chooses its own registers, so hand-written code that shares a kernel with `sfpi` code must declare the registers it uses (`sfpi::l_reg[...].in_use()`).],
  [`_llk_math_eltwise_unary_sfpu_params_(fn, idst, mode)`], [The wrapper nearly every ckernel call goes through: sets the `Dst` write address for tile `idst`, issues the `STALLWAIT(B8, C4)`, then calls `fn` once (`VectorMode::None`) or once per face (`RC`), advancing the `Dst` RWC between faces, and finally clears the address.],
)

=== Rules of thumb for hand-written kernels

+ Count instructions, not lines of C++: that is the cost model when you are issue-bound. Then count *Simple-unit* instructions, because it is the busiest column (section 3.8).
+ Prefer `TTI_` and unrolled loops. A `for` loop with a runtime counter turns into RISC-V branches and address arithmetic between pushes.
+ Keep constants in registers (`LReg4..6` for example) loaded once in the init, not per row.
+ Never rely on the `SFPNOP` count you "need" on paper: check the hazard list in section 1.7.
+ Keep a plain `TTI_` reference implementation of any macro kernel: it is the oracle for debugging (section 3.9).

#pagebreak()

= Optimizing with `SFPLOADMACRO`

== What it is

`SFPLOADMACRO` is an `SFPLOAD` that also *schedules up to four more instructions*, at most one per sub-unit column (Simple, MAD, Round, Store), to execute automatically in later cycles. What is scheduled comes from a small configuration written once, ahead of time, with `SFPCONFIG`. Because the scheduled instructions come from the unit's own state and not from the instruction stream, they cost no issue slots. #isa

#good[*What it buys.* Without it, a row of work costs one issue slot per instruction (5 for the quantize row above). With it, a row costs one slot for the macro plus one slot for every instruction you still issue normally. If the work spreads over the sub-units, the unit can do up to five things per cycle, and the kernel runs at the speed of its busiest sub-unit instead of the speed of the instruction stream.]

The hardware keeps this state per lane (all lanes normally identical): #isa

```c
struct {
  struct { uint4_t StoreMod0; uint4_t UsesLoadMod0ForStore; uint4_t UnitDelayKind; } Misc;   // 12 bits
  uint32_t Sequence[4];            // one 32-bit word per macro: 8 bits per sub-unit
  uint32_t InstructionTemplate[4]; // four arbitrary SFPU instructions, shared by all macros
} LoadMacroConfig;
```

So there are *4 macros* (selected by a 2-bit index in the `SFPLOADMACRO` instruction) and *4 instruction templates* shared between them, plus `Misc`, also shared.

== The instruction and its sequence bytes

```c
TT_SFPLOADMACRO( (MacroIndex << 2) | VDLo,   // MacroIndex 0..3, VDLo 0..3: the row register
                 Mod0,                       // as SFPLOAD
                 AddrMod,                    // as SFPLOAD
                 Imm10 )                     // Dst address; bit 0 (VDHi) selects VD = VDLo or VDLo + 4
```

The load part is a normal `SFPLOAD` into `VD = (Addr & 1) << 2 | VDLo`, so a macro's row register is `LReg0..3`, or `LReg4..7` when the address is odd. (Keep the address even and the row register is `VDLo`.)

The rest is driven by the macro's `Sequence` word, one byte per sub-unit (Simple in bits 7:0, MAD 15:8, Round 23:16, Store 31:24):

#align(center)[
#table(columns: 8, inset: 5pt, stroke: 0.5pt, align: center,
  [*bit 7*], [*bit 6*], [*bit 5*], [*bit 4*], [*bit 3*], [*bit 2*], [*bit 1*], [*bit 0*],
  table.cell(fill: rgb("#fde68a"))[row reg \ is `VB`], table.cell(fill: rgb("#fce7f3"))[`VD = 16`], table.cell(fill: rgb("#dbeafe"), colspan: 3)[delay (0 to 7)], table.cell(fill: rgb("#dcfce7"), colspan: 3)[which instruction],
)
]
#text(size: 8pt)[_Figure 6. One sub-unit byte of a macro sequence. "Which instruction": 0 nothing, 2 `SFPNOP`, 3 `SFPSTORE`, 4..7 = instruction template 0..3. (1 is undefined behaviour.)_]

- *delay.* A 3-bit field: a scheduled instruction executes `delay + 1` cycles after the `SFPLOADMACRO` (delay 0 = the very next cycle). *The maximum is 7, so a scheduled instruction executes at most 8 cycles after the macro.*
- *bit 7 (`VB`).* The macro substitutes its row register into the template instruction. With bit 7 clear it is put into `VC` (and `VB` is set to the template's `VD` if the instruction has no `VB`); with bit 7 set it is put into `VB`. In both cases the template's `VD` is replaced by the row register (or by `LReg16` with bit 6). This decides which operand of the template instruction is "the data" and which stay as the template wrote them (constants).
- *bit 6 (`VD = 16`).* The result goes to the bonus register `LReg16` instead of the row register (and a scheduled `SFPSTORE` reads `LReg16`). `LReg16` is a way to carry a value from one scheduled instruction to a scheduled store without using a normal register.
- *The store.* A scheduled `SFPSTORE` stores the macro's row register (or `LReg16`) at the address computed *at the time of the `SFPLOADMACRO`* (`Imm10` plus offsets), not when it executes. Its `Mod0` is either the macro's `Mod0` (if `UsesLoadMod0ForStore` bit set) or `Misc.StoreMod0`.

*`UnitDelayKind`* (one bit per sub-unit in `Misc`) chooses what the delay counts: `WaitForElapsedCycles` (clock cycles) or `WaitForElapsedInstructions` (instructions issued to the unit since). In instruction mode, scheduled instructions only advance when something is issued, which makes a schedule robust against gaps in the instruction stream, but means the unit does not drain by itself: you have to issue `SFPNOP`s after the last macro (section 3.7).

#note[*A scheduled instruction can overwrite or be overwritten by a normal one.* If a scheduled instruction arrives at a sub-unit in the same cycle a normal instruction is issued to that same sub-unit, *the scheduled one wins and the normal one is silently discarded*. Plan which sub-unit is used in each cycle, as in Table 4. #isa]

== Configuring a macro

You write the three pieces of state with a few instructions in the kernel's init function. The idiom (from tt-metal's own kernels, for example `signbit_init`): #tm

```cpp
// 1) Instruction templates: issue the instruction with VD = 12 + i; it is captured, not executed.
TTI_SFPIADD(0, p_sfpu::LREG5, 12, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);   // template 0
TTI_SFPMAD (p_sfpu::LREG7, p_sfpu::LREG0, p_sfpu::LREG4, 13, 0);                                   // template 1

// 2) The sequence word of macro 0: write it to LReg0 as two 16-bit halves, then SFPCONFIG it in.
constexpr uint simple_bits = 0x80 | (3 << 3) | 4;   // template 0, delay 3, row register is VB
constexpr uint mad_bits    = 0x80 | (0 << 3) | 5;   // template 1, delay 0, row register is VB
constexpr uint round_bits  = 0;
constexpr uint store_bits  = (5 << 3) | 3;          // store, delay 5
TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
TTI_SFPCONFIG(0, 4 + 0, 0);      // VD 4..7 = Sequence[0..3]  (this is "macro 0")
TTI_SFPNOP;                      // required: the next SFPLOADMACRO could see the old or the new value

// 3) Misc: StoreMod0 = INT32, delays count issued instructions on all four sub-units.
TTI_SFPCONFIG(0xF00 | uint(InstrModLoadStore::INT32), 8, 1);   // VD 8 = Misc, mod1 1 = value is the immediate
TTI_SFPNOP;
```

Facts behind this code: #isa
- `SFPCONFIG` VD 0..3 writes `InstructionTemplate[0..3]` (from `LReg0`), 4..7 `Sequence[0..3]`, 8 `Misc`, 11..14 `LReg11..14`, 15 the lane config. Capturing a template by issuing an instruction with `VD = 12 + i` is "more efficient than using `SFPCONFIG`" and requires `LaneConfig.DISABLE_BACKDOOR_LOAD` to be false (the default).
- After an `SFPCONFIG` that changes the sequence or `Misc`, put an `SFPNOP` before the first `SFPLOADMACRO`, and between an `SFPCONFIG` that changes a template and the next instruction that is captured as another template.
- The macro index and the template index are separate: all macros share the four templates and `Misc`. If two kernels both use macros, they must not both claim template 0, and their `Misc` must agree.
- The config survives until it is overwritten, so it does not have to be re-done per call, but a different kernel that configures macros will overwrite it. Do it in the init that precedes the kernel body.

== Example 1: the simplest macros (tt-metal's own)

Two small kernels from tt-metal show the entire range of benefit with almost nothing to schedule. #tm

*`calculate_signbit`* (`ckernel_sfpu_signbit.h`): `out = (x < 0) ? 1.0 : 0.0` as a float, one *row per cycle*:

#align(center)[
#table(columns: (auto, auto, auto, auto, auto), inset: 4pt, stroke: 0.4pt + luma(170), align: center,
  [*cycle*], [*Load*], [*Round*], [*Simple*], [*Store*],
  [0], [`[a]` macro issued], [], [], [],
  [1], [], [`[a] >>= 31`], [], [],
  [2], [], [], [`[a] L16 = cast_fp32(a)`], [],
  [3], [], [], [], [`[a] L16` stored],
)
]
#text(size: 8pt)[_Table 2. `[x]` = scheduled by the macro with row register x. The kernel issues one `SFPLOADMACRO` per row and nothing else (`LREG0`/`LREG1` alternate); the shift is on the Round sub-unit (`SFPSHFT2`), the cast on Simple, and the cast result goes through `LReg16` to the store._]

The unmacroed version needs four instructions per row (`SFPLOAD`, `SFPSHFT`, `SFPCAST`, `SFPSTORE`); the macro version needs one, a 4x higher instruction rate, because three different sub-units each do one job. The Round sub-unit hosts the shift, which is why that file uses `SFPSHFT2` there, not `SFPSHFT` (which belongs to Simple).

*`calculate_unary_max_min`* (`ckernel_sfpu_unary_max_min.h`): `y = max(x, c)` takes 2 cycles per row, because `SFPSWAP` occupies the Simple sub-unit and the next cycle must be idle (rule ‡ below):

#align(center)[
#table(columns: (auto, auto, auto, auto), inset: 4pt, stroke: 0.4pt + luma(170), align: center,
  [*cycle*], [*Load*], [*Simple*], [*Store*],
  [0], [`[a]` macro], [], [],
  [1], [`SFPNOP` (normal)], [`swap_minmax([a], v)`], [],
  [0'], [...], [], [],
  [1'], [], [], [`[a]` stored],
)
]
#text(size: 8pt)[_Table 3. A scheduled `SFPSWAP` forces an idle slot after it, hence 2 cycles per row instead of 1._]

== Example 2: a real two-sub-unit schedule, quantize pass 2 #ms

Our quantized layer kernel converts fp32 values to sign-magnitude int8: `n = round_half_even(x * 2^k)` via the "magic number" trick, then a conversion to sign-magnitude, then store. Per row: `SFPLOAD`, `SFPMAD` (`x*2^k + 1.5*2^23`, the add rounds half to even in the fp32 adder), `SFPIADD` (subtract the bits of the magic number to get the integer), `SFPCAST` (to sign-magnitude), `SFPSTORE`: 5 instructions.

The macro schedules the `SFPMAD` (MAD column), the `SFPIADD` (Simple column) and the `SFPSTORE` (Store column). Only the `SFPCAST`, which is also a Simple-column instruction and the macro can hold only one of those, is issued normally in the slot between two macros. Rows go out every 2 slots; row `r` uses `LReg(r mod 4)`; constants live in `LReg4` (the magic number), `LReg5` (its negative) and `LReg7` (the scale):

#align(center)[
#table(columns: (auto, 1.2fr, 1fr, 1fr, 1fr), inset: 4pt, stroke: 0.4pt + luma(170), align: (center, left, left, left, left),
  [*cycle*], [*issued by the RISC-V*], [*MAD*], [*Simple*], [*Store*],
  [0], [`LOADMACRO row 0`], [], [], [],
  [1], [`SFPNOP`], [`[0] a0 = a0*s + m`], [], [],
  [2], [`LOADMACRO row 1`], [], [], [],
  [3], [`SFPNOP`], [`[1] a1 = ...`], [], [],
  [4], [`LOADMACRO row 2`], [], [`[0] a0 += -m`], [],
  [5], [`SFPCAST a0` (normal)], [`[2] a2 = ...`], [(`SFPCAST` of row 0)], [],
  [6], [`LOADMACRO row 3`], [], [`[1] a1 += -m`], [`[0]` a0 stored],
  [7], [`SFPCAST a1`], [`[3]`], [(cast of row 1)], [],
  [8], [`LOADMACRO row 4`], [], [`[2] a2 += -m`], [`[1]` stored],
  [...], [...], [...], [...], [...],
)
]
#text(size: 8pt)[_Table 4. Steady state of the quantize macro: one macro and one cast per row, two cycles per row. The delays are 0 (MAD), 3 (IADD) and 5 (store); the cast is issued in slot `2r + 5`._]

Design points, all of which are general: #ms
- *Why delay 3 for the `SFPIADD`.* The `SFPMAD` runs at cycle `t + 1` and has latency 2, so its result exists at `t + 3`; the `SFPIADD` could run then, but the Simple sub-unit is already issuing the normal `SFPCAST` of the previous row on odd cycles, so the `SFPIADD` is placed on an even cycle, `t + 4`. The cast then follows at `t + 5` and the store at `t + 6` (delay 5). Choose delays so that no sub-unit receives two instructions in one cycle.
- *Why four row registers.* Row `r` occupies its register from cycle `2r` to `2r + 6`. A new row starts every 2 cycles, so up to 4 rows are in flight; `LReg(r mod 4)` is not reused until its store has read it.
- *Where the data is.* `bit 7 (VB)` is set for both templates: the template `SFPMAD(VA = scale, VB = <row>, VC = magic)` computes `scale * row + magic`, and the template `SFPIADD(VC = -magic)` computes `row = (-magic) + row`. The constants are in the template, the row comes from the macro.
- *The store address* is that of the macro's load (the same unit, in place), computed when the macro is issued.

Result on hardware, pass 2 of the quantization of a 128 x 256 fp32 tensor: 10,953 #sym.arrow.r 7,085 cycles (#sym.minus 35%), and a whole layer call 46.1 #sym.arrow.r 39.5 us, with bit-identical output. #ms

The full body, with the unrolling that makes `TT_SFPLOADMACRO` fold to immediates:

```cpp
inline void quant_sm8_lm_body() {
    TTI_SFPNOP; TTI_SFPNOP; TTI_SFPNOP;                 // after a possible FPU write to Dst (section 3.7)
    TT_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);  TTI_SFPNOP;   // rows 0, 1
    TT_SFPLOADMACRO((0 << 2) | 1, InstrModLoadStore::FP32, ADDR_MOD_7, 2);  TTI_SFPNOP;
#pragma GCC unroll 30
    for (int d = 2; d < 32; d++) {
        TT_SFPLOADMACRO((0 << 2) | (d & 3), InstrModLoadStore::FP32, ADDR_MOD_7, 2 * d);
        TT_SFPCAST(LREG0 + ((d - 2) & 3), LREG0 + ((d - 2) & 3), SFPCAST_MOD1_INT32_TO_SM32);  // row d-2
    }
    TTI_SFPNOP; CAST(row 30); TTI_SFPNOP; CAST(row 31);       // drain: casts of rows 30 and 31
    TTI_SFPNOP; TTI_SFPNOP; TTI_SFPNOP;                       // the scheduled store of row 31
}
```

== The rules and hazards of macros

+ *One instruction per sub-unit per macro*, and at most one `Simple`, one `MAD`, one `Round`, one `Store`. Two Simple-unit instructions per row cannot both be scheduled; one of them has to be issued normally. #isa
+ *Delay 0 to 7.* Executes `delay + 1` cycles after the macro. A chain longer than 8 cycles cannot be scheduled by one macro. #isa
+ *Same-cycle collisions are silent.* A scheduled instruction and a normal instruction to the same sub-unit in the same cycle: the scheduled one runs, the normal one is dropped. #isa
+ *Rule (‡) for `SFPSWAP`.* If `SFPSWAP` is scheduled on the Simple sub-unit, the MAD sub-unit must run nothing but `SFPNOP` that cycle, and the Simple and Round units must be idle (or `SFPNOP`) the next cycle. The scheduled swap reads `VC`/`VD` again on that next cycle. More generally, regular `SFPSWAP` also occupies the Simple unit about 2 cycles (#ms: about 3.1 cycles per swap in a tight kernel). #isa
+ *`Simple` and `Round` in one cycle* need one of them to use `VD = 16` and the other not. #isa
+ *No automatic stalling for scheduled instructions.* An `SFPMAD` scheduled by a macro (or consumed by one) is not protected by the automatic stall: leave at least one cycle between the `SFPMAD` and anything that reads its output (a scheduled `SFPMAD` that writes `LReg16`, or any register, is read correctly only if the consumer runs 2 cycles later). #isa
+ *Hardware bug.* If the MAD byte of a sequence has bit 6 set (`VD = 16`) but schedules `SFPNOP` (or an instruction the MAD unit cannot execute, which turns into `SFPNOP`), and that `SFPNOP` arrives on the cycle right after a different MAD instruction, every later `SFPSWAP` writes its `VD` result to `LReg16` until the MAD unit next runs something else. Never set bit 6 in a MAD byte that schedules `SFPNOP`. #isa
+ *Templates and `Misc` are global.* All macros share the four templates and `Misc`; changing them changes every macro. #isa
+ *Lane predication applies to scheduled instructions too.* Scheduled instructions execute under whatever lane flags are current when they fire (section 1.5). #isa

== Synchronization, and where the `SFPNOP`s come from

There are several different reasons for an `SFPNOP` or an extra instruction, and it helps to tell them apart: #isa

#tbl((auto, 1fr),
  [*1. After an FPU or unpack write to `Dst`*], ["At least three unrelated Tensix instructions" must execute between a Matrix Unit (FPU) instruction that writes `Dst` and an `SFPLOAD` or `SFPLOADMACRO` that reads the same region (this is stated on the `SFPLOAD` and `SFPLOADMACRO` pages; it holds for a plain `SFPLOAD` too). The ISA gives two ways: three unrelated instructions (recommended), or `STALLWAIT(B8, C4)`. tt-metal's wrapper (`_llk_math_eltwise_sfpu_start_`) already issues that `STALLWAIT`. Our kernels additionally put three `SFPNOP`s at the start of a macro kernel, after the stall is released (a Tensix `NOP` would not wait behind a `STALLWAIT(B8, ...)`; an `SFPNOP` is an SFPU instruction and does). We keep them as a precaution and did not measure whether they are necessary.],
  [*2. Draining the macro schedule*], [With `UnitDelayKind = WaitForElapsedInstructions`, scheduled instructions only count down when instructions are issued. After the last `SFPLOADMACRO`, issue enough `SFPNOP` that the last scheduled instruction (typically the store, delay up to 7 + 1) has executed, or the unit is not idle when the kernel returns. In `WaitForElapsedCycles` mode the countdown runs by itself.],
  [*3. After `SFPCONFIG`*], [`SFPNOP` after an `SFPCONFIG` that changes the sequence or `Misc` (the next `SFPLOADMACRO` may see old or new), after one that changes an instruction template and the next instruction would be captured as another template, and after one that changes `DISABLE_BACKDOOR_LOAD`.],
  [*4. `SFPMAD` followed by a consumer the stall logic misses*], [`SFPIADD`, `SFPSHFT`, `SFPAND/OR` with `USE_VB`, `SFPSWAP`, `SFPCONFIG` reading `LReg0`, partial `SFPLOADI` and others (section 1.7). Needs one cycle gap: an `SFPNOP` or, better, an unrelated instruction of another row (software pipelining).],
  [*5. The macro's store and other `Dst` clients*], [A scheduled `SFPSTORE` writes `Dst` about 3 cycles after its macro plus its delay and, unlike normal accesses, the SFPU holds its `Dst` half for a fixed 7 cycles after each load/store/macro. A Matrix Unit instruction or an unpack that touches the same half and row group should come at least `4 + delay` cycles after the macro (or after the draining `SFPNOP`s plus two more instructions); otherwise one of the writes can be lost (the SFPU's write wins). Handing the half to another thread means draining first, as for any `Dst` access, then `STALLWAIT` + `SEMPOST`.],
  [*6. SFPSTORE reading a register an `SFPMAD` just wrote*], [Held back by the automatic stall by one cycle; a Matrix Unit instruction pushed immediately behind such a held-back store can reach `Dst` first. One instruction of any kind between them is enough.],
)

== Example 3: when a macro does not work

Macros are not a universal speed-up. Three cases from our own kernels, with the rule that explains each.

=== The `SFPSWAP` max loop: wrong results #ms

Absmax of a tile with `SFPSWAP`: per row, `|x|` then a swap against a running maximum. The natural macro form is: macro loads the row and schedules `SFPABS` (Simple) for the cycle after next; the normal stream issues the `SFPSWAP`. The first version produced wrong results: *every negative value was missed* (`|x|` never took effect for rows after the first two). The reason is the rule above: `SFPSWAP` occupies the Simple sub-unit for two cycles, and a scheduled `SFPABS` arriving on the cycle after a swap is lost. A sweep of the delay (0 to 5) and the issue lag (1 to 3) found no combination that worked for all rows: the number of failing test cases (about 1,030 in the test) dropped from roughly 1,000 to 486 but never to zero.

Moreover the swaps are the cost, not the issue slots: a pipelined version with 3 swaps per pair of rows and loads in the gaps, 6 slots per pair, ran at 148 cycles per tile, the same as before, about 3.1 cycles per `SFPSWAP`. #ms *Lesson: if the work of a row is dominated by one instruction that occupies its sub-unit for more than one cycle, scheduling around it cannot make that sub-unit faster.*

=== A long integer chain on the Simple unit: the requantize sweep #ms

A requantize sweep (shift by a per-tensor amount with round half to even, sign handling, ReLU) needs 8 to 10 Simple-unit instructions per row: shifts, `SFPAND`, two `SFPIADD`s, `SFPSETCC`/`SFPENCC`. It ran at 9.8 cycles per row for 11 instructions (ReLU) and 11.5 for 13 (no activation), i.e. about 1 cycle per instruction: *issue-bound*. A macro cannot help much, for three independent reasons:
+ *The Simple unit is already the floor.* It does one instruction per cycle. With 8 (ReLU) or 10 Simple instructions per row the floor is 8 or 10 cycles per row; the current kernel is at 9.8 and 11.5. A macro can only take away the load, the store and one Simple instruction.
+ *The chain is longer than a macro can schedule.* The delay field maxes at 7, so a macro can execute things at most 8 cycles after the load; the chain is 9 to 11 instructions long, so the scheduled store would be due before the data is final.
+ *The lane flags are global.* The ReLU and the "no negative zero" rule use `SFPSETCC`/`SFPENCC`, and with several rows in flight the flags set for one row disable lanes of another row's scheduled instructions. An arithmetic version (mask by arithmetic shift, `XOR`, add) has the same instruction count.

Upper bound of what a perfect schedule could win: about 1.5 cycles per row, 1.5% of the whole layer call. Not worth the risk.

=== Other cases where the macro does not pay

- *The kernel is bound by something else.* If the SFPU part is a small share of the time (for example the SFPU is 30% and the rest is the unpack and pack of the tile), a 2.5x faster SFPU part gives a small end-to-end gain. In our layer, the quantize pass 2 got 35% faster but the whole call only 14% faster, because `copy_tile` and the pack are separate costs on the same thread. #ms
- *Few rows.* The macro needs a configuration (about a dozen instructions and two `SFPNOP`s) and the pipeline needs drain `SFPNOP`s; for kernels that touch a handful of rows the set-up outweighs the gain.
- *Data-dependent control flow* (per-lane `v_if` with several paths, flag stack): flags are global, see above.
- *Cross-lane instructions* (`SFPTRANSP`, `SFPSHFT2` lane shifts, `SFPSWAP` sub-vector modes): they involve several registers and cycles and are not offered as scheduled instructions in the straightforward columns (`SFPTRANSP` can be scheduled on Simple; `SFPSHFT2` on Round) but their cross-register semantics are hard to combine with the "row register" substitution.
- *More distinct instructions per sub-unit than the macro allows.* Four templates shared by four macros bound how many different instructions can be scheduled at all.

== A design procedure

+ *Write the plain version first* (all `TTI_`, no macro) and test it against a reference. This is your oracle. Count instructions per row, and per sub-unit.
+ *Identify the bottleneck sub-unit.* For each of Load, Simple, MAD, Round, Store, count the instructions per row. The macro version cannot beat the largest count, whatever else you do. If the Simple count equals the total minus load and store, the gain is bounded.
+ *Decide what to schedule.* One instruction per column. Put into the macro the instructions that sit at the start and the end of the dependency chain around the load and store, and leave the others normal.
+ *Lay out the time table.* One row per cycle, one column per sub-unit and one for the issue stream. Mark where each instruction of each row falls (`delay + 1` after the macro for scheduled ones; the issue slot for normal ones). Check: no sub-unit twice in a cycle; every consumer at least latency cycles after its producer (and one more where there is no automatic stall); the delays are at most 7.
+ *Allocate registers.* Each row holds a register from its load to the last read of it (including the scheduled store). The number of rows in flight is `ceil(lifetime / period)`; use that many registers.
+ *Add the prologue and the drain.* The first rows have nothing scheduled behind them to fill the normal slots; the last rows need `SFPNOP`s until the last scheduled instruction has run.
+ *Test with data that makes holes visible*: negative values, ties, extremes, zero, values in each unit of a tile (spikes at each position). A schedule that is wrong for some rows usually passes random data.
+ *Measure the whole thing*, not just the SFPU part: the profiler zone around the loop and the end-to-end call.

== Quick reference

#tbl((auto, 1fr),
  [*Slot*], [The cycle in which the SFPU accepts one instruction from the stream. Cost of an un-macroed kernel = number of instructions issued.],
  [*Row / unit*], [32 values: 4 rows x 8 columns of `Dst`; `SFPLOAD` offset `2 d` for unit `d` of a tile (fp32).],
  [*`TTI_` / `TT_`*], [Immediate vs runtime instruction word; use `TTI_` or unrolled `TT_` with constants.],
  [*Macro delay*], [0 to 7; executes `delay + 1` cycles after the `SFPLOADMACRO`.],
  [*Sequence byte*], [`[VB:1][VD16:1][delay:3][select:3]`; select 3 = store, 4..7 = template 0..3. Simple/MAD/Round/Store = bytes 0..3.],
  [*Template capture*], [Issue the instruction with `VD = 12 + i`. Only `LaneConfig.DISABLE_BACKDOOR_LOAD = false`.],
  [*Config*], [`SFPCONFIG(imm, VD, mod1)`: VD 0..3 template (from `LReg0`), 4..7 sequence, 8 Misc, 11..14 `LReg11..14`, 15 lane config; mod1 bit 0 = value is the immediate.],
  [*Always `SFPNOP` after*], [`SFPCONFIG` that changes sequence/`Misc`/template; the last macro of a kernel (instruction-count delays); an FPU/unpack write to `Dst` before a load or macro (3 instructions).],
  [*Never*], [Schedule two instructions on one sub-unit in the same cycle; read an `SFPMAD` result the next cycle from a macro; set MAD byte bit 6 with `SFPNOP`; let `SFPSWAP` be followed by a scheduled Simple/Round instruction on the next cycle.],
)

= Pointers

- The ISA repository `tt-isa-documentation`, Blackhole: `BlackholeA0/TensixTile/TensixCoprocessor/` has `VectorUnit.md` (instruction table, sub-units, predication), `LReg.md` (shared with Wormhole), `SFPLOADMACRO.md`, `SFPSWAP.md`, `SFPSHFT.md`, `SFPCAST_*.md`, `SFPSTOCHRND_*.md`, `SFPMAD.md` (the automatic stalling and its holes), `Dst.md` (layout, instruction scheduling), `STALLWAIT.md`, `REPLAY.md`, and `BlackholeA0/TensixTile/BabyRISCV/PushTensixInstruction.md`.
- Diagrams in the repository, `Diagrams/Out/`: `TensixFrontend_BH.svg`, `TensixBackend.svg`, `TensixBackendMisc.svg` (the block diagrams behind Figure 1), `CrossLane_SFPLOAD.svg` / `CrossLane_SFPSTORE.svg` / `CrossLane_SFPTRANSP.svg` / `CrossLane_SHFLROR1.svg` / `CrossLane_SHFLSHR1_BH.svg` (the lane movement patterns of the cross-lane instructions), `Bits32_Dst32_FP32.svg` / `Bits32_Dst32_INT32.svg` / `Bits32_Dst16_*.svg` (the bit layouts of `Dst` data), `Bits32_SFPLOADMACRO_BH.svg` and the other `Bits32_SFP*` (instruction encodings), `UnpackerPipeline.svg` and `PackerPipeline.svg`.
- tt-metal: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/` (kernels such as `ckernel_sfpu_signbit.h`, `ckernel_sfpu_unary_max_min.h`, `ckernel_sfpu_quant.h`, `ckernel_sfpu_recip.h` use `SFPLOADMACRO`), `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h` (the `TT_`/`TTI_` macros), `runtime/sfpi/include/sfpi_constants.h` (the `MOD1` constants).
- Our worked examples: the quantization kernels of the `tt-zkml` repository (`src/device/kernels/compute/quant_sfpu.hpp`, `pot_scale_sfpu.hpp`, `src/device/operations/layer_execution/device/kernels/compute/layer_sfpu.hpp`) and the benchmark `benchmarks/execute_layer_2s_vs_sm_acc/` with its README (including the failed macro attempts).
