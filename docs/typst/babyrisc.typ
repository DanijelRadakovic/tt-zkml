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
#let arrr = text(size: 12pt)[#sym.arrow.r]
#let arrd = text(size: 12pt)[#sym.arrow.b]
#let isa = text(fill: rgb("#1f5f9f"), size: 7.5pt, weight: "bold")[\[ISA\]]
#let tm = text(fill: rgb("#7a4b00"), size: 7.5pt, weight: "bold")[\[tt-metal\]]
#let ms = text(fill: rgb("#1c6b2a"), size: 7.5pt, weight: "bold")[\[observed\]]
#let tbl(cols, ..args) = table(columns: cols, inset: 4pt, stroke: 0.4pt + luma(170), ..args)
#let yes = table.cell(fill: rgb("#dcfce7"))[yes]
#let no = table.cell(fill: rgb("#fee2e2"))[no]

#align(center)[
  #text(size: 18pt, weight: "bold")[The baby RISC-V cores and the Tensix coprocessor] \
  #v(2pt)
  #text(size: 11pt)[Blackhole: what the processors are, what the coprocessor is for, and how one drives the other]
]
#v(6pt)

This document explains the relationship between the five small RISC-V cores of a Tensix tile (the "baby" cores) and the Tensix coprocessor that does the numeric work: why there is a coprocessor at all, which core may touch which part of it, how instructions get from a core into it (including what the MOP and Replay expanders are), and how the two sides synchronize. It is general reference material, not tied to a particular kernel. Statements are tagged by source:

#tbl((auto, 1fr),
  [#isa], [The Tenstorrent ISA documentation (`github.com/tenstorrent/tt-isa-documentation`, Blackhole A0 pages under `BlackholeA0/TensixTile/`, and the Wormhole pages they share). The authority on hardware behaviour.],
  [#tm], [tt-metal / tt-llk conventions: how the software stack names and uses the cores.],
  [#ms], [Seen directly in this project's builds and logs (for example the JIT compile commands of a compute kernel); not an architecture statement.],
)

#outline(indent: 1.2em, depth: 2)
#pagebreak()

= Processors and coprocessor: the division of labour

== What is in a Tensix tile

A Tensix tile contains, per the ISA overview: #isa
- 1536 KiB of RAM called *L1*;
- *five baby RISC-V cores*;
- two NoC connections and a NoC overlay (a small coprocessor that assists NoC transactions);
- one *Tensix coprocessor*, itself made of 2 unpackers, 1 matrix unit (FPU), 1 vector unit (SFPU), 1 scalar unit (ThCon), 4 packers and some control units;
- small utility devices mostly for the RISC-V cores: mailboxes, TDMA-RISC, a debug timestamper, the PIC.

#good[*The intended division of labour* (the ISA's own words): "It is expected that the programmer will provide code for the RISCV cores to execute. High performance will be achieved by having that code oversee the operation of the NoC and the Tensix coprocessor. One very natural division of labour is to have two RISCV cores oversee the NoC and three RISCV cores oversee the Tensix coprocessor. The RISCV cores are not intended to achieve high performance on their own; they are intended to oversee the other components that _actually_ drive performance." #isa]

That sentence is the whole design. The baby cores run *control*: loops, addresses, conditions, when to start and wait. The coprocessor runs the *data path*: moving tiles from L1 into registers, multiplying, doing vector math, converting formats, and writing results back to L1, at rates no scalar core could reach.

== Why a coprocessor and not just the RISC-V cores

A baby core is "a relatively small 32-bit in-order single-issue core, optimized for area and power efficiency rather than for high performance", executing at most one instruction per cycle at 1.35 GHz. Its arithmetic is scalar: one integer or one single-precision float operation per instruction. #isa The coprocessor, in contrast, has units each of which does a whole tile-sized or 32-lane operation per instruction:

#tbl((auto, 1fr),
  [*Unpackers (2)*], [Move data from L1 into the coprocessor's registers (`SrcA`, `SrcB`, or directly `Dst`), converting the storage format (bf16, fp16, fp32, block-float, int8, ...) on the way.],
  [*Matrix Unit (FPU)*], [Low-precision matrix multiply-accumulate and some other matrix operations. This is where almost all of the FLOP/s are.],
  [*Vector Unit (SFPU)*], [32-lane SIMD on 32-bit lanes: fp32 multiply-add, integer and bit operations, conversions, comparisons. (See the SFPU document.)],
  [*Scalar Unit (ThCon)*], [Integer scalar operations and 128-bit memory operations against L1, including atomics; it also owns the coprocessor's general-purpose registers.],
  [*Packers (4)*], [Move results from `Dst` back to L1, converting the format on the way.],
  [*Sync, Configuration, Mover, Miscellaneous*], [Semaphores and stall conditions; backend configuration registers; a block mover; small utilities.],
)

The coprocessor "is simultaneously executing instructions from three separate threads, at a rate of up to one instruction per thread per cycle". It has no program counter and no instruction fetch: it is a slave that executes the 32-bit instructions the baby cores *push* into it. #isa

== The five cores

The ISA names the cores RISCV B, T0, T1, T2 and NC. tt-metal calls them BRISC, TRISC0, TRISC1, TRISC2 and NCRISC. #tm

#tbl((auto, auto, 1fr),
  [*ISA name*], [*tt-metal name*], [*Role in the usual software pattern*],
  [RISCV B], [BRISC], [A data-movement ("dataflow") core: runs reader/writer kernels that drive the NoC, and does set-up work. Unlike NC it also has a restricted path into the coprocessor (below).],
  [RISCV NC], [NCRISC], [The second data-movement core. It has *no* access to the coprocessor.],
  [RISCV T0], [TRISC0], [Drives Tensix thread T0, by convention the *unpack* thread.],
  [RISCV T1], [TRISC1], [Drives Tensix thread T1, by convention the *math* thread (FPU and SFPU instructions).],
  [RISCV T2], [TRISC2], [Drives Tensix thread T2, by convention the *pack* thread. Also has a subset of the RISC-V vector extension.],
)

The assignment of T0 to unpack, T1 to math and T2 to pack is a software convention: "any thread can access any backend execution unit, but it has been found to be a useful software pattern". #isa

= The baby RISC-V core itself

== Instruction set

All cores implement RV32IM with Zicsr, Zaamo (atomics, L1 only; no load-reserved/store-conditional), Zba and Zbb (address generation and basic bit manipulation), `pack`, `brev8` and `grevi`, plus parts of `Zicntr`, `F` and `Zfh`. Notable restrictions: #isa
- *Floating point is partial and not IEEE-exact.* Rounding is always round-to-nearest-even, denormals are flushed to zero, `fdiv.s` and `fsqrt.s` do not exist, and the fused multiply-add forms are neither separate multiply/add nor a true fused multiply-add. The `Zfh` instructions can be switched to bf16 by a CSR bit.
- *RISCV T2 only:* part of the vector extension (16-byte vector registers, 32-bit maximum element).
- *RISCV B and NC only:* `mret`, for interrupt handlers.
- *The one bespoke extension is `.ttinsn`*, "for efficiently pushing static instructions to the Tensix coprocessor" (section 4.1).

== Pipeline, caches and local memory

#align(center)[
#block(width: 100%, stroke: 0.6pt + luma(120), radius: 5pt, inset: 8pt, fill: luma(252))[
  #grid(columns: (1fr, 0.1fr, 1fr, 0.1fr, 1fr, 0.1fr, 1.1fr), column-gutter: 2pt, align: center + horizon,
    blk(w: 100%)[Frontend \ fetch from L1 via the \ 4 x 16 B L0 icache, \ predicts all branches],
    arrr,
    blk(w: 100%, fill: rgb("#dbeafe"))[EX1 \ in-order; ALU, branches \ (mispredict = 4-cycle bubble), \ divide 2 to 33 cycles],
    arrr,
    blk(w: 100%)[EX2 (multiply, FP) \ or \ Load/Store Unit \ (4-entry store queue)],
    arrr,
    blk(w: 100%, fill: rgb("#fde68a"))[Retire Unit \ 8-entry queue; commits \ in program order]
  )
]
]
#text(size: 8pt)[_Figure 1. The baby core pipeline (ISA `BabyRISCV/README.md`; the exact diagram is `Diagrams/Out/BabyRISCV_BH.svg`)._]

- *Instructions come from L1 only.* A small L0 instruction cache sits between each core and L1; in T0, T1 and T2 it can fuse up to four adjacent `.ttinsn` into one cycle. #isa
- *The core is in order up to EX1 and reorders after it.* Loads can overtake stores in the Load/Store Unit, and requests to different memory regions can be processed in a different order than they were issued. Programmers must use `fence` (and the TTSync mechanisms, section 5) where order matters. #isa
- *The L0 data cache is 64 bytes (4 lines of 16 B) and is not coherent.* Any `fence` or atomic flushes it, and every request has about a 0.8% chance of flushing it. Writes by other agents to L1 do not invalidate it. #isa
- *The store queue coalesces* temporally adjacent stores to the same 16-byte-aligned L1 region (stores within +-4 bytes of each other): sustained L1 store throughput is one per cycle if they coalesce, otherwise one coalesced store every 5 cycles. #isa
- *Local data RAM:* 8 KiB for B and NC, 4 KiB for T0, T1 and T2 (twice Wormhole's), reachable at two addresses: `MEM_LOCAL_BASE` (`0xFFB0_0000`), private to the core, 2-cycle loads and never contended; and a second mapping at `0xFFB1_4000`..`0xFFB1_DFFF` reachable from every core and from the NoC, with at least 8 cycles latency. The call stack and hot variables belong in the first. #isa

#tbl((auto, auto),
  [*Load from*], [*Latency (cycles)*],
  [core-local data RAM via `MEM_LOCAL_BASE`; L1 with L0 data cache hit], [2],
  [Mailboxes, PCBufs, Manual TTSync, Tensix semaphores], [>= 3 (more if the FIFO is empty or TTSync waits)],
  [Tensix GPRs, Tensix backend configuration, TDMA-RISC], [>= 4 (more with conflicts or Auto TTSync waits)],
  [tile control / debug / status, PIC, NoC and overlay registers], [>= 7],
  [L1 with L0 data cache miss; local data RAM via the slow path], [>= 8],
  [L1 atomic], [>= 12],
)
#text(size: 8pt)[_Table 1. Load latency by address range (ISA)._]

== What each core can reach

The address space is not the same for all five cores. This table is the quickest way to see who can touch the coprocessor (derived from the memory map of `BabyRISCV/README.md`): #isa

#tbl((1.5fr, 1.3fr, 1.3fr, 0.6fr, 0.9fr, 1.4fr),
  [*Region*], [*B*], [*T0 / T1 / T2*], [*NC*], [*NoC*], [*Notes*],
  [L1 (1536 KiB)], yes, yes, yes, yes, [shared by everything],
  [local data RAM], yes, yes, yes, [slow mapping only], [see above],
  [NoC 0 / NoC 1 / overlay registers], yes, yes, yes, yes, [how data movement is started],
  [tile control / debug / status; PIC], yes, yes, yes, [yes], [clocking, reset, PC snapshots],
  [*`INSTRN` push* (coprocessor instructions)], table.cell(fill: rgb("#fde68a"))[yes, any thread, after the MOP expander], table.cell(fill: rgb("#dcfce7"))[yes, own thread, before the MOP expander], no, no, [section 4],
  [MOP expander config (`MopCfg`)], no, table.cell(fill: rgb("#dcfce7"))[yes, own thread], no, no, [write-only],
  [Tensix backend configuration], yes, yes, no, no, [unpacker/packer/FPU/SFPU config],
  [Tensix GPRs], table.cell(fill: rgb("#fde68a"))[yes, all three threads], table.cell(fill: rgb("#dcfce7"))[yes, own thread], no, no, [the coprocessor's scalar registers],
  [`Dst` (the coprocessor's register file)], no, yes, no, no, [data access from T cores only],
  [Tensix semaphores], no, yes, no, no, [stall/sync; the T cores' address space only],
  [PCBuf (B to T FIFO)], [B side], [T side], no, no, [section 5.3],
  [Mailboxes], yes, yes, no, no, [core to core FIFOs],
)

So the statement "BRISC cannot reach the coprocessor" is true for NCRISC, not entirely for BRISC. B can push coprocessor instructions to any thread, read and write the coprocessor's GPRs and backend configuration. What it cannot do is use the MOP expander or sit at the front of a thread's FIFO, and its pushes are explicitly dangerous when a T core is also pushing (below). In practice tt-metal uses BRISC for data movement and for boot-time set-up. #isa #tm

#pagebreak()

= The coprocessor: threads, units, shared state

The coprocessor is split into a *frontend* (one copy per thread) and a *backend* (execution units, mostly shared by the threads). #isa

#align(center)[
#block(width: 100%, stroke: 0.6pt + luma(120), radius: 5pt, inset: 8pt, fill: luma(252))[
  #grid(columns: (1fr, 0.12fr, 1.5fr, 0.12fr, 1.2fr), column-gutter: 3pt, align: center + horizon,
    [#text(weight: "bold", size: 8.5pt)[baby cores] \
     #v(2pt)
     #blk[T0 (unpack)] #v(2pt) #blk(fill: rgb("#dbeafe"))[T1 (math)] #v(2pt) #blk[T2 (pack)] #v(2pt)
     #blk(fill: rgb("#fde68a"))[B (also can push)] #v(2pt)
     #blk(fill: rgb("#fee2e2"))[NC (cannot push)]],
    [#arrr],
    [#text(weight: "bold", size: 8.5pt)[frontend, per thread: three stages, in order] \
     #v(2pt)
     #blk(w: 100%)[1. FIFO #arrr *MOP expander* (`MOP`, `MOP_CFG`)]
     #v(2pt)
     #blk(w: 100%)[2. *Replay expander* (`REPLAY`)]
     #v(2pt)
     #blk(w: 100%, fill: rgb("#fee2e2"))[3. *Wait Gate* (`STALLWAIT`, `SEMWAIT`, busy unit)]
     #v(2pt)
     #text(size: 7pt)[T cores push before stage 1; B pushes between 1 and 2]],
    [#arrr],
    [#text(weight: "bold", size: 8.5pt)[backend] \
     #v(2pt)
     #grid(columns: (1fr, 1fr), gutter: 3pt,
       blk[Unpackers], blk[FPU],
       blk[Packers], blk[SFPU],
       blk[ThCon], blk[Sync],
       blk[Config], blk[Mover])
     #v(3pt)
     #blk(w: 100%, fill: rgb("#f3e8ff"))[shared: `Dst`, `LReg`, `SrcA/B`, GPRs per thread, config (some per thread)]]
  )
]
]
#text(size: 8pt)[_Figure 2. Cores, per-thread frontends and shared backend (ISA `Tensix Coprocessor README`, `PushTensixInstruction.md`)._]

Key properties: #isa
- *The frontend is strictly in order.* Instructions leave it in the order the core pushed them. After the Wait Gate they are in the backend, where different units run in parallel and can finish in a different order; the Wait Gate (via `STALLWAIT`/`SEMWAIT`) is how software prevents that.
- *Most backend state exists once, not per thread.* `Dst` and `LReg` are shared; two copies of `SrcA`/`SrcB` exist and flip between "unpackers write" and "FPU reads". Software must manage concurrent access. Per-thread state does exist for the GPRs, address counters (ADCs), read/write counters (RWCs) and mutex ownership, and some of the configuration is replicated.
- *Each backend unit accepts one instruction per cycle,* summed over the threads. The FPU, SFPU and each packer can be used by one thread at a time in the usual pattern.
- *Instructions are 32 bits wide.* Operands that do not fit are taken from *configuration registers*, some replicated three times (per thread), some twice. Instructions exist to change them; they are also memory-mapped into the cores' address space (`TENSIX_CFG_BASE`).

= How a core drives the coprocessor

== Pushing an instruction

The coprocessor does not fetch instructions: "the Baby RISCV cores _push_ instructions". A push is a store: the 32-bit instruction word is written (by `sw`, or by the one-instruction `.ttinsn IMM32` extension) to a special address. Which thread receives it depends on the core and the address: #isa

#tbl((auto, 1fr, 1fr, 1fr, 1fr),
  [*Store address*], [*RISCV B*], [*RISCV T0*], [*RISCV T1*], [*RISCV T2*],
  [`INSTRN_BUF_BASE` `0xFFE4_0000`], [thread T0, *after* the MOP expander], [thread T0, before the MOP expander], [thread T1, before the MOP expander], [thread T2, before the MOP expander],
  [`INSTRN1_BUF_BASE` `0xFFE5_0000`], [thread T1, after the MOP expander], table.cell(colspan: 3)[do not use (hangs the core)],
  [`INSTRN2_BUF_BASE` `0xFFE6_0000`], [thread T2, after the MOP expander], table.cell(colspan: 3)[do not use (hangs the core)],
)

- *`.ttinsn IMM32`* stores a compile-time 32-bit constant in one RISC-V instruction (the encoding is the constant rotated left by two bits, which lands in the space other designs use for compressed instructions). The instruction cache fuses up to four adjacent `.ttinsn` into one cycle, so a core can *enqueue* four coprocessor instructions per cycle even though the coprocessor *dequeues* at most one per thread per cycle. In tt-metal the `TTI_xxx` macros emit `.ttinsn`, and the `TT_xxx` macros build the word at run time and store it with `sw`. #isa #tm
- *The core runs ahead.* After the push, the core proceeds. The instruction executes later, asynchronously, as the FIFOs allow. A load from a register that a pushed instruction will write is not ordered behind it; "mechanisms exist to synchronize ... but software needs to explicitly use these mechanisms - it is not automatic" (section 5).
- *FIFOs.* The first FIFO holds 32 instructions (28 plus a 4-wide fused push; once more than 28 are in it, it must fall back to 28 before it accepts more). When it is full the pushing core stalls automatically. This bounds how far ahead a core can get. The sync unit also stalls pushes while it is tracking 29 or more instructions. #isa
- *Never push a word whose opcode is not a Tensix instruction.* In particular a word whose top eight bits are `0x00` is never accepted: that thread stops forever, the FIFO eventually fills and the core hangs, and even a soft reset cannot recover the tile (the debug registers are write-locked); only a full chip reset helps. A macro that builds `(Opcode << 24) + Params` without parentheses around `Params` produces such words easily. #isa

#warn[*Two pushers into one thread can lose instructions.* If instructions from RISCV B and from RISCV T#sub[i] (or its MOP expander) both try to pass through the mux between the expanders on the same cycle, the one from T#sub[i] is silently discarded. Software must ensure that RISCV B pushes only when T#sub[i] is not pushing. #isa]

== The frontend stages

The three stages and their purpose, which answers "what is the difference between MOP, the MOP expander and the Replay expander":

#tbl((auto, auto, 1fr),
  [*Term*], [*Kind*], [*What it is*],
  [`MOP`], [instruction], [One pushed instruction that the MOP expander replaces with a pre-programmed sequence of up to 32,639 instructions (Wormhole; more on Blackhole). Two templates: *template 0* uses `MopCfg` entries for up to four "A" instructions and one "B" instruction plus skip variants, repeated `Count1 + 1` times with a mask (low 16 bits in the instruction, high 16 from `MOP_CFG`) that chooses between the normal and the skip sequence per iteration; *template 1* is a nested loop (outer count, inner count, a start op, up to two end ops, one or two alternating loop ops, and replacement ops for the last iterations).],
  [`MOP_CFG`], [instruction], [Sets the high half of the template-0 mask.],
  [MOP expander], [hardware stage], [The first stage of each thread's frontend. It is the only place `MOP` and `MOP_CFG` are handled; everything else passes through. It is programmed through `MopCfg`, a *write-only* array of nine 32-bit words per core (T0 can only write T0's, T1 T1's, T2 T2's; RISCV B cannot). It ingests and emits one instruction per cycle, and after expanding a `MOP` there is a one-cycle transition penalty if the next instruction is not another `MOP`.],
  [`REPLAY(Index, Count, Exec, Load)`], [instruction], [With `Load` set: record the next `Count` instructions into the replay buffer (and also execute them if `Exec` is set). With `Load` clear: emit `Count` instructions from the buffer starting at `Index`.],
  [Replay expander], [hardware stage], [The second stage, after the MOP expander and before the Wait Gate. It holds a per-thread replay buffer (32 entries on Wormhole; check `REPLAY.md` for Blackhole). During a replay it ingests nothing: incoming instructions queue in the FIFO before it, while it emits one buffered instruction per cycle.],
  [Wait Gate], [hardware stage], [The last stage. Holds an instruction back if its backend unit is busy, another thread wants the same unit in the same cycle, a previous scalar-unit instruction is still executing, or a `STALLWAIT`/`SEMWAIT` is latched and its block mask catches the instruction.],
)

Relations between them: #isa
- *MOP is for loops, REPLAY for linear sequences.* "The MOP Expander is typically used for repetitive looping instruction sequences, whereas the Replay Expander is typically used for linear non-looping sequences."
- *A `MOP` can output `REPLAY` instructions*, which the Replay expander then expands; a `REPLAY` cannot contain a `MOP`, and a `REPLAY` cannot contain another `REPLAY`.
- *Both are optional,* and their purpose is the same: they let a core push at a rate below one instruction per cycle while the backend still receives one per thread per cycle. That frees the core for other work. To avoid the MOP expander's one-cycle transition penalty, at least one instruction a `MOP` emits should be a `REPLAY` that expands to at least two instructions.
- *Neither expander can beat a backend unit's own rate.* A backend unit (for example the SFPU) still accepts one instruction per cycle.
- *Change `MopCfg` only when no expansion is in progress.* Hardware samples the configuration while expanding, so a write in the middle changes the sequence. Use TTSync (section 5) to wait for the expansion to finish first.

#note[*How tt-metal uses them.* The LLK layer programs `MopCfg` once per operation (the `_llk_*_mop_config_` functions) and the compute kernel then issues a single `MOP` per tile, which the expander turns into the unpack, matmul or pack loop. For short straight-line sequences, for example an SFPU row sequence, the LLK records a replay buffer once and then replays it. The `TTI_` instructions of a hand-written SFPU kernel, by contrast, are pushed one by one, so such a kernel is limited by the core's push rate as well as the SFPU's one-per-cycle rate. #tm]

= Synchronization between the cores and the coprocessor

Because the cores run ahead and the units finish out of order, there are several layers of synchronization. They answer different questions.

== Inside the coprocessor: the Wait Gate, `STALLWAIT`, `SEMWAIT`

`STALLWAIT(BlockMask, ConditionMask)` latches a rule at the Wait Gate: instructions of the classes named by the *block mask* (B0 to B8: scalar-class, sync, packer, unpacker, mover, ThCon, FPU, configuration, SFPU) are held until all the *condition bits* are satisfied (for example "the current thread has no instruction in the FPU pipeline", C4; "in the SFPU pipeline", C11; "in unpacker 0's pipeline", C1). `SEMWAIT` is the same but waits on the coprocessor's semaphores, which is how the unpack, math and pack threads hand `Dst` halves to each other (`SEMPOST` / `SEMGET`). A new wait does not replace a latched one; instructions queue behind it. The instruction right after a `STALLWAIT` is always subject to its block mask for at least one cycle. #isa

== Between a core and its coprocessor thread: Auto TTSync and Manual TTSync

Problem: a core stores to a configuration register and then pushes an instruction that uses it (or the reverse), but the push and the store travel on different paths and can arrive in either order.

- *Auto TTSync* (turned on by `SETC16` writing `TENSIX_TRISC_SYNC_TrackTensixInstructions` and one or more `Track...` bits) tracks eight resources: the Tensix backend configuration (six pieces), the GPRs, and TDMA-RISC state, and makes the common patterns race-free by default: a load from a resource followed by a push that writes it, a store followed by a push that reads or writes it, a push followed by a store or a load. Instructions wait at the Wait Gate or the core's store/load is stalled so that events happen as in program order. A `MOP` or `REPLAY` is tracked by its resource declaration, and only the first instruction of the expansion waits. Tracking is per resource, not per register, which can cause stalls that are safe but unnecessary. #isa
- *Manual TTSync*: at `PC_BUF_BASE`, T cores can load from `CoprocessorDoneCheck` (returns after thread T#sub[i] has no instruction in flight) and `MOPExpanderDoneCheck` (returns after the MOP expander has finished). The load has "exotic" semantics: the *delay* of the load is the point, so you write the address (`sw x0, 0(t0)`), load it, then consume the loaded value with an `andi t1, t1, 0` so that later instructions really wait. A latched `STALLWAIT`/`SEMWAIT` is not an in-flight instruction, so a done-check can return while a wait is still latched. #isa

#note[*Hardware constraint of Manual TTSync.* While a load from `CoprocessorDoneCheck` or `MOPExpanderDoneCheck` is outstanding, the core must not start another load from the `PC_BUF_BASE` range or from `TENSIX_MAILBOX0_BASE`..`+0x3FFF`; doing so can return wrong values or hang the core. Put one ALU instruction between such loads. #isa]

== Between cores: PCBufs, mailboxes, semaphores

- *PCBufs.* Three FIFOs, one from RISCV B to each of T0, T1, T2, each holding up to 16 32-bit values (more writes wait in shared buffers in B's memory subsystem). T#sub[i] blocks reading its end while the FIFO is empty. B can also wait for "T#sub[i] drained the FIFO, is waiting on it, and the coprocessor thread is idle". This is the mechanism by which BRISC hands work (typically "start kernel at this address") to the compute cores. PCBufs cannot be accessed by NC, the NoC or the coprocessor. #isa
- *Mailboxes.* 16 FIFOs of 32-bit values, one from each of B, T0, T1, T2 to each of B, T0, T1, T2, including a core to itself. A mailbox holds up to four values (and a core's four outgoing mailboxes together at most four); a read of an empty mailbox blocks until written. *There are no mailboxes to or from NC*; mailboxes cannot be accessed by the NoC or the coprocessor. #isa
- *Tensix semaphores.* Mapped at `PC_BUF_BASE + 0x20`.. in the address space of T0, T1 and T2 (not B), and used by the coprocessor's `SEMPOST`/`SEMWAIT`, so a T core and the coprocessor threads can signal each other. #isa

Where NC fits: NCRISC talks to the rest only through L1 and the NoC (and tile control registers). It has no PCBuf, no mailbox and no coprocessor access, so in the usual software a compute kernel and an NCRISC kernel synchronize through data in L1, typically the circular-buffer counters. #isa #tm

= How the software stack maps onto this

#tbl((auto, 1fr),
  [*One compute kernel, three programs*], [A tt-metal compute kernel is one C++ source compiled *three times*, once for each of T0, T1 and T2, with `UCK_CHLKC_UNPACK`, `UCK_CHLKC_MATH` or `UCK_CHLKC_PACK` defined (and `COMPILE_FOR_TRISC` 0, 1, 2). Code wrapped in `UNPACK(...)`, `MATH(...)` or `PACK(...)` exists only in the matching copy; unwrapped code (such as `copy_tile` or `pack_tile`) expands to different instruction streams per thread through the LLK. We saw exactly these three compile commands for every kernel in this project's JIT build logs. #ms],
  [*The math thread issues the SFPU code*], [Our SFPU kernels are wrapped in `MATH(...)`: they run on TRISC1 and push their `TTI_SFPxxx` instructions into Tensix thread T1. This is why the SFPU's instruction rate and the cost of the RISC-V instructions around the pushes matter (see the SFPU document).],
  [*Dataflow kernels on BRISC/NCRISC*], [Reader and writer kernels (NoC reads from DRAM or other cores into L1 circular buffers, and writes back) run on the two data-movement cores and never push coprocessor instructions. The compute threads and the data-movement cores meet in L1: circular buffers with producer/consumer counters.],
  [*Where the expanders show up*], [LLK `*_mop_config` functions write `MopCfg` once per op; `TTI_MOP`-style instructions then trigger the loop. `REPLAY` is used to record and replay short fixed sequences.],
  [*Where the sync mechanisms show up*], [`tile_regs_acquire / commit / wait / release` are the semaphore handshakes between T0/T1 (unpack and math) and T2 (pack) for the two halves of `Dst`; the `STALLWAIT` that opens each SFPU kernel call waits for the FPU to finish before the SFPU reads `Dst`. #tm],
)

== A tile through the system

#align(center)[
#block(width: 100%, stroke: 0.6pt + luma(120), radius: 5pt, inset: 8pt, fill: luma(252))[
  #grid(columns: (1fr, 0.1fr, 1fr, 0.1fr, 1fr, 0.1fr, 1fr, 0.1fr, 1fr), column-gutter: 2pt, align: center + horizon,
    blk(w: 100%)[*reader* \ BRISC or NCRISC \ NoC read #arrr L1 CB],
    arrr,
    blk(w: 100%)[*unpack* \ TRISC0 pushes \ UNPACR #arrr `Dst` / `SrcA/B`],
    arrr,
    blk(w: 100%, fill: rgb("#dbeafe"))[*math* \ TRISC1 pushes \ FPU / SFPU instructions],
    arrr,
    blk(w: 100%)[*pack* \ TRISC2 pushes \ PACR #arrr L1 CB],
    arrr,
    blk(w: 100%)[*writer* \ BRISC or NCRISC \ L1 CB #arrr NoC write]
  )
  #v(4pt)
  #text(size: 8pt)[Between the stages: circular-buffer counters in L1 (cores) and semaphores (coprocessor threads).]
]
]
#text(size: 8pt)[_Figure 3. One tile's path. The five cores each run control code; the data moves only through the NoC, L1 and the coprocessor units._]

= Checklist and pitfalls

+ *A pushed instruction has not necessarily run.* The core is ahead of the coprocessor. If the core must observe a result (or change a configuration register the instruction reads), use Auto TTSync, a Manual TTSync done-check, or an explicit `STALLWAIT`/semaphore.
+ *Only the core that owns a thread should push into it,* except for the explicit B-after-MOP path, and then only when the owner is not pushing.
+ *NCRISC cannot reach the coprocessor, PCBufs or mailboxes.* Plan communication through L1.
+ *Do not rewrite `MopCfg` during an expansion.* It is write-only; there is no way to read back what you set.
+ *The L0 data cache and the store queue are not coherent with other writers.* `fence` flushes the cache and drains the store queue, but a store reaching its memory region is another matter; writes to different regions can complete in a different order than they were issued.
+ *The 32-entry first FIFO and the 29-instruction tracking limit throttle a core that pushes faster than the coprocessor executes.* This is a feature: it bounds how far ahead the core runs.
+ *A `REPLAY` with `Load` set and `Exec` clear* interacts with the sync unit's tracking of in-flight instructions: the load must be pushed with specific care (see `PushTensixInstruction.md`), and instruction fusion must be disabled when loading a replay buffer with `.ttinsn`.

= Pointers

- ISA repository, Blackhole: `BlackholeA0/TensixTile/README.md` (tile overview), `BabyRISCV/README.md` (pipeline, memory map, local RAM, load latencies), `BabyRISCV/InstructionSet.md`, `BabyRISCV/PushTensixInstruction.md`, `BabyRISCV/AutoTTSync.md`, `BabyRISCV/ManualTTSync.md`, `BabyRISCV/PCBufs.md`, `BabyRISCV/Mailboxes.md`, `BabyRISCV/MemoryOrdering.md`; the shared Wormhole pages `TensixCoprocessor/README.md`, `MOPExpander.md`, `MOP.md`, `REPLAY.md`, `WaitGate.md`, and Blackhole's `STALLWAIT.md`.
- Diagrams in `Diagrams/Out/`: `BabyRISCV_BH.svg` (the pipeline), `TensixFrontend_BH.svg`, `TensixBackend.svg`, `TensixBackendMisc.svg`.
- tt-metal: `tt_metal/hw/firmware/src/tt-1xx/` (the firmware loops of each core, `trisck.cc`), `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/` (the `_llk_*_mop_config_` and `_llk_*_init_` functions), `tt_metal/hw/ckernels/blackhole/` (the compute-kernel API).
- The SFPU document `docs/sfpu.pdf` for the vector unit in detail and `SFPLOADMACRO`.
