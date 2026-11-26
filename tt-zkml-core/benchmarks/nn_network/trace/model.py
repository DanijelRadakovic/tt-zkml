# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""The state of one Tensix core during execute_network (32 x 64, bf16) and the frames (one per change) that the renderer draws.

The *times* of the frames come from the device profiler (zones of the five RISC-V cores, see tracelib.py). The *state* of the hardware units
(SrcA/SrcB banks, DST, LRegs, circular buffers) is a model derived from the kernel code and the tt-metal / tt-llk protocol: the profiler sees the
RISC-V cores, not the coprocessor units. The model is tagged as such in the PDF.
"""
import copy
from collections import OrderedDict

RUN, WAIT, IDLE = "running", "waiting", "idle"
MT, KT, NT = 1, 2, 2  # 32 x 64 activations, 64 x 64 weights: tiles
TILE = {"bf16": 2048, "u8": 1024, "i32": 4096, "f32": 4096}


def cb(role, fmt, pages):
    return dict(role=role, fmt=fmt, pages=pages, bytes=TILE[fmt], cells=[None] * pages, rd=0, wr=0, pend=0)


ZONE_TEXT = {
    "SP_COPY": {"TRISC_0": "scale_pass: unpacker copies tiles L1 -> SrcA", "TRISC_1": "scale_pass: issues the datacopy (SrcA -> DST)", "TRISC_2": "scale_pass: waits for DST"},
    "SP_SFPU": {"TRISC_0": "(finished its part of scale_pass)", "TRISC_1": "scale_pass: SFPU absmax sweep (issue + wait)", "TRISC_2": "(waits for the scaler)"},
    "SP_SCALER": {"TRISC_0": "(finished its part)", "TRISC_1": "scale_pass: absmax -> scaler tile", "TRISC_2": "scale_pass: packs the scaler tile and pushes it"},
    "QP_COPY": {"TRISC_0": "quant_pass: unpacker copies tiles into DST", "TRISC_1": "quant_pass: issues the datacopy / waits for the unpacker", "TRISC_2": "quant_pass: reserves the output pages"},
    "QP_SFPU": {"TRISC_0": "(finished its part)", "TRISC_1": "quant_pass: SFPU sweep", "TRISC_2": "quant_pass: waits for the SFPU"},
    "QP_PACK": {"TRISC_0": "(finished its part)", "TRISC_1": "quant_pass: commits DST, releases", "TRISC_2": "quant_pass: pack_tile + push"},
    "MM_FPU": {"TRISC_0": "matmul: unpacks weight and activation tiles", "TRISC_1": "matmul: issues matmul_tiles (MVMUL)", "TRISC_2": "matmul: waits for DST"},
    "MM_SWEEP": {"TRISC_0": "(finished its part)", "TRISC_1": "matmul: SFPU absmax sweep of the accumulator", "TRISC_2": "matmul: waits"},
    "MM_PACK": {"TRISC_0": "(finished its part)", "TRISC_1": "matmul: commit, release", "TRISC_2": "matmul: packs the accumulators to c_3"},
    "RQ_SCALER": {"TRISC_0": "requantize: (waits)", "TRISC_1": "requantize: absmax -> scaler 2^k", "TRISC_2": "requantize: packs the scaler to c_19"},
    "WAIT_W": {"*": "waiting for the first weight tile"}, "WAIT_ACC": {"*": "waiting until the writer freed c_3"},
    "READ_W": {"NCRISC": "NoC reads of the weights (or waiting for space in the weight CB)"},
    "WAIT_DONE": {"BRISC": "waiting until compute finished the layer (scaler in c_19)"}, "WRITE": {"BRISC": "writing accumulator and scaler to DRAM"},
    "QUANT_X": {"*": "quantizing x"}, "QUANT_W": {"*": "quantizing the weights"}, "MATMUL": {"*": "matmul stage"}, "REQUANT": {"*": "requantize stage"},
    "QW_SCALE": {"*": "weights: pass 1 (absmax)"}, "QW_QUANT": {"*": "weights: pass 2 (quantize)"},
}


def new_state(pre_quantized):
    cbs = OrderedDict()
    cbs["c_0"] = cb("x (input, resident)", "bf16", MT * KT)
    if not pre_quantized:
        cbs["c_1"] = cb("w (weights of one layer)", "bf16", KT * NT)
    cbs["c_2"] = cb("q_w (quantized weights)", "u8", KT * NT)
    cbs["c_3"] = cb("acc (int32, raw sign-magnitude)", "i32", MT * NT)
    cbs["c_4"] = cb("act A (int8 activations)", "u8", MT * KT)
    cbs["c_5"] = cb("act B (int8 activations)", "u8", MT * KT)
    cbs["c_16"] = cb("out ring (last layer)", "u8", 8)
    cbs["c_17"] = cb("scaler_x", "f32", 2)
    if not pre_quantized:
        cbs["c_18"] = cb("scaler_w", "f32", 2)
    cbs["c_19"] = cb("scaler_out", "f32", 2)
    return dict(
        cb=cbs,
        srcA=[None, None], srcB=[None, None],
        u0="idle", u1="idle", fpu="idle", sfpu="idle", packer="idle", pack_cfg="(not configured)",
        dst=[None] * 8, dst_owner=["free", "free"],
        lregs=["-"] * 8, macro="(not programmed)",
        noc=[], arrows=[],
        risc=dict(NCRISC=("idle", "-"), BRISC=("idle", "-"), TRISC_0=("idle", "-"), TRISC_1=("idle", "-"), TRISC_2=("idle", "-")),
    )


class Script:
    """Builds the frames: every mutation method changes the state, `frame()` records a snapshot."""

    def __init__(self, pre_quantized, zones):
        self.pre = pre_quantized
        self.zones = zones
        self.cur_risc = {}
        self.bg_noc = []
        self.s = new_state(pre_quantized)
        self.frames = []
        self.acq = 0  # tile_regs_acquire counter: the DST half alternates
        self.layer = 0

    # ---- circular buffers
    def reserve(self, name, labels):
        c = self.s["cb"][name]
        for i, lab in enumerate(labels):
            c["cells"][(c["wr"] + i) % c["pages"]] = ("W", lab)
        c["pend"] = len(labels)

    def push(self, name):
        c = self.s["cb"][name]
        for i in range(c["pend"]):
            k, lab = c["cells"][(c["wr"] + i) % c["pages"]]
            c["cells"][(c["wr"] + i) % c["pages"]] = ("R", lab)
        c["wr"] = (c["wr"] + c["pend"]) % c["pages"]
        c["pend"] = 0

    def pop(self, name, n):
        c = self.s["cb"][name]
        for i in range(n):
            c["cells"][(c["rd"] + i) % c["pages"]] = None
        c["rd"] = (c["rd"] + n) % c["pages"]

    def half(self):
        return self.acq % 2

    def acquire(self):
        self.acq += 1
        h = self.half()
        self.s["dst_owner"][h] = "math"
        self.s["dst_owner"][1 - h] = self.s["dst_owner"][1 - h]
        for i in range(4):
            self.s["dst"][4 * h + i] = None

    def dst_set(self, idx, label):
        self.s["dst"][4 * self.half() + idx] = label

    def release(self):
        self.s["dst_owner"][self.half()] = "free"
        for i in range(4):
            self.s["dst"][4 * self.half() + i] = None

    def frame(self, title, bullets, t, arrows=(), stage=""):
        s = self.s
        s["risc"] = {k: self.default_risc(k, t) for k in s["risc"]}
        s["risc"].update(self.cur_risc)
        s["noc"] = list(dict.fromkeys(s["noc"] + self.bg_noc))
        self.frames.append(dict(title=title, bullets=bullets, t=t, state=copy.deepcopy(s), arrows=list(arrows), stage=stage, layer=self.layer))
        # what the units do belongs to one step: the next step starts with idle units (the data they left behind stays)
        self.cur_risc = {}
        for k in ("u0", "u1", "fpu", "sfpu", "packer"):
            s[k] = "idle"
        s["noc"] = []

    def set_risc(self, **kw):
        self.cur_risc.update(kw)

    def default_risc(self, key, t):
        z = None
        for n, a, b, d in self.zones[key]:
            if n.endswith("-KERNEL") or n.endswith("-FW"):
                continue
            if a <= t <= b and (z is None or d >= z[3]):
                z = (n, a, b, d)
        if key in ("NCRISC", "BRISC") and z is None:
            return (IDLE, "between profiler zones")
        if z is None:
            return (WAIT, "between profiler zones: the thread issues instructions or waits on a semaphore")
        txt = ZONE_TEXT.get(z[0], {}).get(key) or ZONE_TEXT.get(z[0], {}).get("*") or f"in zone {z[0]}"
        st = WAIT if z[0] in ("WAIT_W", "WAIT_ACC", "WAIT_DONE") else RUN
        return (st, txt)




def build(pre_quantized, Z, zones):
    """Z(risc, name, k) -> (start, end) of the k-th zone `name` on `risc` in the profiled run (cycles). Returns the frames of layer 0 and of the last layer."""
    S = Script(pre_quantized, zones)
    s = S.s
    q = pre_quantized
    n = 0  # step counter for titles is added by the renderer
    # =====================================================================================================================
    S.layer = 0
    t = Z("BRISC", "BRISC-KERNEL", 0)[0]
    S.set_risc(NCRISC=(RUN, "issues NoC reads: x tiles (2 x 2 KB) DRAM -> c_0"), BRISC=(WAIT, "waits for the scaler of x (c_17)"),
               TRISC_0=(RUN, "hw startup, then waits for tiles in c_0"), TRISC_1=(RUN, "hw startup; issues the first datacopy / SFPU instructions"), TRISC_2=(RUN, "hw startup; waits for DST"))
    s["noc"] = ["NCRISC reads x0, x1: DRAM -> c_0"]
    S.reserve("c_0", ["x0", "x1"])
    S.frame("The program starts", [
        "The host enqueued the program; the three kernels start on this core: NCRISC (reader), BRISC (writer) and the compute kernel on the three TRISC cores.",
        "Every circular buffer (CB) is empty. NCRISC reserves c_0 and issues the NoC reads of the two bf16 tiles of x (32 x 64) from DRAM.",
        "The TRISC cores run ahead of the data: they issue Tensix instructions into the coprocessor, which then waits (semaphores / CB counters).",
    ], t, arrows=["NOC-L1"], stage="start")

    t = Z("NCRISC", "READ_W", 0)[0]
    S.push("c_0")
    s["noc"] = []
    if not q:
        S.reserve("c_1", ["w0", "w1", "w2", "w3"])
        s["noc"] = ["NCRISC reads w0..w3: DRAM -> c_1 (4 x 2 KB)"]
        S.set_risc(NCRISC=(RUN, "x done (barrier + push 2). Now reads the 4 weight tiles of layer 0 into c_1"), TRISC_0=(RUN, "c_0 has 2 tiles: starts unpacking x"))
        what = "bf16 weights (4 x 2 KB) into c_1, where compute will quantize them"
    else:
        S.reserve("c_2", ["q0", "q1", "q2", "q3"])
        s["noc"] = ["NCRISC reads q0..q3: DRAM -> c_2 (4 x 1 KB, already int8)"]
        S.set_risc(NCRISC=(RUN, "x done (barrier + push 2). Now reads the 4 pre-quantized weight tiles (UINT8) of layer 0 straight into c_2"), TRISC_0=(RUN, "c_0 has 2 tiles: starts unpacking x"))
        what = "the 4 UINT8 tiles of the pre-quantized weights (4 x 1 KB) straight into c_2: compute will not quantize them"
    S.frame("x arrives in c_0; the reader moves on to the weights of layer 0", [
        "NCRISC pushed the 2 tiles of x (the barrier of the batch completed): c_0 is full and stays resident for the whole run.",
        f"NCRISC now reads {what}.",
        "The unpacker thread (TRISC0) wakes up as soon as c_0 has a tile.",
    ], t + 0, arrows=["NOC-L1", "L1-U0"], stage="read")

    # ---------------------------------------------------------------- quantize x
    S.layer = 0
    t = Z("TRISC_0", "SP_COPY", 0)[1]
    S.acquire()
    s["srcA"] = [("x0", "reading"), ("x1", "writing")]
    s["u0"] = "copy_tile x0, x1: L1 bf16 -> SrcA (2 KB each)"
    s["fpu"] = "datacopy A2D: SrcA -> DST (bf16 -> fp32)"
    S.dst_set(0, "x0 fp32"); S.dst_set(1, "x1 fp32")
    s["dst_owner"][S.half()] = "math"
    s["pack_cfg"] = "(not configured)"
    S.set_risc(TRISC_0=(RUN, "scale_pass(x): copy_tile x -> SrcA (unpacker 0)"), TRISC_1=(RUN, "scale_pass(x): MOVA2D datacopy instructions SrcA -> DST"), TRISC_2=(WAIT, "waits for DST"),
               NCRISC=(RUN if not q else RUN, "reads weights (see the NoC box)"))
    S.frame("scale_pass(x), block of 2 tiles: unpack and copy into DST", [
        "TRISC0 programs the unpacker: each x tile goes L1 (bf16) -> SrcA, one bank at a time (SrcA has two banks: while the FPU/datacopy reads one, the unpacker fills the other).",
        "TRISC1 issues the datacopy (MOVA2D): SrcA -> DST, converting to fp32 (the program runs the DST in fp32 mode). DST half 0 now holds x0, x1.",
        "No FPU math runs: the FPU data path is only used as a copy engine here.",
    ], t, arrows=["L1-U0", "U0-A", "A-F", "F-D"], stage="quantize x")

    t = Z("TRISC_1", "SP_SFPU", 0)[1]
    s["u0"] = "idle"; s["fpu"] = "idle"; s["srcA"] = [None, None]
    s["sfpu"] = "absmax_tile x2: 3 SFPSWAP per 2 rows"
    s["lregs"] = ["rows", "rows", "-", "-", "-", "-", "running min of x (<= 0)", "running max of x (>= 0)"]
    S.set_risc(TRISC_0=(WAIT, "(done with its copy; runs on to the next phase)"), TRISC_1=(RUN, "scale_pass(x): SFPU absmax sweep over the 2 tiles in DST"))
    S.frame("scale_pass(x): the SFPU sweeps the tiles for the absolute maximum", [
        "The SFPU (32 lanes) loads 4 rows x 8 lanes at a time from DST into LReg0..3 and compares them with SFPSWAP (sign-magnitude compare): the larger goes against the running maximum in LReg7, the smaller against the running minimum in LReg6.",
        "It needs no absolute value: |x| <= max(LReg7, -LReg6). Cost: about 3 SFPSWAP (6 Simple-unit cycles) per two rows, the SFPU is bound by SFPSWAP (about 160 cycles per tile).",
        "The tiles stay in DST, unchanged.",
    ], t, arrows=["D-S"], stage="quantize x")

    t = Z("TRISC_1", "SP_SCALER", 0)[1]
    S.release()
    S.acquire()
    S.dst_set(3, "2^kx (scaler tile)")
    s["sfpu"] = "absmax_to_scaler: reduce 32 lanes -> 2^kx"
    s["lregs"] = ["tmp", "tmp", "tmp", "tmp", "-", "-", "-", "2^kx  (the scaler stays in LReg7)"]
    s["dst_owner"][S.half()] = "math"
    S.set_risc(TRISC_1=(RUN, "SFPU: transpose + butterfly reduce, exponent -> 2^k, writes the scaler tile"), TRISC_2=(WAIT, "waits for DST (math thread will commit)"))
    S.frame("scale_pass(x): the absolute maximum becomes the scaler 2^k", [
        "The first DST half is released without packing (the tiles were only read). A second acquire gives the SFPU the other DST half.",
        "The SFPU reduces the 32 lane maxima (transpose + shuffle butterfly), takes the exponent and builds 2^k with k = the largest integer such that max|x| * 2^k <= 127. The scaler tile (2^k in every element) is written to DST slot 3 of this half.",
        "LReg7 now holds 2^k as an fp32: the quantize pass reads it from there.",
    ], t, arrows=["D-S"], stage="quantize x")

    t = Z("TRISC_2", "SP_SCALER", 0)[1]
    s["sfpu"] = "idle"
    s["packer"] = "pack_tile: DST slot 3 -> c_17 (fp32, 4 KB)"
    s["pack_cfg"] = "format Float32 -> c_17"
    S.reserve("c_17", ["2^kx"]); S.push("c_17")
    s["dst_owner"][S.half()] = "pack"
    S.set_risc(TRISC_2=(RUN, "packer: scaler tile -> c_17, then push (c_17 count + 1)"), BRISC=(RUN, "sees c_17 filled: writes the scaler tile to the scalers tensor (DRAM)"))
    s["noc"] = ["BRISC writes scaler_x: c_17 -> DRAM scalers[0]"]
    S.frame("The scaler of x is packed to c_17; the writer sends it to DRAM", [
        "TRISC2 (pack thread) waits for the commit of the math thread, then the packer writes the scaler tile (fp32, 4 KB) from DST to c_17 and pushes it.",
        "BRISC (the writer) was waiting on c_17: it now writes the tile to page 0 of the packed `scalers` output tensor with a NoC write and pops c_17.",
        "This is a result for the host (it is part of the trace); the next steps do not need it.",
    ], t, arrows=["D-P", "P-L1", "NOC-L1"], stage="quantize x")
    S.pop("c_17", 1)
    s["noc"] = []

    t = Z("NCRISC", "READ_W", 0)[1]
    s["noc"] = []
    if not q:
        S.push("c_1")
        S.set_risc(NCRISC=(WAIT, "w0 landed (push 4). Next: reserve_back(c_1, 4) for layer 1: c_1 is full, so it waits for compute to pop"))
        S.frame("The weights of layer 0 are in L1; the reader runs ahead and then blocks", [
            "The 4 weight tiles of layer 0 landed in c_1 (full: 8 KB). NCRISC pushed them.",
            "NCRISC immediately starts the next layer: reserve_back(c_1, 4). c_1 has no free page, so it blocks until compute pops the current weights. This is how reading the next layer's weights overlaps the compute of this one.",
        ], t, arrows=["NOC-L1"], stage="read")
    else:
        S.push("c_2")
        S.set_risc(NCRISC=(WAIT, "q0..q3 landed (push 4). Next: reserve_back(c_2, 4) for layer 1: c_2 is full, so it waits until the matmul of layer 0 pops it"))
        S.frame("The quantized weights of layer 0 are in c_2; the reader runs ahead and then blocks", [
            "The 4 UINT8 weight tiles of layer 0 landed directly in c_2 (4 KB): no quantization needed.",
            "NCRISC starts layer 1: reserve_back(c_2, 4) blocks until the matmul of layer 0 has consumed its weights.",
        ], t, arrows=["NOC-L1"], stage="read")

    # quantize pass of x
    t = Z("TRISC_1", "QP_COPY", 0)[1]
    S.release()
    S.acquire()
    s["srcA"] = [("x0", "reading"), ("x1", "writing")]
    s["u0"] = "copy_tile x0, x1 again: L1 -> SrcA"
    s["fpu"] = "datacopy A2D"
    S.dst_set(0, "x0 fp32"); S.dst_set(1, "x1 fp32")
    S.reserve("c_4", ["a0", "a1"])
    s["pack_cfg"] = "Int8 view (sign-magnitude bytes) -> c_4"
    S.set_risc(TRISC_0=(RUN, "quant_pass(x): copy_tile again"), TRISC_1=(RUN, "quant_pass(x): datacopy, then SFPU init (macro programming)"), TRISC_2=(WAIT, "reserve_back(c_4, 2) done; waits for DST"))
    S.frame("quant_pass(x): the tiles are unpacked a second time", [
        "The scale is known (in LReg7), so the x tiles are read again from c_0 (the first pass only measured them) and copied into DST like before. c_0 is not popped: x stays in L1.",
        "In parallel the pack thread reserved two pages in c_4 (act A) for the result.",
    ], t, arrows=["L1-U0", "U0-A", "A-F", "F-D"], stage="quantize x")

    t = Z("TRISC_1", "QP_SFPU", 0)[1]
    s["u0"] = "idle"; s["fpu"] = "idle"; s["srcA"] = [None, None]
    s["sfpu"] = "quant_sm8_lm: SFPLOADMACRO pipeline, 2 issue slots per 4x8 lanes"
    s["lregs"] = ["rows", "rows", "rows", "rows", "magic = 1.5*2^23", "-magic", "-", "2^kx"]
    s["macro"] = "macro 0: MAD (x*2^k + magic) @+1, IADD (-magic) @+4, STORE @+6; SFPCAST -> sign-magnitude issued normally"
    S.dst_set(0, "x0 int8 (sign-mag in int32)"); S.dst_set(1, "x1 int8 (sign-mag in int32)")
    S.set_risc(TRISC_1=(RUN, "SFPU quantize sweep over 2 tiles (SFPLOADMACRO)"))
    S.frame("quant_pass(x): the SFPU quantizes with a macro pipeline", [
        "Per 4 rows x 8 lanes: SFPLOADMACRO loads the rows and schedules a multiply-add (x * 2^k + 1.5 * 2^23, which rounds to nearest even in the fp32 adder), an integer add of -bits(1.5 * 2^23) (leaving the integer n) and the store; a SFPCAST turns n into sign-magnitude. Three sub-units of the SFPU work on different rows at once.",
        "DST now holds the quantized values as sign-magnitude int32 (sign bit 31, |n| <= 127).",
    ], t, arrows=["D-S"], stage="quantize x")

    t = Z("TRISC_2", "QP_PACK", 0)[1]
    s["sfpu"] = "idle"
    s["packer"] = "pack_tile x2: DST (sign-mag int32) -> c_4 as int8 bytes"
    S.push("c_4")
    s["dst_owner"][S.half()] = "pack"
    S.set_risc(TRISC_2=(RUN, "packer: 2 tiles -> c_4 (Int8 view), push 2"))
    S.frame("The quantized x (act A) is packed to c_4", [
        "The packer is configured by hand for the Int8 format (the buffer is declared UInt8): it takes sign + 7 magnitude bits from each int32 and writes 1 KB tiles to c_4 (act A). Two pages pushed.",
        "This is the first (and for the network the only) place where the float input is quantized.",
    ], t, arrows=["D-P", "P-L1"], stage="quantize x")
    S.release()
    s["dst_owner"][S.half()] = "free"

    # ---------------------------------------------------------------- quantize weights (not pre-quantized)
    if not q:
        t = Z("TRISC_1", "SP_SFPU", 1)[1]
        S.acquire()
        s["srcA"] = [("w2", "reading"), ("w3", "writing")]
        s["u0"] = "copy_tile w0..w3: L1 bf16 -> SrcA"
        s["fpu"] = "datacopy A2D"
        for i in range(4):
            S.dst_set(i, f"w{i} fp32")
        s["sfpu"] = "absmax_tile x4"
        s["lregs"] = ["rows", "rows", "-", "-", "-", "-", "running min", "running max of w"]
        S.set_risc(TRISC_0=(RUN, "scale_pass(w): copy_tile of the 4 weight tiles (it ran ahead of the others)"), TRISC_1=(RUN, "scale_pass(w): datacopy + SFPU absmax over 4 tiles (one full DST half)"))
        S.frame("scale_pass(w): the weights are copied into DST and swept for the absolute maximum", [
            "Same as for x, with 4 tiles: a full DST half (4 slots). The unpacker thread TRISC0 ran ahead during the previous steps, so the copy was already queued.",
            "The SFPU absmax sweep over the weights is the slowest part of the weight quantization (swap-bound, about 160 cycles per tile).",
        ], t, arrows=["L1-U0", "U0-A", "A-F", "F-D", "D-S"], stage="quantize w")

        t = Z("TRISC_2", "SP_SCALER", 1)[1]
        S.release(); S.acquire()
        S.dst_set(3, "2^kw (scaler tile)")
        s["srcA"] = [None, None]; s["u0"] = "idle"; s["fpu"] = "idle"
        s["lregs"] = ["tmp", "tmp", "tmp", "tmp", "-", "-", "-", "2^kw"]
        s["sfpu"] = "idle"
        s["packer"] = "pack_tile: scaler -> c_18"
        s["pack_cfg"] = "format Float32 -> c_18"
        S.reserve("c_18", ["2^kw"]); S.push("c_18")
        s["noc"] = ["BRISC writes scaler_w: c_18 -> DRAM scalers[1]"]
        S.set_risc(TRISC_2=(RUN, "packer: scaler_w tile -> c_18"), BRISC=(RUN, "writes scaler_w to DRAM"))
        S.frame("The scaler of the weights is packed to c_18 and written to DRAM", [
            "The absolute maximum of the weights becomes 2^kw (same reduction as for x). It is packed to c_18 and BRISC writes it as page 1 of the `scalers` tensor.",
        ], t, arrows=["D-S", "D-P", "P-L1", "NOC-L1"], stage="quantize w")
        S.pop("c_18", 1); s["noc"] = []

        t = Z("TRISC_0", "QP_COPY", 1)[1]
        S.pop("c_1", 4)
        S.release(); S.acquire()
        S.reserve("c_2", ["q0", "q1", "q2", "q3"])
        s["srcA"] = [("w2", "reading"), ("w3", "writing")]
        s["u0"] = "copy_tile w0..w3 again"
        s["fpu"] = "datacopy A2D"
        for i in range(4):
            S.dst_set(i, f"w{i} fp32")
        S.bg_noc = ["NCRISC reads w' (layer 1): DRAM -> c_1 (4 x 2 KB), lands at about cycle %d" % Z("NCRISC", "READ_W", 1)[1]]
        S.reserve("c_1", ["w'0", "w'1", "w'2", "w'3"])
        S.set_risc(TRISC_0=(RUN, "quant_pass(w): second copy of the weights, then pop_front(c_1, 4)"), NCRISC=(RUN, "c_1 popped: reserve_back succeeds, reads the 4 tiles of layer 1 (w')"))
        S.frame("quant_pass(w): second copy; c_1 is popped and the reader fetches layer 1", [
            "The weights are unpacked and copied into DST a second time (with the scale known). After this copy compute pops c_1 (4 pages).",
            "NCRISC, blocked in reserve_back, wakes up and reads the weights of layer 1 (marked w') into c_1 over the NoC while this core keeps computing layer 0. A DRAM read takes about 1.3 thousand cycles here, so it only just fits behind the compute of the current layer.",
        ], t, arrows=["L1-U0", "U0-A", "A-F", "F-D", "NOC-L1"], stage="quantize w")

        t = Z("TRISC_1", "QP_SFPU", 1)[1]
        s["u0"] = "idle"; s["fpu"] = "idle"; s["srcA"] = [None, None]
        s["sfpu"] = "quant_sm8_lm over 4 tiles"
        s["lregs"] = ["rows", "rows", "rows", "rows", "magic", "-magic", "-", "2^kw"]
        s["macro"] = "macro 0: MAD @+1, IADD @+4, STORE @+6 (SFPCAST in between)"
        for i in range(4):
            S.dst_set(i, f"q{i} int8")
        S.frame("quant_pass(w): the SFPU quantizes the weights", [
            "Same macro pipeline as for x, over the 4 weight tiles: DST now holds the quantized weights as sign-magnitude int32.",
        ], t, arrows=["D-S"], stage="quantize w")

        t = Z("TRISC_2", "QP_PACK", 1)[1]
        s["sfpu"] = "idle"
        s["packer"] = "pack_tile x4: DST -> c_2 (Int8 view)"
        s["pack_cfg"] = "Int8 view -> c_2"
        S.push("c_2")
        s["dst_owner"][S.half()] = "pack"
        S.set_risc(TRISC_2=(RUN, "packer: 4 tiles -> c_2, push 4"))
        S.frame("The quantized weights q_w are packed to c_2", [
            "The packer writes the 4 weight tiles (1 KB each, int8 sign-magnitude) to c_2. The weights of this layer are now ready for the matmul: they were quantized from bf16 inside the kernel.",
            "In the pre-quantized version of this program this whole phase does not exist: the reader delivers c_2 directly.",
        ], t, arrows=["D-P", "P-L1"], stage="quantize w")
        S.release()

    # ---------------------------------------------------------------- matmul
    t = Z("TRISC_0", "MM_FPU", 0)[0] + 150
    S.acquire()
    if not q:
        S.push("c_1")
        S.bg_noc = []
        S.set_risc(NCRISC=(WAIT, "w' landed in c_1 (push 4); next: reserve_back(c_1, 4) for layer 2 waits for compute to pop"))
    s["u0"] = "unpack weight tile (kt, nt) -> SrcA"
    s["u1"] = "unpack activation tile kt -> SrcB"
    s["srcA"] = [("w0", "reading"), ("w1", "writing")]
    s["srcB"] = [("a0", "reading"), ("a0", "writing")]
    s["fpu"] = "MVMUL HiFi4: a0 x w0 -> DST slot 0 (int32 accumulate)"
    S.dst_set(0, "acc0 (partial, kt=0)")
    s["pack_cfg"] = "Int32 -> c_3"
    s["lregs"] = ["-", "-", "-", "-", "-", "-", "-", "0 (absmax_init)"]
    S.set_risc(TRISC_0=(RUN, "matmul: unpack operands (srcA = weights, srcB = activations), FPU configured for Int8"), TRISC_1=(RUN, "matmul_init, then matmul_tiles x 4 (MVMUL)"), TRISC_2=(WAIT, "waits for DST"))
    mm_extra = [] if q else ["Meanwhile the prefetch of the layer-1 weights into c_1 completed (NCRISC pushed it)."]
    S.frame("Matmul, kt = 0: both operands are unpacked and the FPU multiplies", [
        "The unpackers are reconfigured by hand to read the UInt8 buffers as Int8. Unpacker 0 fills SrcA with a weight tile, unpacker 1 fills SrcB with an activation tile; both registers are double banked, so the next tile is unpacked while the FPU works on the current one.",
        "The FPU (HiFi4: four fidelity phases, needed for an exact int8 product) multiplies a 32 x 32 tile pair and accumulates the int32 result into a DST slot. matmul_tiles is called for (a0, w0) -> slot 0 and (a0, w1) -> slot 1.",
        "About 67 cycles per tile product: the FPU is the bottleneck of this phase.",
    ] + mm_extra, t, arrows=["L1-U0", "L1-U1", "U0-A", "U1-B", "A-F", "B-F", "F-D"], stage="matmul")

    t = Z("TRISC_1", "MM_FPU", 0)[1]
    s["srcA"] = [("w2", "reading"), ("w3", "writing")]
    s["srcB"] = [("a1", "reading"), ("a1", "writing")]
    s["fpu"] = "MVMUL: a1 x w2 -> slot 0, a1 x w3 -> slot 1 (accumulate)"
    S.dst_set(0, "acc0 (K = 64, final)"); S.dst_set(1, "acc1 (K = 64, final)")
    S.frame("Matmul, kt = 1: the second K block accumulates into the same DST slots", [
        "The same two output slots accumulate the products with the second half of K: a1 x w2 and a1 x w3. After this the two DST tiles hold the int32 accumulators of the layer (32 x 64 outputs = 2 tiles).",
    ], t, arrows=["L1-U0", "L1-U1", "U0-A", "U1-B", "A-F", "B-F", "F-D"], stage="matmul")

    t = Z("TRISC_1", "MM_SWEEP", 0)[1]
    s["u0"] = "idle"; s["u1"] = "idle"; s["fpu"] = "idle"; s["srcA"] = [None, None]; s["srcB"] = [None, None]
    s["sfpu"] = "acc_absmax_tile x2 (ReLU: negatives never win)"
    s["lregs"] = ["rows", "rows", "-", "-", "-", "-", "-", "running max |acc| (integer, sign-magnitude)"]
    S.set_risc(TRISC_1=(RUN, "SFPU: absmax of the activated accumulator"))
    S.frame("The SFPU folds the largest accumulator value into LReg7", [
        "The requantization needs the global maximum of the (activated) accumulator before it can start. The SFPU sweeps both DST tiles once (SFPSWAP against LReg7 on the raw sign-magnitude bits, about 180 cycles per tile) and leaves the tiles untouched: the packer will write the raw accumulators.",
        "The FPU and the SFPU do not overlap: the SFPU start waits until the FPU is idle (a barrier in the math thread).",
    ], t, arrows=["D-S"], stage="matmul")

    t = Z("TRISC_2", "MM_PACK", 0)[1]
    s["sfpu"] = "idle"
    s["packer"] = "pack_tile x2: DST -> c_3 as Int32 (raw bits)"
    S.reserve("c_3", ["acc0", "acc1"]); S.push("c_3")
    s["dst_owner"][S.half()] = "pack"
    S.set_risc(TRISC_2=(RUN, "packer: 2 accumulator tiles (4 KB each) -> c_3, push 2"))
    S.frame("The raw accumulators are packed to c_3", [
        "The packer writes the two int32 accumulator tiles (4 KB each) to c_3. These raw sign-magnitude values are what the host receives as the `acc` output (the witness of the matmul).",
        "The matmul operands are no longer needed: the next step shows them popped.",
    ], t, arrows=["D-P", "P-L1"], stage="matmul")
    S.release()
    S.pop("c_4", 2); S.pop("c_2", 4)
    if q:
        S.bg_noc = ["NCRISC reads q' (layer 1): DRAM -> c_2 (4 x 1 KB), lands at about cycle %d" % Z("NCRISC", "READ_W", 1)[1]]
        S.reserve("c_2", ["q'0", "q'1", "q'2", "q'3"])
        S.set_risc(NCRISC=(RUN, "c_2 popped by the matmul: reads the quantized weights of layer 1 into c_2"))
    S.frame("c_2 and c_4 are popped" + ("; the reader fetches the next weights" if q else ""), [
        "The unpacker thread popped the matmul operands. c_4 (act A) is free again: it will receive the output of layer 1.",
    ] + (["NCRISC was blocked in reserve_back(c_2): now it reads the pre-quantized weights of layer 1 into c_2 (4 KB), overlapping the requantization below."] if q else []), t + 1, arrows=["NOC-L1"] if q else [], stage="matmul")

    # ---------------------------------------------------------------- requantize
    t = Z("TRISC_1", "RQ_SCALER", 0)[1]
    S.acquire()
    S.dst_set(3, "2^k (requant scaler tile)")
    s["sfpu"] = "absmax_to_scaler (float variant) + scaler_to_lreg7"
    s["lregs"] = ["tmp", "tmp", "tmp", "tmp", "-", "-", "-", "2^k (fp32, from the scaler tile)"]
    s["packer"] = "pack_tile: scaler -> c_19 (reserved, NOT pushed yet)"
    s["pack_cfg"] = "format Float32 -> c_19"
    S.reserve("c_19", ["2^k out"])
    s["dst_owner"][S.half()] = "math"
    S.set_risc(TRISC_1=(RUN, "SFPU: from max |acc| to the scaler 2^k of the requantization"), TRISC_2=(RUN, "packer: scaler tile -> c_19 (the push comes last)"))
    S.frame("The absolute maximum of the accumulator becomes the requantization scaler", [
        "The integer maximum in LReg7 is reduced across the lanes, k = the largest integer with max|acc| * 2^k <= 127 is derived with a count-leading-zeros, and 2^k is written as an fp32 scaler tile to DST. It is also loaded back into LReg7 as the multiplier of the sweep (the compiler may use LReg7 as scratch, so it is re-derived from the tile).",
        "The scaler tile is packed to c_19 but NOT pushed yet: pushing it is the signal 'this layer is done, the accumulator in c_3 has been read' for the writer, so it must come last.",
    ], t, arrows=["D-S", "D-P", "P-L1"], stage="requantize")

    t = Z("TRISC_1", "QP_COPY", 1 if q else 2)[1]
    S.release(); S.acquire()
    s["u0"] = "unpack-to-DST: acc tiles (Int32, 4 KB) c_3 -> DST directly"
    s["srcA"] = [None, None]
    s["fpu"] = "idle (the unpacker bypasses SrcA)"
    S.dst_set(0, "acc0 raw"); S.dst_set(1, "acc1 raw")
    S.reserve("c_5", ["b0", "b1"])
    if q:
        S.push("c_2")
        S.bg_noc = []
        S.set_risc(NCRISC=(WAIT, "q' landed in c_2 (push 4); next: reserve_back(c_2, 4) for layer 2 waits for the matmul of layer 1"))
    s["sfpu"] = "idle: waits for the unpacker (math_unpack_to_dest_math_ready)"
    s["pack_cfg"] = "Int8 view + packer ReLU (hidden layers) -> c_5"
    S.set_risc(TRISC_0=(RUN, "copy_tile with unpack-to-dest: the 32-bit accumulator goes straight into DST"), TRISC_1=(WAIT, "waits until the unpacker has written DST"))
    S.frame("Requantize: the accumulator is unpacked directly into DST (the SFPU waits)", [
        "The accumulator tiles are 4 KB of 32-bit data: the unpacker writes them straight into DST (bypassing SrcA and SrcB). While it does, the SFPU may not touch DST: the math thread waits for the unpacker (a semaphore). This is why the unpack is not overlapped with the SFPU sweep.",
        "This copy (about 130 - 180 cycles per tile) is the largest part of what remains of the requantization after the SFPU sweep was shortened.",
    ], t, arrows=["L1-U0", "U0-D"], stage="requantize")

    t = Z("TRISC_1", "QP_SFPU", 1 if q else 2)[1]
    s["u0"] = "idle"
    s["sfpu"] = "requant_lm: SFPLOADMACRO, 3 issue slots per 4x8 lanes"
    s["lregs"] = ["rows", "rows", "rows", "rows", "magic", "-magic", "-", "2^k"]
    s["macro"] = "macro 0: MAD (v*2^k + magic) @+3, SFPCAST int->sign-mag @+6, STORE @+7; SFPCAST sm->fp32 @+1 and SFPIADD @+2 issued normally"
    S.dst_set(0, "b0 (int8 value as int32 sign-mag)"); S.dst_set(1, "b1")
    S.set_risc(TRISC_1=(RUN, "SFPU requantize sweep over 2 tiles"))
    S.frame("Requantize: the SFPU pipeline converts the accumulator to int8", [
        "Per 4 rows x 8 lanes in 3 issue slots: SFPLOADMACRO loads the raw accumulator; SFPCAST converts sign-magnitude -> fp32 (exact for |acc| <= 2^24, i.e. K <= 1024); the MAD sub-unit computes v * 2^k + 1.5 * 2^23 (round to nearest even); an integer add leaves n; a SFPCAST gives sign-magnitude; STORE writes it back.",
        "Compared with the integer sequence this replaced (11 to 13 instructions) it is about 3 times shorter. The ReLU is not done here: the packer will do it.",
    ], t, arrows=["D-S"], stage="requantize")

    t = Z("TRISC_2", "QP_PACK", 1 if q else 2)[1]
    s["sfpu"] = "idle"
    s["packer"] = "pack_tile x2 with hardware ReLU: DST -> c_5 as int8"
    S.push("c_5")
    S.push("c_19")
    s["dst_owner"][S.half()] = "pack"
    S.set_risc(TRISC_2=(RUN, "packer: relu_config(zero); 2 tiles -> c_5; then pushes the scaler (c_19)"))
    S.frame("The packer applies the ReLU and writes the next layer's input to c_5", [
        "The packer's own ReLU zeroes every value whose sign bit is set (the data is sign-magnitude, so negative = sign bit): the ReLU of the layer costs nothing. The 2 tiles of int8 activations go to c_5 (act B) and are pushed: they are the input of layer 1, never converted back to float.",
        "Then c_19 (the scaler of this requantization) is pushed: the writer's signal that this layer is complete.",
    ], t, arrows=["D-P", "P-L1"], stage="requantize")
    S.release()

    t = Z("BRISC", "WRITE", 0)[0]
    S.set_risc(BRISC=(RUN, "WAIT_DONE over: writes the 2 accumulator tiles (c_3) and the scaler (c_19) to DRAM, pops them"))
    s["noc"] = ["BRISC writes acc0, acc1: c_3 -> DRAM acc[tile (0, c)]", "BRISC writes scaler_out: c_19 -> DRAM scalers[..]"]
    S.frame("The writer sends the accumulator and the scaler of layer 0 to DRAM", [
        "BRISC waited for c_19 (the scaler) and for c_3 to be complete. It writes the 2 accumulator tiles as pages (0, c) of the packed `acc` tensor (4 KB each), then the scaler tile, and pops them. c_3 is free for the next layer.",
        "The output tensors are all written by this core directly to DRAM; the host only allocated them.",
    ], t, arrows=["NOC-L1"], stage="write")
    S.pop("c_3", 2); S.pop("c_19", 1)
    s["noc"] = []
    S.set_risc(BRISC=(WAIT, "waits for the next layer's scaler (c_19)"))
    return S


def build_last(S, Z):
    """Continues `S` (after layer 0) with the last layer of a network: input in act B (c_5), output into the ring c_16."""
    s = S.s
    q = S.pre
    S.layer = 1
    # ---- weights of the layer
    if not q:
        t = Z("TRISC_2", "QP_PACK", 3)[1]
        S.pop("c_1", 4)
        S.reserve("c_2", ["q'0", "q'1", "q'2", "q'3"]); S.push("c_2")
        s["noc"] = []
        s["packer"] = "pack_tile x4 (Int8 view) -> c_2"
        S.set_risc(NCRISC=(WAIT, "(has read w'; waits for c_1 for layer 2)"), TRISC_0=(RUN, "scale_pass / quant_pass of w' (the data was prefetched into c_1)"), TRISC_2=(RUN, "packs q' to c_2"))
        S.frame("Layer 1 (the last): the weights are quantized from the prefetched c_1", [
            "Same two passes as for layer 0 (absmax + scaler, then quantize), on the weights that NCRISC prefetched into c_1 during layer 0: compute never waits for DRAM (WAIT_W is 0 in the profile).",
            "(Layers in the middle of a network are identical to layer 0 and to this one, with the two activation buffers swapped; see the summary page.)",
        ], t, arrows=["L1-U0", "U0-A", "A-F", "F-D", "D-S", "D-P", "P-L1"], stage="quantize w")
    # ---- matmul
    t = Z("TRISC_1", "MM_FPU", 1)[1]
    S.acquire()
    s["u0"] = "unpack weight tiles -> SrcA"; s["u1"] = "unpack act B (c_5) tiles -> SrcB"
    s["srcA"] = [("q'2", "reading"), ("q'3", "writing")]; s["srcB"] = [("b1", "reading"), ("b1", "writing")]
    s["fpu"] = "MVMUL HiFi4: act B x q' -> DST (int32)"
    S.dst_set(0, "acc'0"); S.dst_set(1, "acc'1")
    s["pack_cfg"] = "Int32 -> c_3"
    s["lregs"] = ["-", "-", "-", "-", "-", "-", "-", "0 (absmax_init)"]
    S.set_risc(TRISC_0=(RUN, "unpacks operands: srcB reads c_5 (act B), the output of layer 0"), TRISC_1=(RUN, "matmul_tiles x 4"), TRISC_2=(WAIT, "waits for DST"))
    S.frame("Layer 1 matmul: the input is act B, the int8 output of layer 0", [
        "The activations never left int8 and never left L1: layer 0 wrote them to c_5 and layer 1 reads them from there (ping-pong: even layers read c_4 and write c_5, odd layers the opposite).",
        "The matmul itself is the same as before; both DST slots accumulate over the two K blocks.",
    ], t, arrows=["L1-U0", "L1-U1", "U0-A", "U1-B", "A-F", "B-F", "F-D"], stage="matmul")

    t = Z("TRISC_2", "MM_PACK", 1)[1]
    s["u0"] = "idle"; s["u1"] = "idle"; s["fpu"] = "idle"; s["srcA"] = [None, None]; s["srcB"] = [None, None]
    s["sfpu"] = "idle"; s["packer"] = "pack_tile x2: DST -> c_3 (Int32)"
    s["lregs"] = ["-", "-", "-", "-", "-", "-", "-", "max |acc'| (after the sweep)"]
    S.reserve("c_3", ["acc'0", "acc'1"]); S.push("c_3")
    s["dst_owner"][S.half()] = "pack"
    S.release()
    S.pop("c_5", 2); S.pop("c_2", 4)
    S.set_risc(TRISC_1=(RUN, "SFPU absmax sweep done"), TRISC_2=(RUN, "packs the raw accumulators to c_3"))
    S.frame("Layer 1: sweep, pack, and the operands are popped", [
        "Absmax sweep (SFPU) and the pack of the raw accumulators to c_3 (4 KB x 2) as before. The matmul operands c_5 and c_2 are popped.",
    ], t, arrows=["D-S", "D-P", "P-L1"], stage="matmul")

    # ---- requant into the output ring
    t = Z("TRISC_1", "QP_SFPU", 2 if q else 4)[1]
    S.acquire()
    S.dst_set(0, "o0 (int8 value, sign-mag)"); S.dst_set(1, "o1")
    s["u0"] = "idle"
    s["sfpu"] = "requant_lm over 2 tiles"
    s["lregs"] = ["rows", "rows", "rows", "rows", "magic", "-magic", "-", "2^k'"]
    s["pack_cfg"] = "Int8 view, no ReLU (last layer) -> c_16 ring"
    S.reserve("c_19", ["2^k' out"])
    S.set_risc(TRISC_0=(RUN, "unpack-to-dest of the accumulator"), TRISC_1=(RUN, "requantize sweep"))
    S.frame("Layer 1 requantization: scaler, unpack-to-DST and the SFPU sweep", [
        "Same steps as for layer 0 (scaler from the integer maximum, accumulator unpacked into DST, SFPU requant pipeline). The last layer has no ReLU in this network.",
        "Its output does not go to an activation buffer but to the output ring c_16 (8 pages): the writer drains it as it is produced.",
    ], t, arrows=["L1-U0", "U0-D", "D-S"], stage="requantize")

    t = Z("TRISC_2", "QP_PACK", 2 if q else 4)[1]
    s["sfpu"] = "idle"; s["packer"] = "pack_tile x2: DST -> c_16 (Int8 view)"
    S.reserve("c_16", ["y0", "y1"]); S.push("c_16")
    s["dst_owner"][S.half()] = "pack"
    S.push("c_19")
    s["noc"] = ["BRISC writes y0, y1: c_16 -> DRAM out"]
    S.set_risc(TRISC_2=(RUN, "packs the output tiles to the ring c_16 and pushes the scaler"), BRISC=(RUN, "drains c_16: NoC writes of the 2 output tiles (1 KB each) to the `out` tensor"))
    S.frame("The network's output is packed to the ring and written to DRAM", [
        "2 tiles of int8 output go to the ring c_16 and BRISC writes each tile to the `out` tensor in DRAM as soon as it is pushed (the ring has 8 pages: 2 batches of 4).",
        "Then the scaler is pushed to c_19, the writer's 'layer done' signal.",
    ], t, arrows=["D-P", "P-L1", "NOC-L1"], stage="requantize")
    S.release()
    S.pop("c_16", 2)
    s["noc"] = []

    t = Z("BRISC", "WRITE", 1)[1]
    s["noc"] = ["BRISC writes acc', scaler_out': c_3, c_19 -> DRAM"]
    S.set_risc(BRISC=(RUN, "writes the last accumulator (2 tiles) and the last scaler to DRAM, then waits for the NoC writes to complete (barrier)"), NCRISC=(IDLE, "done"), TRISC_0=(IDLE, "kernel ended"), TRISC_1=(IDLE, "kernel ended"), TRISC_2=(IDLE, "kernel ended"))
    S.pop("c_3", 2); S.pop("c_19", 1)
    S.frame("The last accumulator and scaler are written; the kernel ends", [
        "BRISC writes the accumulator tiles and the scaler of the last layer, then does a final async_write_barrier: when it returns, every output (out, scalers, acc) is complete in DRAM.",
        "The three compute cores and NCRISC finished earlier; BRISC ends last (that is the end of the program).",
    ], t, arrows=["NOC-L1"], stage="write")
    s["noc"] = []
    S.set_risc(BRISC=(IDLE, "kernel ended"))
    return S
