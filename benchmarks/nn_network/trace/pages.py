# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""The pages around the steps: guide, hardware map, circular buffers, the whole run (8 layers), the middle layers, why the requantization is bound."""
import os
from svg import Svg
import draw
from draw import W, H, wrap, ZONE_COLOR, COARSE, legend
import model

PANEL = "#fff"


def header(g, title, sub):
    g.rect(0, 0, W, 96, fill="#1b2430", stroke="none", rx=0)
    g.text(20, 42, title, size=28, fill="#fff", weight="bold")
    g.text(20, 74, sub, size=15, fill="#d0d7e2")


def para(g, x, y, text, width, size=15, dy=None, **kw):
    dy = dy or size * 1.45
    n = 0
    for ln in wrap(text, width):
        g.text(x, y + n * dy, ln, size=size, **kw)
        n += 1
    return y + n * dy


def save(d, name, g):
    p = os.path.join(d, name)
    open(p, "w").write(g.done())
    return name


def stage_totals(czones, risc="TRISC_1"):
    """-> per layer {stage: cycles} from the coarse zones of the run (T1)."""
    out, cur = [], None
    for n, s, e, dpt in sorted(czones[risc], key=lambda z: z[1]):
        if n == "QUANT_X":
            pass
        if n in ("QUANT_W", "MATMUL", "REQUANT", "WAIT_ACC", "WAIT_W"):
            if n in ("QUANT_W",) or (n == "MATMUL" and (cur is None or "MATMUL" in cur)) or cur is None:
                pass
    layers = []
    for n, s, e, dpt in sorted(czones[risc], key=lambda z: z[1]):
        if n in ("QUANT_W", "MATMUL", "REQUANT") and dpt <= 3:
            layers.append((n, s, e))
    return layers


def front_pages(key, v, frames, zones, mhz, tmax, czones, cmhz, d):
    pre = v["pre"]
    out = []
    # ---------------------------------------------------------------- 1: guide
    g = Svg(W, H)
    header(g, f"One Tensix core running a 32 x 64 network of 8 layers: {v['short']}", "Step-by-step diagrams of execute_network (bf16 input) on a Blackhole Tensix core. Each page is a snapshot; the next page shows what changed.")
    y = 130
    g.text(30, y, "What these pages show", size=22, weight="bold"); y += 14
    blocks = [
        "Each page after this one is one STEP of the execution on ONE Tensix core: the circular buffers in L1 (left), the coprocessor units - unpackers, SrcA/SrcB (double banked), FPU, DST, SFPU with its local registers, packer (middle) - and the five baby RISC-V cores with the NoC / DRAM (right). The bottom shows the profiler timeline of the whole run with a cursor at the time of the step, and the text says what happened.",
        "A step is not a fixed number of cycles: it ends at the next change of state (a push or pop of a circular buffer, a unit starting or finishing, a NoC transfer landing). Components that changed since the previous page have an ORANGE outline, so you can see what ran in parallel.",
        "What is MEASURED and what is MODELLED. Measured with the device profiler: the start and end cycle of zones on each of the five RISC-V cores (NCRISC, BRISC, TRISC0/1/2). The timeline, the time of every step and the 'profiler zone now' line of every core come from the profile of this program (2 layers, fine-grained zones). Modelled: the state of the coprocessor units (SrcA/SrcB banks, DST slots, LRegs, the SFPLOADMACRO configuration) and the exact fill of the circular buffers: they follow from the kernel code and the tt-metal / tt-llk protocol but the profiler cannot observe them. The three TRISC cores issue instructions to the coprocessor and wait on semaphores; the coprocessor units run behind them, so a TRISC zone is the time that core spent issuing or waiting, not the busy time of a unit.",
        "The network: x is 32 x 64 (bf16, 2 x 2 KB tiles), every layer is 64 x 64 (4 weight tiles), ReLU after every layer except the last. Per layer: quantize the weights (absmax pass, scaler, quantize pass), int8 matmul (FPU, HiFi4) into an int32 accumulator, absmax sweep, requantize (SFPU) into the next layer's int8 input. Layer 0 is shown in full, then the last layer; the layers in between repeat layer 0 with the two activation buffers swapped.",
    ]
    if pre:
        blocks.append("This is the PRE-QUANTIZED version (execute_network_quantized): the weights were quantized once beforehand, so the reader delivers int8 tiles straight into c_2, there is no c_1 and no c_18, and compute has no weight quantization phase. Compare with 32x64l8.pdf.")
    else:
        blocks.append("The weights are quantized inside the kernel on every call (execute_network).")
    for b in blocks:
        y = para(g, 30, y + 26, b, 200, size=15) + 10
    y += 20
    g.text(30, y, "Colors", size=20, weight="bold")
    y += 26
    for i, (name, col) in enumerate([("RISC-V core running", draw.C_RUN), ("waiting (on a semaphore / CB)", draw.C_WAIT), ("idle", draw.C_IDLE), ("CB page ready (pushed)", "#b2f2bb"), ("CB page reserved (being written)", "#ffe8cc"), ("changed since the previous step", "#fff")]):
        xx = 30 + i * 285
        g.rect(xx, y, 22, 22, fill=col, stroke=draw.CHANGED if name.startswith("changed") else "#555", sw=3 if name.startswith("changed") else 1)
        g.text(xx + 30, y + 17, name, size=14)
    y += 50
    g.text(30, y, "Timeline colors", size=20, weight="bold")
    legend(g, 30, y + 22)
    out.append(save(d, "page_a1.svg", g))

    # ---------------------------------------------------------------- 2: hardware map
    g = Svg(W, H)
    header(g, "The Tensix core: units and data paths", "Who moves data, where it is stored and which units can work at the same time")
    # DRAM / NoC
    g.rect(40, 130, 340, 90, fill="#e7f5ff", stroke="#555")
    g.text(52, 156, "DRAM (8 banks) + NoC", size=18, weight="bold")
    g.text(52, 180, "inputs (x, weights, scalers), outputs (out, scalers, acc)", size=12)
    g.text(52, 198, "reached only by NoC reads / writes issued by NCRISC and BRISC", size=12)
    # RISCs
    for i, (n, d_) in enumerate([("NCRISC", "reader: DRAM -> L1 (x, weights)"), ("BRISC", "writer: L1 -> DRAM (outputs)")]):
        g.rect(40, 250 + i * 80, 340, 66, fill="#d3f9d8", stroke="#555")
        g.text(52, 276 + i * 80, n, size=16, weight="bold")
        g.text(52, 298 + i * 80, d_, size=12)
    for i, (n, d_) in enumerate([("TRISC0", "unpack thread: programs the unpackers"), ("TRISC1", "math thread: FPU + SFPU instructions"), ("TRISC2", "pack thread: programs the packer")]):
        g.rect(40, 420 + i * 80, 340, 66, fill="#fff3bf", stroke="#555")
        g.text(52, 446 + i * 80, n, size=16, weight="bold")
        g.text(52, 468 + i * 80, d_, size=12)
    g.text(52, 665, "The 5 cores run C++; the compute kernel is compiled for", size=12)
    g.text(52, 681, "all three TRISCs. They push instructions into the coprocessor.", size=12)
    # L1
    g.rect(470, 130, 300, 560, fill="#f8f9fa", stroke="#555")
    g.text(482, 156, "L1 (1.5 MB)", size=18, weight="bold")
    for i, nme in enumerate(["c_0 x", "c_1 w" if not pre else "(no c_1 in this version)", "c_2 q_w", "c_3 acc", "c_4 / c_5 act", "c_16 out ring", "c_17..c_19 scalers"]):
        g.rect(486, 180 + i * 70, 268, 54, fill="#fff", stroke="#888")
        g.text(498, 212 + i * 70, nme, size=14)
    # coprocessor
    g.rect(860, 130, 880, 560, fill="#fff", stroke="#555", sw=2)
    g.text(872, 156, "Tensix coprocessor", size=18, weight="bold")
    boxes = {
        "U0": (890, 190, 150, 54, "Unpacker 0"), "U1": (890, 290, 150, 54, "Unpacker 1"),
        "A": (1110, 190, 170, 54, "SrcA (2 banks)"), "B": (1110, 290, 170, 54, "SrcB (2 banks)"),
        "F": (1350, 230, 160, 74, "FPU (matrix)"), "D": (1110, 410, 400, 90, "DST (8 tiles fp32 = 2 halves)"),
        "S": (890, 430, 170, 74, "SFPU (32 lanes)"), "P": (1110, 590, 220, 60, "Packer"),
    }
    for k, (x, y_, w_, h_, t_) in boxes.items():
        g.rect(x, y_, w_, h_, fill="#e7f5ff" if k in "AB" else "#f1f3f5", stroke="#555")
        g.text(x + 8, y_ + 22, t_, size=14, weight="bold")
    def a(x1, y1, x2, y2):
        g.line(x1, y1, x2, y2, stroke="#1971c2", sw=2.2, arrow=True)
    a(770, 217, 890, 217); a(770, 317, 890, 317)
    a(1040, 217, 1110, 217); a(1040, 317, 1110, 317)
    a(1280, 217, 1350, 255); a(1280, 317, 1350, 285); a(1430, 304, 1330, 410)
    a(965, 344, 1110, 440); g.text(905, 390, "unpack-to-DST", size=11, fill="#1971c2")
    a(1060, 466, 1110, 466); a(1110, 480, 1060, 480)
    a(1250, 500, 1250, 590); a(1110, 620, 770, 620)
    g.text(890, 175, "L1 -> unpackers", size=11, fill="#1971c2"); g.text(860, 655, "packer -> L1", size=11, fill="#1971c2")
    notes = [
        "Unpackers read tiles from L1 and convert the format (bf16, fp32, int8, int32 ...) into SrcA / SrcB, or - for 32-bit data - straight into DST.",
        "SrcA and SrcB have two banks each: the unpacker fills one while the FPU reads the other (double banking).",
        "The FPU multiplies a 32x32 tile pair (HiFi4 = 4 fidelity phases) and accumulates into DST. Here it also acts as a copy engine (datacopy SrcA -> DST).",
        "DST holds 8 fp32 tiles in two halves (half sync): the math threads own one half while the packer drains the other.",
        "The SFPU is a 32-lane vector unit with 8 local registers (LReg0..7); it reads and writes DST rows. SFPLOADMACRO lets several of its sub-units work on different rows at once.",
        "The packer reads DST, converts the format, can apply a ReLU, accumulate in L1, mask rows or untilize, and writes tiles to L1.",
    ]
    yy = 730
    for n in notes[:3]:
        yy = para(g, 40, yy, "• " + n, 112, size=13) + 2
    yy = 730
    for n in notes[3:]:
        yy = para(g, 900, yy, "• " + n, 112, size=13) + 2
    out.append(save(d, "page_a2.svg", g))

    # ---------------------------------------------------------------- 3: CBs
    g = Svg(W, H)
    header(g, "Circular buffers of this program (32 x 64, 8 layers)", "Sizes follow the plan of execute_network: one buffer set reused by every layer, padded to its capacity so that the read / write pointers are back at the start")
    st = model.new_state(pre)
    rows = [
        ("c_0", "x", "NCRISC", "compute (unpack), twice", "resident for the whole run: never popped"),
        ("c_1", "w (bf16 weights of one layer)", "NCRISC", "compute (unpack), twice", "popped after the quantize pass; refilled with the next layer's weights while this layer computes"),
        ("c_2", "q_w (int8 weights, UInt8 buffer)", "compute pack" if not pre else "NCRISC", "matmul (unpack)", "popped after the matmul; refilled for the next layer"),
        ("c_3", "acc (int32 raw sign-magnitude)", "compute pack", "compute (requantize) + BRISC", "popped by BRISC after the layer is done"),
        ("c_4", "act A (int8)", "compute pack (even layers' input: x quantized)", "matmul of even layers", "ping-pong with c_5"),
        ("c_5", "act B (int8)", "requantize of even layers", "matmul of odd layers", "ping-pong with c_4"),
        ("c_16", "out ring (int8)", "requantize of the last layer", "BRISC (drains it)", "ring of 8 pages (2 blocks of 4)"),
        ("c_17", "scaler_x", "compute pack", "BRISC", "one tile, written once"),
        ("c_18", "scaler_w", "compute pack", "BRISC", "one tile per layer"),
        ("c_19", "scaler_out", "compute pack (pushed last)", "BRISC", "its push = 'the layer is done'"),
    ]
    xs = [30, 120, 560, 760, 1010, 1300]
    for x_, t_ in zip(xs, ["CB", "content", "written by", "read by", "life", ""]):
        g.text(x_, 140, t_, size=15, weight="bold")
    y = 160
    for r in rows:
        if r[0] not in st["cb"]:
            continue
        c = st["cb"][r[0]]
        g.rect(20, y - 18, W - 40, 62, fill="#fff", stroke="#ccc", sw=0.8)
        g.text(30, y + 6, r[0], size=14, weight="bold")
        g.text(30, y + 26, f"{c['pages']} x {c['bytes'] // 1024} KB = {c['pages'] * c['bytes'] // 1024} KB", size=11, fill="#555")
        for x_, t_ in zip(xs[1:5], r[1:5]):
            ln = wrap(t_, {120: 54, 560: 24, 760: 30, 1010: 38}[x_] if x_ in (120, 560, 760, 1010) else 40)
            for j, l_ in enumerate(ln[:2]):
                g.text(x_, y + 6 + j * 16, l_, size=12)
        y += 70
    total = sum(c["pages"] * c["bytes"] for c in st["cb"].values())
    g.text(30, y + 20, f"Total circular buffers in this version: {total // 1024} KB of the 1.5 MB L1 (plus the kernel code and the runtime arguments, about 23 KB of text for any number of layers).", size=15)
    y = para(g, 30, y + 50, "Why 'reused, padded to capacity'? A CB is read by tile INDEX from its current read pointer. The tt-metal pointers rotate by one page per push and pop, so a buffer that holds fewer tiles than its capacity in some layer would leave the pointer in the middle. Every layer therefore pushes and pops the FULL capacity (the unused pages are padding), and the pointers are back at the start. With the two activation buffers capacity = the largest layer input.", 215, size=14)
    out.append(save(d, "page_a3.svg", g))

    # ---------------------------------------------------------------- 4: whole run (8 layers)
    g = Svg(W, H)
    tmax8 = max(e for r in czones for n, s_, e, d_ in czones[r] if n.endswith("-KERNEL"))
    header(g, "The whole run: 8 layers (measured, coarse zones)", f"Kernel time {tmax8} cycles = {tmax8 / cmhz:.1f} us at {cmhz / 1000:.2f} GHz. The five cores run decoupled: each lane is one core")
    lanes = [("NCRISC", "NCRISC reader"), ("BRISC", "BRISC writer"), ("TRISC_0", "TRISC0 unpack"), ("TRISC_1", "TRISC1 math"), ("TRISC_2", "TRISC2 pack")]
    lx, lw, y0 = 160, W - 200, 150
    sx = lw / tmax8
    colz = {"QUANT_X": "#74c0fc", "QUANT_W": "#b197fc", "MATMUL": "#ffa94d", "REQUANT": "#69db7c", "WAIT_ACC": "#ff8787", "WAIT_W": "#ff8787", "READ_W": "#63e6be", "WAIT_DONE": "#ced4da", "WRITE": "#51cf66"}
    for i, (r, lab) in enumerate(lanes):
        yy = y0 + i * 110
        g.rect(lx, yy, lw, 90, fill="#fff", stroke="#bbb", sw=0.8, rx=2)
        g.text(30, yy + 50, lab, size=15)
        for n, s_, e, d_ in sorted(czones[r], key=lambda z: z[3]):
            if n in colz:
                g.rect(lx + s_ * sx, yy + 6, max((e - s_) * sx, 1.5), 78, fill=colz[n], stroke="#fff", sw=0.5, rx=0)
    for k in range(0, int(tmax8) + 1, 1000):
        g.line(lx + k * sx, y0 + 5 * 110 - 6, lx + k * sx, y0 + 5 * 110 + 2, stroke="#777", sw=1)
        g.text(lx + k * sx, y0 + 5 * 110 + 18, f"{k}", size=11, anchor="middle", fill="#666")
    xx = 30
    for n, c in [("quantize x", "#74c0fc"), ("quantize weights", "#b197fc"), ("matmul + absmax sweep", "#ffa94d"), ("requantize", "#69db7c"), ("NoC read of weights (incl. wait)", "#63e6be"), ("writer waits for compute", "#ced4da"), ("writer writes", "#51cf66")]:
        g.rect(xx, y0 + 5 * 110 + 40, 16, 16, fill=c, stroke="#888", sw=0.6); g.text(xx + 22, y0 + 5 * 110 + 53, n, size=12); xx += 255
    # per layer table
    lay = []
    t1 = sorted(czones["TRISC_1"], key=lambda z: z[1])
    qx = [z for z in t1 if z[0] == "QUANT_X"]
    mm = [z for z in t1 if z[0] == "MATMUL"]
    rq = [z for z in t1 if z[0] == "REQUANT"]
    qw = [z for z in t1 if z[0] == "QUANT_W"]
    yy = y0 + 5 * 110 + 90
    g.text(30, yy, "Cycles per layer on the math thread (TRISC1):", size=16, weight="bold")
    hdr = ["layer", "quantize weights", "matmul stage", "requantize stage", "layer total"]
    for j, h_ in enumerate(hdr):
        g.text(30 + j * 220, yy + 28, h_, size=13, weight="bold")
    for L_ in range(len(mm)):
        r_ = yy + 50 + L_ * 22
        a = (qw[L_][2] - qw[L_][1]) if qw and L_ < len(qw) else 0
        b = mm[L_][2] - mm[L_][1]
        c = rq[L_][2] - rq[L_][1]
        g.text(30, r_, f"{L_}" + (" (also quantizes x first)" if L_ == 0 and qx else ""), size=13)
        g.text(250, r_, f"{a}" if not pre else "- (pre-quantized)", size=13)
        g.text(470, r_, f"{b}", size=13)
        g.text(690, r_, f"{c}", size=13)
        g.text(910, r_, f"{a + b + c}", size=13)
    out.append(save(d, "page_a4.svg", g))
    return out


def back_pages(key, v, frames, zones, mhz, tmax, czones, cmhz, d):
    pre = v["pre"]
    out = []
    # ---------------------------------------------------------------- middle layers
    g = Svg(W, H)
    header(g, "The layers in the middle: the same steps with the two activation buffers swapped", "Ping-pong activations: layer l reads one int8 buffer and writes the other; the weights are read ahead; everything else is reused")
    y = 140
    g.text(30, y, "Which buffers layer l uses (L = 8)", size=20, weight="bold")
    cols = ["layer", "reads (input)", "writes (output)", "weights", "notes"]
    xs = [30, 150, 470, 790, 1180]
    for x_, c_ in zip(xs, cols):
        g.text(x_, y + 34, c_, size=14, weight="bold")
    for L_ in range(8):
        yy = y + 62 + L_ * 34
        g.rect(20, yy - 20, W - 40, 30, fill="#f8f9fa" if L_ % 2 == 0 else "#fff", stroke="none", rx=0)
        inp = "c_0 x (quantized into c_4 first)" if L_ == 0 else ("c_4 act A" if L_ % 2 == 0 else "c_5 act B")
        outp = "c_16 out ring" if L_ == 7 else ("c_5 act B" if L_ % 2 == 0 else "c_4 act A")
        if L_ == 0:
            inp = "c_4 act A (x quantized into it)"
        wts = ("c_1 -> quantized into c_2" if not pre else "c_2 (read ahead by NCRISC)")
        note = {0: "x is quantized once, here", 7: "no ReLU; output drained by BRISC"}.get(L_, "ReLU in the packer")
        g.text(30, yy, str(L_), size=14); g.text(150, yy, inp, size=14); g.text(470, yy, outp, size=14); g.text(790, yy, wts, size=14); g.text(1180, yy, note, size=14)
    y = 140 + 62 + 8 * 34 + 30
    y = para(g, 30, y, "Steady state of every middle layer (see the timeline page of the whole run): while the compute cores work on layer l, NCRISC is already reading the weights of layer l + 1 into " + ("c_1" if not pre else "c_2") + " (its zone READ_W is 80% busy but that includes waiting), and BRISC writes the accumulator and the scalers of layer l to DRAM as soon as compute pushed the scaler of the requantization (the signal that the layer is done). Compute never waits for DRAM (WAIT_W and WAIT_ACC are about 0), so the layer time is the compute time of the three TRISC cores.", 215, size=15)
    y += 14
    y = para(g, 30, y, "Per layer the compute cores do: " + ("quantize weights (absmax pass, scaler, quantize pass); " if not pre else "") + "int8 matmul (4 tile products here, 67 cycles each plus the unpack), SFPU absmax sweep of the accumulator, requantize (scaler, unpack-to-DST of the accumulator, SFPU pipeline, pack with ReLU). The three TRISC cores are decoupled: the unpack thread runs ahead (it is often one stage in front of the math thread) and the pack thread behind it; they synchronise only through the circular buffers, DST ownership and the SrcA / SrcB valid flags.", 215, size=15)
    out.append(save(d, "page_b1.svg", g))

    # ---------------------------------------------------------------- why requant is bound by the unpack
    g = Svg(W, H)
    header(g, "Why the requantization is bound by the unpack of the accumulator", "Zoom into REQUANT of layer 0 (measured thread zones) and the reason the unpack and the SFPU sweep do not overlap")
    # locate the layer-0 REQUANT zone on T1 and the QP zones inside it (fine run)
    t1 = [z for z in zones["TRISC_1"]]
    rq = [z for z in t1 if z[0] == "REQUANT"][0]
    t0_, t1_ = rq[1] - 60, rq[2] + 60
    lanes = [("TRISC_0", "TRISC0 unpack"), ("TRISC_1", "TRISC1 math"), ("TRISC_2", "TRISC2 pack")]
    lx, lw, y0 = 200, W - 260, 160
    sx = lw / (t1_ - t0_)
    for i, (r, lab) in enumerate(lanes):
        yy = y0 + i * 130
        g.rect(lx, yy, lw, 110, fill="#fff", stroke="#bbb", sw=0.8, rx=2)
        g.text(30, yy + 60, lab, size=16)
        for n, s_, e, d_ in sorted(zones[r], key=lambda z: z[3]):
            if e < t0_ or s_ > t1_ or n.endswith("-KERNEL") or n.endswith("-FW"):
                continue
            col = ZONE_COLOR.get(n) or COARSE.get(n)
            if not col:
                continue
            inset = 0 if n in COARSE else 14
            xx = lx + (max(s_, t0_) - t0_) * sx
            ww = (min(e, t1_) - max(s_, t0_)) * sx
            g.rect(xx, yy + inset, max(ww, 1.5), 110 - 2 * inset, fill=col, stroke="#fff", sw=0.6, rx=0)
            if n in ZONE_COLOR and ww > 60:
                g.text(xx + 4, yy + 36, n, size=12)
                g.text(xx + 4, yy + 54, f"{int(e - s_)} cyc", size=11, fill="#333")
    for k in range(int(t0_ // 100 * 100), int(t1_) + 1, 100):
        if k >= t0_:
            g.line(lx + (k - t0_) * sx, y0 + 3 * 130 - 12, lx + (k - t0_) * sx, y0 + 3 * 130 - 4, stroke="#777", sw=1)
            g.text(lx + (k - t0_) * sx, y0 + 3 * 130 + 8, f"{k}", size=10, anchor="middle", fill="#666")
    y = y0 + 3 * 130 + 50
    # per-tile numbers from this very zone set
    def zone(r, n, k):
        zz = [z for z in zones[r] if z[0] == n]
        return zz[k]
    kq = 1 if pre else 2
    cp = zone("TRISC_1", "QP_COPY", kq); sf = zone("TRISC_1", "QP_SFPU", kq); pk = zone("TRISC_2", "QP_PACK", kq)
    txt = [
        f"Measured for the 2 accumulator tiles of layer 0: the math thread spends {cp[2] - cp[1]} cycles in QP_COPY (waiting for the unpacker to write the two 4 KB tiles into DST: about {(cp[2] - cp[1]) // 2} cycles per tile), then {sf[2] - sf[1]} cycles in QP_SFPU (the SFPU pipeline: about {(sf[2] - sf[1]) // 2} cycles per tile including the macro setup), then the pack thread needs {pk[2] - pk[1]} cycles. They follow each other: about {(cp[2] - cp[1] + sf[2] - sf[1]) // 2} cycles per tile in total.",
        "Why not overlapped? The accumulator is 32-bit data, so the unpacker writes it STRAIGHT into DST (unpack-to-DST), bypassing SrcA/SrcB. tt-metal makes the math thread wait until the SFPU is idle before the unpacker may write DST, and the SFPU may not read the tile until the unpacker is done (a semaphore handshake per copy). The SFPU sweep of tile 0 therefore cannot start before tile 0 is complete, and the unpack of the next block cannot start while the SFPU works on the half it uses.",
        "This is what 'what is left is the unpack of the 4 KB accumulator tile' means: after the SFPU sweep was shortened from 11 - 13 instructions to 3 issue slots per row group (about 100 cycles per tile), the unpack of the 4 KB int32 tile (about 110 - 130 cycles per tile, one tile at a time) is as large as the sweep itself, and the two are serialised.",
        "The same holds for the first pass over bf16 data (x and the weights): there the unpack goes through SrcA and the datacopy, which the SFPU can overlap, but the datacopy and the SFPU absmax sweep are separated by the barrier of the SFPU start (the SFPU waits until the FPU is idle).",
    ]
    for t_ in txt:
        y = para(g, 30, y, "• " + t_, 215, size=14) + 6
    out.append(save(d, "page_b2.svg", g))
    return out
