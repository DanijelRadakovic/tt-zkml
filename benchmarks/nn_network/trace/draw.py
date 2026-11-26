# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Draws one frame (the state of one Tensix core) as an SVG."""
import textwrap
from svg import Svg

W, H = 1800, 1273
C_RUN, C_WAIT, C_IDLE = "#8fd19e", "#f7d56b", "#e3e3e3"
STATE_COLOR = {"running": C_RUN, "waiting": C_WAIT, "idle": C_IDLE}
CHANGED = "#e8590c"
KIND_COLOR = {"x": "#a5d8ff", "w": "#d0bfff", "q": "#96f2d7", "a": "#b2f2bb", "b": "#8ce99a", "2": "#ffec99", "y": "#ffd8a8", "o": "#ffd8a8"}


def wrap(s, n):
    return textwrap.wrap(s, n) or [""]


def cell_color(label):
    if label.startswith("acc"):
        return "#ffc9c9"
    if label.startswith("2^"):
        return "#ffec99"
    if label.startswith("w'") or label.startswith("q'"):
        return "#e5dbff" if label.startswith("w'") else "#c3fae8"
    return KIND_COLOR.get(label[0], "#dee2e6")


ZONE_COLOR = {
    "SP_COPY": "#74c0fc", "QP_COPY": "#74c0fc", "SP_SFPU": "#ffa94d", "QP_SFPU": "#ffa94d", "MM_SWEEP": "#ffa94d",
    "SP_SCALER": "#ffe066", "RQ_SCALER": "#ffe066", "QP_PACK": "#69db7c", "MM_PACK": "#69db7c", "MM_FPU": "#b197fc",
    "READ_W": "#63e6be", "WAIT_DONE": "#ced4da", "WRITE": "#51cf66", "WAIT_W": "#ff8787", "WAIT_ACC": "#ff8787",
}
COARSE = {"QUANT_X": "#e7f5ff", "QUANT_W": "#f3f0ff", "QW_SCALE": "#eee8fc", "QW_QUANT": "#e5dbff", "MATMUL": "#fff4e6", "REQUANT": "#ebfbee"}


def innermost(zones, risc, t):
    best = None
    for n, s, e, d in zones[risc]:
        if n.endswith("-KERNEL") or n.endswith("-FW"):
            continue
        if s <= t <= e and (best is None or d >= best[3]):
            best = (n, s, e, d)
    return best


def draw_timeline(g, zones, mhz, t, x0, y0, w, h, tmax):
    lanes = [("NCRISC", "NCRISC reader"), ("BRISC", "BRISC writer"), ("TRISC_0", "TRISC0 unpack"), ("TRISC_1", "TRISC1 math"), ("TRISC_2", "TRISC2 pack")]
    lh = (h - 40) / len(lanes)
    lx = x0 + 130
    sx = (w - 140) / tmax
    g.text(x0, y0 + 14, "Profiler timeline of the whole run (measured; cursor = this step)", size=15, weight="bold")
    for i, (risc, label) in enumerate(lanes):
        y = y0 + 26 + i * lh
        g.rect(lx, y, w - 140, lh - 5, fill="#fff", stroke="#bbb", sw=0.8, rx=2)
        g.text(x0, y + lh / 2 + 4, label, size=12)
        zs = sorted([z for z in zones[risc] if not z[0].endswith("-KERNEL") and not z[0].endswith("-FW")], key=lambda z: z[3])
        for n, s, e, d in zs:
            col = ZONE_COLOR.get(n) or COARSE.get(n)
            if not col:
                continue
            inset = 0 if n in COARSE else 4
            g.rect(lx + s * sx, y + inset, max((e - s) * sx, 1.2), lh - 5 - 2 * inset, fill=col, stroke="none", sw=0, rx=0)
    cx = lx + t * sx
    g.line(cx, y0 + 20, cx, y0 + 26 + len(lanes) * lh, stroke="#d6336c", sw=2.2)
    g.text(cx + 4, y0 + 18, f"t = {t} cycles = {t / mhz:.2f} us", size=12, fill="#d6336c", weight="bold")
    # time axis ticks
    for k in range(0, int(tmax) + 1, 2000):
        g.line(lx + k * sx, y0 + 26 + len(lanes) * lh - 3, lx + k * sx, y0 + 26 + len(lanes) * lh + 3, stroke="#777", sw=1)
        g.text(lx + k * sx, y0 + 26 + len(lanes) * lh + 16, f"{k}", size=10, anchor="middle", fill="#666")


def legend(g, x, y):
    items = [("copy (unpack -> DST)", "#74c0fc"), ("SFPU sweep", "#ffa94d"), ("scaler", "#ffe066"), ("pack", "#69db7c"), ("matmul (FPU)", "#b197fc"), ("NoC read", "#63e6be"),
             ("NoC write", "#51cf66"), ("writer waits for compute", "#ced4da"), ("compute waits", "#ff8787")]
    for i, (n, c) in enumerate(items):
        xx = x + (i % 5) * 330 if False else x + i * 195
        g.rect(xx, y, 14, 14, fill=c, stroke="#888", sw=0.6, rx=2)
        g.text(xx + 20, y + 12, n, size=11)


def draw_frame(fr, idx, total, ctx, prev):
    g = Svg(W, H)
    s = fr["state"]
    p = prev["state"] if prev else None
    zones, mhz, tmax = ctx["zones"], ctx["mhz"], ctx["tmax"]

    def changed(key, sub=None):
        if p is None:
            return False
        if sub is not None:
            return s[key][sub] != p[key][sub] if isinstance(s[key], (list, dict)) else s[key] != p[key]
        return s[key] != p[key]

    # ---------------- header
    g.rect(0, 0, W, 96, fill="#1b2430", stroke="none", rx=0)
    g.text(20, 38, f"Step {idx + 1} / {total}   {fr['title']}", size=26, fill="#fff", weight="bold")
    g.text(20, 68, f"{ctx['name']}  |  layer {fr['layer']} (stage: {fr['stage']})  |  t = {fr['t']} cycles = {fr['t'] / mhz:.2f} us after the first kernel start ({mhz / 1000:.2f} GHz)",
           size=15, fill="#d0d7e2")
    g.text(W - 20, 30, "ORANGE outline = changed since the previous step", size=14, fill="#ffa94d", anchor="end", weight="bold")
    g.text(W - 20, 52, "units of the coprocessor: MODELLED from the kernel code", size=12, fill="#d0d7e2", anchor="end")
    g.text(W - 20, 70, "timeline and RISC-V zones: MEASURED (device profiler)", size=12, fill="#d0d7e2", anchor="end")

    # ---------------- L1 panel
    X0, Y0 = 15, 108
    g.rect(X0, Y0, 575, 650, fill="#fff", stroke="#444", sw=1.5)
    g.text(X0 + 12, Y0 + 24, "L1: circular buffers (CB) of this program", size=17, weight="bold")
    g.text(X0 + 12, Y0 + 44, "page = one tile; green = ready (pushed); orange dashed = reserved (being written); empty = free", size=10.5, fill="#555")
    row = 60
    for i, (name, c) in enumerate(s["cb"].items()):
        y = Y0 + 56 + i * row
        ch = p is not None and p["cb"][name] != c
        g.rect(X0 + 6, y - 4, 563, row - 4, fill="#fafafa", stroke=CHANGED if ch else "#ccc", sw=3 if ch else 0.8, rx=3)
        g.text(X0 + 14, y + 14, f"{name}  {c['role'].split(' (')[0]}", size=13, weight="bold")
        g.text(X0 + 14, y + 32, f"{c['fmt']} · {c['pages']} x {c['bytes'] // 1024} KB = {c['pages'] * c['bytes'] // 1024} KB", size=11, fill="#555")
        n = c["pages"]
        cw = 30 if n <= 4 else 26
        for k in range(n):
            cx = X0 + 215 + k * (cw + 4)
            cell = c["cells"][k]
            if cell is None:
                g.rect(cx, y, cw, 36, fill="#fff", stroke="#aaa", sw=0.8, rx=2)
            else:
                kind, lab = cell
                if kind == "R":
                    g.rect(cx, y, cw, 36, fill=cell_color(lab), stroke="#2b8a3e", sw=2.2, rx=2)
                else:
                    g.rect(cx, y, cw, 36, fill="#ffe8cc", stroke=CHANGED, sw=1.6, rx=2, dash="4 2")
                g.text(cx + cw / 2, y + 22, lab if len(lab) <= 5 else lab[:5], size=10 if len(lab) > 3 else 11, anchor="middle")
            if k == c["rd"] and any(cc for cc in c["cells"]):
                g.text(cx + cw / 2, y + 50, "rd", size=9, anchor="middle", fill="#1971c2")
            if k == c["wr"]:
                g.text(cx + cw / 2 + 0, y + 50 if k != c["rd"] else y + 50, "wr" if k != c["rd"] else "rd/wr", size=9, anchor="middle", fill="#c2255c")
        ready = sum(1 for cc in c["cells"] if cc and cc[0] == "R")
        g.text(X0 + 15 + 485 + 40, y + 14, f"{ready} ready", size=11, anchor="end", fill="#2b8a3e" if ready else "#999")

    # ---------------- datapath panel
    Xd = 600
    g.rect(Xd, Y0, 635, 650, fill="#fff", stroke="#444", sw=1.5)
    g.text(Xd + 12, Y0 + 24, "Tensix coprocessor (modelled)", size=17, weight="bold")

    def box(x, y, w, h, title, lines, key=None, sub=None, active=False, fill="#f8f9fa"):
        ch = changed(key, sub) if key else False
        g.rect(x, y, w, h, fill="#d3f9d8" if active else fill, stroke=CHANGED if ch else "#555", sw=3 if ch else 1.3, rx=5)
        g.text(x + 8, y + 17, title, size=13, weight="bold")
        for j, ln in enumerate(lines):
            g.text(x + 8, y + 34 + j * 14, ln, size=11)

    arrows = set(fr["arrows"])

    def arrow(aid, x1, y1, x2, y2, label=""):
        on = aid in arrows
        g.line(x1, y1, x2, y2, stroke="#e8590c" if on else "#bbb", sw=3.2 if on else 1.4, arrow=True)
        if label:
            g.text((x1 + x2) / 2 + 3, (y1 + y2) / 2 - 3, label, size=10, fill="#e8590c" if on else "#999")

    # unpackers
    box(Xd + 12, Y0 + 45, 150, 58, "Unpacker 0", wrap(s["u0"], 24)[:3], key="u0", active=s["u0"] != "idle")
    box(Xd + 12, Y0 + 118, 150, 58, "Unpacker 1", wrap(s["u1"], 24)[:3], key="u1", active=s["u1"] != "idle")

    # SrcA / SrcB with two banks
    def src_box(x, y, title, banks, key):
        ch = changed(key)
        g.rect(x, y, 200, 70, fill="#f8f9fa", stroke=CHANGED if ch else "#555", sw=3 if ch else 1.3, rx=5)
        g.text(x + 8, y + 16, title + " (2 banks, double buffered)", size=12, weight="bold")
        for b in range(2):
            bx = x + 8 + b * 96
            cell = banks[b]
            col = "#fff"
            txt = "free"
            if cell:
                lab, st = cell
                col = "#d0ebff" if st == "writing" else "#fff3bf"
                txt = f"{lab}"
            g.rect(bx, y + 24, 90, 38, fill=col, stroke="#777", sw=1, rx=3)
            g.text(bx + 45, y + 40, f"bank {b}", size=10, anchor="middle", fill="#555")
            g.text(bx + 45, y + 54, txt + (" (unpacker writes)" if cell and cell[1] == "writing" else " (FPU reads)" if cell else ""), size=9, anchor="middle")

    src_box(Xd + 190, Y0 + 45, "SrcA", s["srcA"], "srcA")
    src_box(Xd + 190, Y0 + 118, "SrcB", s["srcB"], "srcB")
    # FPU
    box(Xd + 440, Y0 + 70, 180, 80, "FPU (matrix unit)", wrap(s["fpu"], 30)[:4], key="fpu", active=s["fpu"] not in ("idle",) and not s["fpu"].startswith("idle"))
    arrow("U0-A", Xd + 162, Y0 + 74, Xd + 190, Y0 + 74)
    arrow("U1-B", Xd + 162, Y0 + 147, Xd + 190, Y0 + 147)
    arrow("A-F", Xd + 390, Y0 + 80, Xd + 440, Y0 + 95)
    arrow("B-F", Xd + 390, Y0 + 152, Xd + 440, Y0 + 130)

    # DST
    dy = Y0 + 215
    g.rect(Xd + 190, dy, 430, 150, fill="#fff", stroke=CHANGED if changed("dst") or changed("dst_owner") else "#555", sw=3 if changed("dst") or changed("dst_owner") else 1.3, rx=5)
    g.text(Xd + 198, dy + 17, "DST register file (fp32 mode, half sync: 2 halves x 4 tiles)", size=12, weight="bold")
    for h_ in range(2):
        hx = Xd + 198 + h_ * 212
        own = s["dst_owner"][h_]
        col = {"math": "#d3f9d8", "pack": "#fff3bf", "free": "#f1f3f5"}[own]
        g.rect(hx, dy + 26, 205, 118, fill=col, stroke="#888", sw=1, rx=4)
        g.text(hx + 6, dy + 40, f"half {h_}: {'math (FPU/SFPU) owns' if own == 'math' else 'packer reads' if own == 'pack' else 'free'}", size=10)
        for k in range(4):
            lab = s["dst"][4 * h_ + k]
            g.rect(hx + 6 + k * 49, dy + 48, 45, 90, fill="#fff" if not lab else "#e7f5ff", stroke="#999", sw=0.8, rx=2)
            g.text(hx + 6 + k * 49 + 22, dy + 62, f"slot {k}", size=8, anchor="middle", fill="#777")
            if lab:
                for j, ln in enumerate(wrap(lab, 8)[:6]):
                    g.text(hx + 6 + k * 49 + 22, dy + 78 + j * 11, ln, size=9, anchor="middle")
    arrow("F-D", Xd + 530, Y0 + 150, Xd + 530, dy)
    arrow("U0-D", Xd + 87, Y0 + 176, Xd + 190, dy + 40, "unpack-to-DST")

    # SFPU + lregs
    sy = Y0 + 385
    box(Xd + 12, sy, 170, 92, "SFPU (32 lanes)", wrap(s["sfpu"], 27)[:5], key="sfpu", active=s["sfpu"] != "idle" and not s["sfpu"].startswith("idle"))
    arrow("D-S", Xd + 190, dy + 90, Xd + 182, sy + 40)
    ch_l = changed("lregs")
    g.rect(Xd + 190, sy, 430, 175, fill="#fff", stroke=CHANGED if ch_l else "#555", sw=3 if ch_l else 1.3, rx=5)
    g.text(Xd + 198, sy + 16, "SFPU local registers (LReg0..7; 32 lanes each)", size=12, weight="bold")
    for k in range(8):
        v = s["lregs"][k]
        yy = sy + 24 + k * 18
        pc = p is not None and p["lregs"][k] != v
        g.rect(Xd + 198, yy, 414, 16, fill="#fff3bf" if pc else "#f8f9fa", stroke="#bbb", sw=0.6, rx=2)
        g.text(Xd + 204, yy + 12, f"LReg{k}", size=10, weight="bold")
        g.text(Xd + 260, yy + 12, v, size=10)
    ch_m = changed("macro")
    g.rect(Xd + 12, sy + 100, 170, 75, fill="#f8f9fa", stroke=CHANGED if ch_m else "#555", sw=3 if ch_m else 1.3, rx=5)
    g.text(Xd + 18, sy + 116, "SFPLOADMACRO config", size=11, weight="bold")
    for j, ln in enumerate(wrap(s["macro"], 29)[:4]):
        g.text(Xd + 18, sy + 130 + j * 12, ln, size=9)

    # packer
    py = Y0 + 575
    box(Xd + 190, py, 250, 66, "Packer", wrap(s["packer"], 42)[:2] + wrap("config: " + s["pack_cfg"], 42)[:1], key="packer", active=s["packer"] != "idle" and not s["packer"].startswith("idle"))
    arrow("D-P", Xd + 400, dy + 150, Xd + 330, py)
    g.text(Xd + 450, py + 20, "format conversion,", size=10, fill="#555")
    g.text(Xd + 450, py + 33, "ReLU, L1 accumulate,", size=10, fill="#555")
    g.text(Xd + 450, py + 46, "untilize, row masks", size=10, fill="#555")
    arrow("P-L1", Xd + 190, py + 33, Xd - 10, py + 33, "-> L1")
    arrow("L1-U0", Xd - 10, Y0 + 70, Xd + 12, Y0 + 70, "")
    arrow("L1-U1", Xd - 10, Y0 + 143, Xd + 12, Y0 + 143, "")

    # ---------------- RISC panel
    Xr = 1250
    g.rect(Xr, Y0, 535, 650, fill="#fff", stroke="#444", sw=1.5)
    g.text(Xr + 12, Y0 + 24, "Baby RISC-V cores, NoC and DRAM", size=17, weight="bold")
    noc_lines = s["noc"] or ["(no NoC transfer in flight)"]
    chn = changed("noc")
    g.rect(Xr + 8, Y0 + 36, 519, 92, fill="#e7f5ff" if s["noc"] else "#f8f9fa", stroke=CHANGED if chn else "#555", sw=3 if chn else 1.3, rx=5)
    g.text(Xr + 16, Y0 + 54, "NoC and DRAM (8 banks)", size=13, weight="bold")
    for j, ln in enumerate(noc_lines[:3]):
        g.text(Xr + 16, Y0 + 74 + j * 17, ln[:78], size=11)
    risc_info = [("NCRISC", "NCRISC (reader): DRAM -> L1"), ("BRISC", "BRISC (writer): L1 -> DRAM"), ("TRISC_0", "TRISC0: unpack thread"), ("TRISC_1", "TRISC1: math thread"), ("TRISC_2", "TRISC2: pack thread")]
    for i, (key, label) in enumerate(risc_info):
        y = Y0 + 138 + i * 102
        stt, txt = s["risc"][key]
        chr_ = p is not None and p["risc"][key] != s["risc"][key]
        g.rect(Xr + 8, y, 519, 96, fill=STATE_COLOR[stt], stroke=CHANGED if chr_ else "#555", sw=3 if chr_ else 1.3, rx=5)
        g.text(Xr + 16, y + 18, label, size=13, weight="bold")
        g.text(Xr + 519, y + 18, stt, size=11, anchor="end", fill="#333")
        for j, ln in enumerate(wrap(txt, 76)[:2]):
            g.text(Xr + 16, y + 37 + j * 14, ln, size=11)
        z = innermost(zones, key, fr["t"])
        zn = "between profiler zones" if z is None else f"profiler zone now: {z[0]} ({z[1]} .. {z[2]} cycles, {(z[2] - z[1])} long)"
        g.text(Xr + 16, y + 82, "measured: " + zn, size=10, fill="#1c3d5a", mono=False)

    # ---------------- timeline
    draw_timeline(g, zones, mhz, fr["t"], 15, 770, W - 30, 250, tmax)
    legend(g, 25, 1045)

    # ---------------- what changed
    g.rect(15, 1068, W - 30, 190, fill="#fff", stroke="#444", sw=1.5)
    g.text(27, 1092, "What happens in this step", size=17, weight="bold")
    yy = 1118
    for b in fr["bullets"]:
        ls = wrap(b, 205)
        g.text(27, yy, "•", size=14)
        for j, ln in enumerate(ls):
            g.text(45, yy + j * 18, ln, size=14)
        yy += 18 * len(ls) + 6
    return g.done()
