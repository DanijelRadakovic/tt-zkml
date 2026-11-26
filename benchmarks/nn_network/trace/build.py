# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Builds docs/32x64l8.pdf and docs/32x64l8-pre-quant.pdf: one page per step of the execution of execute_network on one Tensix core.

usage: python3 build.py <out_dir_for_svg_and_typst> [nonq|q|both]
"""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
from tracelib import load_run, seq
import model, draw, pages

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "trace")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

VARIANTS = {
    "nonq": dict(pre=False, fine="device_32x64_L2_fine.csv", coarse="device_32x64_L8_coarse.csv", pdf="32x64l8.pdf",
                 name="execute_network: weights quantized inside the kernel", short="weights quantized in the kernel"),
    "q": dict(pre=True, fine="deviceq_32x64_L2_fine.csv", coarse="deviceq_32x64_L8_coarse.csv", pdf="32x64l8-pre-quant.pdf",
              name="execute_network_quantized: weights quantized beforehand", short="pre-quantized weights"),
}


def build_variant(key, outdir):
    v = VARIANTS[key]
    mhz, zones, t0 = load_run(os.path.join(RES, v["fine"]))
    cmhz, czones, _ = load_run(os.path.join(RES, v["coarse"]))
    tmax = max(e for r in zones for n, s, e, d in zones[r] if n.endswith("-KERNEL"))

    def Z(risc, name, k):
        return seq(zones, risc, name)[k]

    S = model.build(v["pre"], Z, zones)
    model.build_last(S, Z)
    frames = S.frames
    last = 0
    for f in frames:  # the cursor never goes back
        f["t"] = max(f["t"], last)
        last = f["t"]
    d = os.path.join(outdir, key)
    os.makedirs(d, exist_ok=True)
    ctx = dict(zones=zones, mhz=mhz, tmax=tmax, name=v["name"])
    svgs = []
    for i, fr in enumerate(frames):
        svg = draw.draw_frame(fr, i, len(frames), ctx, frames[i - 1] if i else None)
        p = os.path.join(d, f"frame_{i + 1:02d}.svg")
        open(p, "w").write(svg)
        svgs.append(os.path.basename(p))
    front = pages.front_pages(key, v, frames, zones, mhz, tmax, czones, cmhz, d)
    back = pages.back_pages(key, v, frames, zones, mhz, tmax, czones, cmhz, d)
    typ = os.path.join(d, "doc.typ")
    with open(typ, "w") as f:
        f.write('#set page(width: 420mm, height: 297mm, margin: 0pt)\n#set text(font: "DejaVu Sans")\n')
        for s in front + svgs + back:
            f.write(f'#image("{s}", width: 100%)\n#pagebreak(weak: true)\n')
    pdf = os.path.join(REPO, "docs", v["pdf"])
    subprocess.run(["typst", "compile", "--root", d, typ, pdf], check=True)
    print("wrote", pdf, len(frames), "steps")
    return frames


if __name__ == "__main__":
    out = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    for k in (["nonq", "q"] if which == "both" else [which]):
        build_variant(k, out)
