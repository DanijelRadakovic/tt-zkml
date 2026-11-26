#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Device-profiler logs (results/raw/prof_<input>_<variant>_<mode>_<tiles>t_r<reps>.csv) -> results/cycles.csv, cycles.md.

Each log has 11 runs of the LOOP zone on the three TRISCs (unpack, math, pack). The first run is dropped (cold caches), the
median of the other 10 is divided by tiles * reps. Usage: analyze.py <results dir>"""
import csv, glob, os, re, statistics, sys

out = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out, "raw", "prof_*.csv"))):
    m = re.match(r"prof_(\w+?)_(twos_lib|twos_own|sm_int8pack|sm_uint8pack)_(full|sfpu)_(\d+)t_r(\d+)\.csv", os.path.basename(path))
    inp, variant, mode, tiles, reps = m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5))
    with open(path) as f:
        freq = float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", f.readline()).group(1))
        start, durs = {}, {}
        for r in csv.DictReader(f, skipinitialspace=True):
            if r["zone name"] != "LOOP":
                continue
            key = (r["core_x"], r["core_y"], r["RISC processor type"])
            t = int(r["time[cycles since reset]"])
            if r["type"] == "ZONE_START":
                start[key] = t
            else:
                durs.setdefault(key, []).append(t - start[key])
    per = {}
    for (_, _, risc), d in durs.items():
        per[risc] = statistics.median(d[1:]) / (tiles * reps)
    rows.append(dict(input=inp, mode=mode, tiles=tiles, variant=variant, reps=reps, mhz=freq,
                     unpack=per["TRISC_0"], math=per["TRISC_1"], pack=per["TRISC_2"],
                     slowest=max(per.values())))

with open(os.path.join(out, "cycles.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["input", "mode", "tiles", "variant", "cycles_unpack", "cycles_math", "cycles_pack", "cycles_slowest", "ns_slowest"])
    for r in rows:
        w.writerow([r["input"], r["mode"], r["tiles"], r["variant"], f"{r['unpack']:.1f}", f"{r['math']:.1f}",
                    f"{r['pack']:.1f}", f"{r['slowest']:.1f}", f"{r['slowest'] / r['mhz'] * 1000:.1f}"])

lines = ["# Device cycles per tile (one core, device profiler)", "",
         "Median of 10 runs of the LOOP zone divided by tiles * reps; 1 tile = 1024 elements, clock 1350 MHz. `full` = copy + quantize + pack of "
         "blocks of 3 tiles, `sfpu` = only the quantize calls. The unpack, math and pack columns are the three TRISC threads; the "
         "slowest of them bounds the loop. `vs twos_lib` is the change of the slowest thread.", ""]
for inp in ("fp32", "bf16"):
    for mode in ("full", "sfpu"):
        for tiles in sorted({r["tiles"] for r in rows}):
            sel = [r for r in rows if r["input"] == inp and r["mode"] == mode and r["tiles"] == tiles]
            if not sel:
                continue
            base = next((r["slowest"] for r in sel if r["variant"] == "twos_lib"), None)
            lines += [f"## input {inp}, {mode}, {tiles} tiles", "",
                      "| variant | unpack | math | pack | slowest (cycles) | slowest (ns) | vs twos_lib |", "|---|---|---|---|---|---|---|"]
            for r in sel:
                d = f"{(r['slowest'] / base - 1) * 100:+.1f}%" if base else ""
                lines.append(f"| {r['variant']} | {r['unpack']:.1f} | {r['math']:.1f} | {r['pack']:.1f} | {r['slowest']:.1f} | "
                             f"{r['slowest'] / r['mhz'] * 1000:.1f} | {d} |")
            lines.append("")
with open(os.path.join(out, "cycles.md"), "w") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
