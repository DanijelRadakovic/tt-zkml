#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Build results/summary.md from cycles.csv, wall.txt and power.txt (see README.md).

usage: summarize.py <results dir>
"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def mean_sd(v):
    return statistics.mean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)


def pct(new, old):
    return 100.0 * (new - old) / old


def cycles_table(results):
    rows = list(csv.DictReader(open(results / "cycles.csv")))
    by = defaultdict(dict)
    for r in rows:
        by[(r["in"], int(r["bits"]), r["out"], int(r["tiles"]))][r["unit"]] = r
    out = ["| input | bits | out | tiles | SFPU cycles | FPU cycles | FPU vs SFPU | SFPU µs | FPU µs | pass 1 cyc/tile (both) | pass 2 cyc/tile SFPU → FPU |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (dt, bits, o, t), d in sorted(by.items()):
        s, f = d["sfpu"], d["fpu"]
        sc, fc = float(s["kernel_cycles"]), float(f["kernel_cycles"])
        out.append(
            f"| {dt} | {bits} | {o} | {t} | {sc:,.0f} | {fc:,.0f} | {pct(fc, sc):+.1f}% | {float(s['kernel_us']):.1f} | "
            f"{float(f['kernel_us']):.1f} | {float(s['pass1_cycles_per_tile']):.0f} | "
            f"{float(s['pass2_cycles_per_tile']):.0f} → {float(f['pass2_cycles_per_tile']):.0f} |"
        )
    return "\n".join(out)


def wall_table(results):
    rx = re.compile(r"wall in (\w+) unit (\w+) bits (\d+) out (\w+) tiles (\d+) .*host_latency_us_median ([\d.]+) min ([\d.]+) stream_us_per_run ([\d.]+)")
    by = defaultdict(dict)
    for line in open(results / "wall.txt"):
        m = rx.search(line)
        if m:
            dt, unit, bits, o, t, lat, _, stream = m.groups()
            by[(dt, int(bits), o, int(t))][unit] = (float(lat), float(stream))
    out = ["| input | bits | out | tiles | SFPU latency µs | FPU latency µs | FPU vs SFPU | SFPU stream µs/run | FPU stream µs/run | FPU vs SFPU |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (dt, bits, o, t), d in sorted(by.items()):
        (sl, ss), (fl, fs) = d["sfpu"], d["fpu"]
        out.append(f"| {dt} | {bits} | {o} | {t} | {sl:.1f} | {fl:.1f} | {pct(fl, sl):+.1f}% | {ss:.1f} | {fs:.1f} | {pct(fs, ss):+.1f}% |")
    return "\n".join(out)


def power_table(results):
    rx = re.compile(
        r"round (\d+) \| power in (\w+) unit (\w+) bits (\d+) out (\w+) tiles (\d+) cores (\d+) seconds ([\d.]+) \| wrong (\d+) (\d+) \| "
        r"runs (\d+) us_per_run ([\d.]+) \| idle_W mean ([\d.]+) sd ([\d.]+) n (\d+) \| load_W mean ([\d.]+) sd ([\d.]+) n (\d+) \| "
        r"delta_W (-?[\d.]+) \| energy_total_mJ_per_core_run ([\d.]+) energy_dynamic_mJ_per_core_run (-?[\d.]+)"
    )
    data = defaultdict(lambda: defaultdict(list))
    cores = tiles = None
    for line in open(results / "power.txt"):
        m = rx.search(line)
        if not m:
            continue
        g = m.groups()
        dt, unit, bits, o = g[1], g[2], int(g[3]), g[4]
        tiles, cores = int(g[5]), int(g[6])
        assert g[8] == "0" and g[9] == "0", f"wrong output in {line}"
        d = data[(dt, bits, o)][unit]
        d.append({"us": float(g[11]), "idle": float(g[12]), "load": float(g[15]), "dw": float(g[18]), "et": float(g[19]), "ed": float(g[20])})
    head = ("| input | bits | out | unit | rounds | µs / run | idle W | load W | Δ W (load − idle) | energy total mJ / core-run | energy dynamic mJ / core-run |")
    out = [head, "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (dt, bits, o), units in sorted(data.items()):
        for unit in ("sfpu", "fpu"):
            d = units[unit]
            cell = lambda k, p=2: "{:.{p}f} ± {:.{p}f}".format(*mean_sd([x[k] for x in d]), p=p)
            out.append(f"| {dt} | {bits} | {o} | {unit.upper()} | {len(d)} | {cell('us', 1)} | {cell('idle', 1)} | {cell('load', 1)} | {cell('dw', 1)} | {cell('et', 3)} | {cell('ed', 3)} |")
        s, f = units["sfpu"], units["fpu"]
        ch = lambda k: pct(statistics.mean(x[k] for x in f), statistics.mean(x[k] for x in s))
        out.append(f"| {dt} | {bits} | {o} | **FPU vs SFPU** | | {ch('us'):+.1f}% | | | {ch('dw'):+.1f}% | {ch('et'):+.1f}% | {ch('ed'):+.1f}% |")
    return "\n".join(out), cores, tiles


LEGEND = '**Columns.**\n- `input`: element type of the tensor read from DRAM (BFLOAT16, 2 bytes per element, on both units).\n- `bits`: the `bit_width` of the operation = the width of the quantized integers. `Q = 2^(bits-1) - 1` (127, 32767,\n  2147483647 for 8, 16, 32). The scaler `2^k` is the largest power of two with `absmax * 2^k <= Q`, so every output\n  value lies in `[-Q, Q]`. It is neither the input size nor the output container size.\n- `out`: dtype of the quantized output tensor. INT32 (4 bytes per element) works for any `bits`; INT8 (1 byte per\n  element, so a 1 KB output tile instead of 4 KB) only for `bits = 8`. For example `bits 32, out int32` quantizes to\n  +-2147483647, `bits 8, out int32` quantizes to +-127 but stores it in 4-byte integers.\n- `tiles`: tensor size in 32x32 tiles (1024 elements each).\n- unit (SFPU / FPU): which math unit multiplies by `2^k` in pass 2. The absmax of pass 1 is on the SFPU in both.\n- cycles: device cycles at 1350 MHz; `pass 1` / `pass 2` are per tile.\n'


def main(results):
    results = Path(results)
    ptable, cores, tiles = power_table(results)
    text = (
        "**Input tile type: every row is BFLOAT16 input on BOTH units** (the SFPU column is the SFPU processing the same "
        "bf16 tiles, not fp32). No FLOAT32 rows were run: the FPU cannot take a FLOAT32 input, so there is no FPU "
        "counterpart to compare with.\n\n" + LEGEND + "\n"
        "## Device cycles (one core, device profiler, median of 10 runs)\n\n" + cycles_table(results)
        + "\n\n## Host wall time (one core, no profiler)\n\n" + wall_table(results)
        + f"\n\n## Power and energy ({cores} cores at once, {tiles} tiles per core, mean ± sd over rounds)\n\n" + ptable + "\n"
    )
    (results / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
