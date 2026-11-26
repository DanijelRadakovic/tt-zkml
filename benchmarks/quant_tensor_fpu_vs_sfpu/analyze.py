#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Reduce device-profiler logs of the SFPU-vs-FPU benchmark to one row per configuration.

usage: analyze.py <results dir>         reads <dir>/raw/prof_*.csv, writes <dir>/cycles.csv, prints markdown

A log is profile_log_device.csv of one `bench profile` process: the program ran N times back to back on
core (0,0). Every row is a zone boundary (RISC, cycle counter, zone name, START|END). The i-th *-KERNEL start
on a RISC belongs to run i. Run 0 (JIT, cold caches) is dropped and the others are reduced with the median.
All RISCs of a core read the same counter, so zones of different RISCs can be compared directly.

Metrics per run (cycles; the chip runs at CHIP_FREQ MHz, 1350 on this card):
  kernel     earliest *-KERNEL start to latest *-KERNEL end on the core (the whole program)
  pass1      PASS1 zone on TRISC_1 (math): copy_tile + SFPU absmax over the tensor
  scaler     SCALER zone: reduce the 32 lanes, derive 2^k, pack it
  pass2      PASS2 zone: multiply, round, clamp, convert, pack. Includes time compute waits for c_16 space
             (the writer) and for c_0 tiles (the reader)
  reader     NCRISC-KERNEL: the reader RISC, from start to end
  writer     BRISC-KERNEL: the writer RISC, from start to end
"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    with open(path) as f:
        header = f.readline()
        freq = float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", header).group(1))
        rows = []
        for r in csv.DictReader(f, skipinitialspace=True):
            rows.append((r["RISC processor type"], r["zone name"], r["type"], int(r["time[cycles since reset]"])))
    return freq, rows


def zones_by_run(rows):
    stacks = defaultdict(list)
    run_of = defaultdict(lambda: -1)
    out = defaultdict(list)
    for risc, name, kind, t in rows:
        if kind == "ZONE_START":
            if name.endswith("-KERNEL"):
                run_of[risc] += 1
            stacks[risc].append((name, t, run_of[risc]))
        elif kind == "ZONE_END":
            n, t0, run = stacks[risc].pop()
            assert n == name, (risc, n, name)
            out[run].append((risc, name, t0, t))
    return out


def per_run(zones):
    def one(risc, name):
        z = [(s, e) for r, n, s, e in zones if r == risc and n == name]
        return z[0]

    kernels = [(s, e) for r, n, s, e in zones if n.endswith("-KERNEL")]
    m = {"kernel": max(e for _, e in kernels) - min(s for s, _ in kernels)}
    for key, (risc, name) in {"reader": ("NCRISC", "NCRISC-KERNEL"), "writer": ("BRISC", "BRISC-KERNEL")}.items():
        s, e = one(risc, name)
        m[key] = e - s
    for key, name in {"pass1": "PASS1", "scaler": "SCALER", "pass2": "PASS2"}.items():
        s, e = one("TRISC_1", name)
        m[key] = e - s
    return m


def summarize(path):
    freq, rows = load(path)
    runs = zones_by_run(rows)
    per = [per_run(runs[i]) for i in sorted(runs) if i > 0]
    med = {k: statistics.median(p[k] for p in per) for k in per[0]}
    spread = (max(p["kernel"] for p in per) - min(p["kernel"] for p in per)) / med["kernel"]
    return freq, len(per), med, spread


def main(results):
    results = Path(results)
    rows = []
    for p in sorted((results / "raw").glob("prof_*.csv")):
        # prof_<in>_<unit>_<bits>b_<out>_<tiles>t.csv
        _, dt, unit, bits, out, tiles = p.stem.split("_")
        freq, n, m, spread = summarize(p)
        t = int(tiles[:-1])
        rows.append(
            {
                "in": dt, "unit": unit, "bits": int(bits[:-1]), "out": out, "tiles": t, "runs": n,
                "freq_mhz": freq, "spread_pct": 100 * spread,
                **{f"{k}_cycles": m[k] for k in m},
                **{f"{k}_us": m[k] / freq for k in m},
                "kernel_cycles_per_tile": m["kernel"] / t,
                "pass1_cycles_per_tile": m["pass1"] / t,
                "pass2_cycles_per_tile": m["pass2"] / t,
            }
        )
    with open(results / "cycles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(dict)
    for r in rows:
        by[(r["in"], r["bits"], r["out"], r["tiles"])][r["unit"]] = r
    print("| in | bits | out | tiles | SFPU cycles | FPU cycles | gain | SFPU µs | FPU µs | SFPU pass1 / pass2 / writer cyc/tile | FPU pass1 / pass2 / writer cyc/tile |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(by):
        s, f = by[k].get("sfpu"), by[k].get("fpu")
        if not (s and f):
            continue
        gain = 100 * (s["kernel_cycles"] - f["kernel_cycles"]) / s["kernel_cycles"]
        t = k[3]
        print(
            f"| {k[0]} | {k[1]} | {k[2]} | {t} | {s['kernel_cycles']:.0f} | {f['kernel_cycles']:.0f} | {gain:+.1f}% | "
            f"{s['kernel_us']:.1f} | {f['kernel_us']:.1f} | "
            f"{s['pass1_cycles']/t:.0f} / {s['pass2_cycles']/t:.0f} / {s['writer_cycles']/t:.0f} | "
            f"{f['pass1_cycles']/t:.0f} / {f['pass2_cycles']/t:.0f} / {f['writer_cycles']/t:.0f} |"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
