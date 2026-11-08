#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Turn device-profiler logs (profile_log_device.csv) into one row per benchmark configuration.

usage: analyze.py results/raw/*.csv          -> prints a markdown table, writes results/summary.csv

How a log is read (see README.md, "Reading the profiler log"):
  * every row is one zone boundary: (core, RISC, cycle counter, zone name, ZONE_START | ZONE_END)
  * the program ran N times back to back; the i-th TRISC-KERNEL start on a RISC belongs to run i
  * run 0 is dropped (JIT / cold caches); the rest are reduced with the median
  * all RISCs of a core read the same wall-clock counter, so their zones can be compared directly
"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    """-> (freq_mhz, list of (risc, zone, kind, cycle))"""
    with open(path) as f:
        header = f.readline()
        freq = float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", header).group(1))
        rows = []
        for r in csv.DictReader(f, skipinitialspace=True):
            rows.append((r["RISC processor type"], r["zone name"], r["type"], int(r["time[cycles since reset]"])))
    return freq, rows


def zones_by_run(rows):
    """Pair START/END per RISC (zones nest, so a stack) and number the runs by *-KERNEL starts.
    -> {run: [(risc, name, start, end)]}"""
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
    """Metrics of one run, in cycles."""
    def pick(risc, name):
        return [(s, e) for r, n, s, e in zones if r == risc and n == name]

    kernels = [(s, e) for r, n, s, e in zones if n.endswith("-KERNEL")]
    m = {"kernel": max(e for _, e in kernels) - min(s for s, _ in kernels)}
    for r in ("NCRISC", "BRISC"):  # reader / writer data-movement kernels
        k = pick(r, f"{r}-KERNEL")
        if k:
            m[f"{r.lower()}_kernel"] = k[0][1] - k[0][0]
    p1, sc, p2 = (pick("TRISC_1", n)[0] for n in ("PASS1", "SCALER", "PASS2"))
    m["pass1"], m["scaler"], m["pass2"] = p1[1] - p1[0], sc[1] - sc[0], p2[1] - p2[0]
    m["compute_start_to_end"] = p2[1] - p1[0]
    waits = pick("TRISC_0", "WAIT_C0")
    m["wait_p1"] = sum(e - s for s, e in waits if p1[0] <= s < p1[1])
    m["wait_p2"] = sum(e - s for s, e in waits if p2[0] <= s < p2[1])
    m["n_waits_p1"] = sum(1 for s, _ in waits if p1[0] <= s < p1[1])
    m["n_waits_p2"] = sum(1 for s, _ in waits if p2[0] <= s < p2[1])
    reads = pick("NCRISC", "READ_BLOCK")
    m["read_total"] = sum(e - s for s, e in reads)
    m["read_blocks"] = len(reads)
    m["read_span"] = max(e for _, e in reads) - min(s for s, _ in reads)
    w = pick("BRISC", "WRITE_ALL")
    m["write_all"] = w[0][1] - w[0][0]
    return m


def summarize(path):
    freq, rows = load(path)
    runs = zones_by_run(rows)
    per = [per_run(runs[i]) for i in sorted(runs) if i > 0]
    keys = per[0].keys()
    med = {k: statistics.median(p[k] for p in per) for k in keys}
    spread = (max(p["kernel"] for p in per) - min(p["kernel"] for p in per)) / med["kernel"]
    return freq, len(per), med, spread


def main(paths):
    out_rows = []
    for p in sorted(paths):
        # file name: <mode>_<in>_<bits>b_<tiles>t_<block>blk.csv
        mode, dt, bits, tiles, blk = Path(p).stem.split("_")
        freq, n, m, spread = summarize(p)
        us = lambda c: c / freq
        out_rows.append(
            {
                "mode": mode, "in": dt, "bits": int(bits[:-1]), "tiles": int(tiles[:-1]), "block": int(blk[:-3]),
                "runs": n, "freq_mhz": freq, "kernel_us": us(m["kernel"]), "spread_pct": 100 * spread,
                "pass1_us": us(m["pass1"]), "scaler_us": us(m["scaler"]), "pass2_us": us(m["pass2"]),
                "wait_p1_us": us(m["wait_p1"]), "wait_p2_us": us(m["wait_p2"]),
                "read_total_us": us(m["read_total"]), "read_blocks": int(m["read_blocks"]),
                "write_all_us": us(m["write_all"]),
            }
        )
    Path("results").mkdir(exist_ok=True)
    with open("results/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # Side-by-side: stream vs resident for the same (in, bits, tiles, block)
    by = defaultdict(dict)
    for r in out_rows:
        by[(r["in"], r["bits"], r["tiles"], r["block"])][r["mode"]] = r
    print("| in | bits | tiles | block | stream µs | resident µs | gain | stream wait p1/p2 µs | resident wait p1/p2 µs | stream read µs | resident read µs |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(by):
        s, r = by[k].get("stream"), by[k].get("resident")
        if not (s and r):
            continue
        gain = 100 * (s["kernel_us"] - r["kernel_us"]) / s["kernel_us"]
        print(
            f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {s['kernel_us']:.1f} | {r['kernel_us']:.1f} | {gain:+.1f}% | "
            f"{s['wait_p1_us']:.1f} / {s['wait_p2_us']:.1f} | {r['wait_p1_us']:.1f} / {r['wait_p2_us']:.1f} | "
            f"{s['read_total_us']:.1f} | {r['read_total_us']:.1f} |"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
