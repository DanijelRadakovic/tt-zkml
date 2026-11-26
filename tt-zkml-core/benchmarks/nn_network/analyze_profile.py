#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Device-profiler logs of `zkml_nn_network_bench device|layers` (TT_METAL_DEVICE_PROFILER=1) -> where the time of one network goes.

usage: analyze_profile.py device.csv layers.csv L

Per run (the first two runs are warm-up and dropped, medians of the rest): kernel span (first TRISC start .. last end, all RISCs),
per-stage compute time (zones of TRISC_1: QUANT_X, QUANT_W, MATMUL, REQUANT, summed over the layers), the gaps inside the span that
no zone covers, and for the layer loop the gap between consecutive programs (dispatch).
"""
import csv, re, statistics as st, sys
from collections import defaultdict

def load(path):
    with open(path) as f:
        mhz = float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", f.readline()).group(1))
        rows = [(r["RISC processor type"], r["zone name"], r["type"], int(r["time[cycles since reset]"])) for r in csv.DictReader(f, skipinitialspace=True)]
    return mhz, rows

def runs(rows):
    """-> list of runs; run = {risc: [(name, start, end)]}"""
    stacks, nrun, out = defaultdict(list), defaultdict(lambda: -1), defaultdict(lambda: defaultdict(list))
    for risc, name, kind, t in rows:
        if kind == "ZONE_START":
            if name.endswith("-KERNEL"):
                nrun[risc] += 1
            stacks[risc].append((name, t, nrun[risc]))
        elif kind == "ZONE_END":
            n, t0, r = stacks[risc].pop()
            out[r][risc].append((n, t0, t))
    return [out[i] for i in sorted(out) if i >= 0]

def summarize(path, per_net):
    mhz, rows = load(path)
    rs = runs(rows)
    us = lambda c: c / mhz
    progs = []
    for r in rs:
        k = [(s, e) for risc, z in r.items() for n, s, e in z if n.endswith("-KERNEL")]
        t1 = {n: 0 for n in ("QUANT_X", "QUANT_W", "MATMUL", "REQUANT")}
        for n, s, e in r.get("TRISC_1", []):
            if n in t1:
                t1[n] += e - s
        kern = {risc: ks[0] for risc, z in r.items() if (ks := [(s, e) for n, s, e in z if n.endswith("-KERNEL")])}
        progs.append((min(s for s, _ in k), max(e for _, e in k), t1, kern))
    progs.sort(key=lambda p: p[0])
    nets = [progs[i:i + per_net] for i in range(0, len(progs), per_net)][2:]  # drop 2 warm-up networks
    res = defaultdict(list)
    for net in nets:
        res["span_us"].append(us(net[-1][1] - net[0][0]))
        res["kernel_sum_us"].append(sum(us(p[1] - p[0]) for p in net))
        res["gaps_us"].append(sum(us(net[i + 1][0] - net[i][1]) for i in range(len(net) - 1)))
        for n in ("QUANT_X", "QUANT_W", "MATMUL", "REQUANT"):
            res[n].append(sum(us(p[2][n]) for p in net))
        # data movement kernels busy time (their own kernel zones)
        for risc in ("NCRISC", "BRISC"):
            res[risc + "_kernel_us"].append(sum(us(p[3][risc][1] - p[3][risc][0]) for p in net if risc in p[3]))
    return {k: st.median(v) for k, v in res.items()}

dev, lay, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
for name, path, per in (("fused execute_network", dev, 1), (f"{L} x execute_layer", lay, L)):
    r = summarize(path, per)
    print(f"\n{name}")
    for k, v in r.items():
        print(f"  {k:16s} {v:8.1f} us")
