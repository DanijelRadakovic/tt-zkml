#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Zone cycles from a device-profiler log: analyze.py <profile_log_device.csv>
Prints `QUANT_X QUANT_W MATMUL REQUANT` as cycles (median over the runs after the first, slowest of the three compute RISCs)."""
import csv, statistics, sys
from collections import defaultdict

ZONES = ("QUANT_X", "QUANT_W", "MATMUL", "REQUANT", "QX_P1", "QX_P2")
start, runs = {}, defaultdict(list)
with open(sys.argv[1]) as f:
    f.readline()
    for r in csv.DictReader(f, skipinitialspace=True):
        name = r["zone name"]
        if name not in ZONES:
            continue
        key = (r["core_x"], r["core_y"], r["RISC processor type"], name)
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            start[key] = t
        else:
            runs[(r["RISC processor type"], name)].append(t - start[key])
out = []
for name in ZONES:
    per_risc = [statistics.median(runs[(risc, name)][1:]) for risc in ("TRISC_0", "TRISC_1", "TRISC_2") if runs[(risc, name)]]
    out.append(int(max(per_risc)))
print(*out)
