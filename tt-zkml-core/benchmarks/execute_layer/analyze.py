#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Zone cycles of execute_layer from a device-profiler log: analyze.py <profile_log_device.csv> <Mt> <Kt> <Nt>

The compute kernel has the zones QUANT_X, QUANT_W, MATMUL and REQUANT (on the three compute RISCs). The first run (JIT, cold
caches) is dropped; the median of the others is printed, with the slowest of the three RISCs of every zone."""
import csv, re, statistics, sys
from collections import defaultdict

path, Mt, Kt, Nt = sys.argv[1], *map(int, sys.argv[2:5])
with open(path) as f:
    freq = float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", f.readline()).group(1))
    start, runs = {}, defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(f, skipinitialspace=True):
        name = r["zone name"]
        if name not in ("QUANT_X", "QUANT_W", "MATMUL", "REQUANT"):
            continue
        key = (r["core_x"], r["core_y"], r["RISC processor type"], name)
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            start[key] = t
        else:
            runs[(r["RISC processor type"], name)]["d"].append(t - start[key])
zone = {}
for name in ("QUANT_X", "QUANT_W", "MATMUL", "REQUANT"):
    per_risc = [statistics.median(runs[(risc, name)]["d"][1:]) for risc in ("TRISC_0", "TRISC_1", "TRISC_2") if runs[(risc, name)]["d"]]
    zone[name] = max(per_risc)
tiles_mm = Mt * Kt * Nt
print(f"Mt {Mt} Kt {Kt} Nt {Nt}: {tiles_mm} tile matmuls ({Mt * Nt} output tiles, {-(-Mt * Nt // 4)} blocks of up to 4)")
total = sum(zone.values())
for name, c in zone.items():
    print(f"  {name:8s} {c:10.0f} cycles  {c / freq:8.2f} us")
print(f"  {'sum':8s} {total:10.0f} cycles  {total / freq:8.2f} us")
print(f"  matmul: {zone['MATMUL'] / tiles_mm:.1f} cycles per 32x32x32 tile matmul")
