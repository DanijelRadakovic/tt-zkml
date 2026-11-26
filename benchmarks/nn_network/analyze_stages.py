#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Where the device time of one execute_network call goes, per stage (device profiler log of `zkml_nn_network_bench device`).

usage: analyze_stages.py profile_log_device.csv M W L

Zones (see execute_network.cpp, reader.cpp, writer.cpp; the WAIT_* zones exist only in profiler builds):
  compute (TRISC_1): QUANT_X once, then per layer WAIT_W, QUANT_W, WAIT_ACC, MATMUL, REQUANT
  reader  (NCRISC):  READ_W per layer          writer (BRISC): WAIT_DONE, WRITE per layer
The first two runs are warm-up and dropped; medians over the rest. Cycles are converted with the chip clock of the log.
"""
import csv, re, statistics as st, sys
from collections import defaultdict

path, M, W, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with open(path) as f:
    mhz = float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", f.readline()).group(1))
    rows = [(r["RISC processor type"], r["zone name"], r["type"], int(r["time[cycles since reset]"])) for r in csv.DictReader(f, skipinitialspace=True)]

stacks, nrun, runs = defaultdict(list), defaultdict(lambda: -1), defaultdict(lambda: defaultdict(list))
for risc, name, kind, t in rows:
    if kind == "ZONE_START":
        if name.endswith("-KERNEL"):
            nrun[risc] += 1
        stacks[risc].append((name, t, nrun[risc]))
    elif kind == "ZONE_END":
        n, t0, r = stacks[risc].pop()
        if r >= 0:
            runs[r][risc].append((n, t0, t))

us = lambda c: c / mhz
Mt, Wt = M // 32, W // 32
res = defaultdict(list)
for r in sorted(runs)[2:]:
    z = runs[r]
    kern = [(s, e) for risc in z for n, s, e in z[risc] if n.endswith("-KERNEL")]
    res["kernel"].append(us(max(e for _, e in kern) - min(s for s, _ in kern)))
    tot = defaultdict(float)
    for risc, names in (("TRISC_1", ("QUANT_X", "WAIT_W", "QUANT_W", "WAIT_ACC", "MATMUL", "REQUANT")), ("NCRISC", ("READ_W",)), ("BRISC", ("WAIT_DONE", "WRITE"))):
        for n, s, e in z.get(risc, []):
            if n in names:
                tot[n] += us(e - s)
    for n, v in tot.items():
        res[n].append(v)
med = {k: st.median(v) for k, v in res.items()}
k = med["kernel"]
macs = L * M * W * W
print(f"M={M} W={W} L={L}: kernel {k:.1f} us for {macs / 1e6:.1f} M MACs ({macs / (k * mhz):.0f} MACs/cycle)\n")
print("compute (TRISC_1), summed over the run")
print(f"  {'stage':10s} {'us':>8s} {'% of kernel':>12s} {'us/layer':>9s}  note")
tiles = {"QUANT_X": Mt * Wt, "QUANT_W": L * Wt * Wt, "MATMUL": L * Mt * Wt * Wt, "REQUANT": L * Mt * Wt}
notes = {"QUANT_X": "tiles of x", "QUANT_W": "tiles of w (absmax pass + quantize pass)", "MATMUL": "output tiles x Kt matmul_tiles", "REQUANT": "tiles of acc (scaler + sweep)",
         "WAIT_W": "waiting for the first weight tile (reader behind)", "WAIT_ACC": "waiting for the writer to free the accumulator"}
acc = 0.0
for n in ("QUANT_X", "WAIT_W", "QUANT_W", "WAIT_ACC", "MATMUL", "REQUANT"):
    v = med.get(n, 0.0)
    acc += v
    extra = f"{v * mhz / tiles[n]:.0f} cycles/tile" if n in tiles else ""
    print(f"  {n:10s} {v:8.1f} {100 * v / k:11.1f}% {v / L:9.2f}  {notes[n]} {('- ' + extra) if extra else ''}")
print(f"  {'(rest)':10s} {k - acc:8.1f} {100 * (k - acc) / k:11.1f}%            start-up, kernel end, wait inside the stages not in a zone")
if "MATMUL" in med:
    mm_calls = L * Mt * Wt * Wt
    print(f"  matmul: {med['MATMUL'] * mhz / mm_calls:.0f} cycles per matmul_tiles call (one 32x32 x 32x32 tile product)")
print("\nreader (NCRISC): READ_W", f"{med.get('READ_W', 0):.1f} us (includes waiting for compute to free c_1; {100 * med.get('READ_W', 0) / k:.0f}% of the kernel)")
print("writer (BRISC):  WAIT_DONE", f"{med.get('WAIT_DONE', 0):.1f} us, WRITE {med.get('WRITE', 0):.1f} us ({100 * med.get('WRITE', 0) / k:.0f}% of the kernel)")
