#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Cycles of the LOOP/LOOP2 zones of a probe run: zones.py <profile_log_device.csv> LOOP,LOOP2 <tiles*reps>"""
import csv, re, sys
path, names = sys.argv[1], sys.argv[2].split(",")
tiles = int(sys.argv[3])
with open(path) as f:
    hdr = f.readline(); freq = float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", hdr).group(1))
    st = {}
    res = {}
    for r in csv.DictReader(f, skipinitialspace=True):
        n = r["zone name"]
        if n not in names: continue
        key = (r["RISC processor type"], n)
        if r["type"] == "ZONE_START": st[key] = int(r["time[cycles since reset]"])
        else: res.setdefault(key, []).append(int(r["time[cycles since reset]"]) - st[key])
for k, v in sorted(res.items()):
    print(f"{k[0]:8s} {k[1]:6s} cycles {v[-1]:8d}  per tile {v[-1]/tiles:7.1f}")
