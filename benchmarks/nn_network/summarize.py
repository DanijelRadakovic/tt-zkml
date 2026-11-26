#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Markdown tables from results/net.txt: network time (us) per batch size M and width W, rows = depth L, and the speedups."""
import re, sys
from collections import defaultdict

rows = defaultdict(dict)  # (M, W) -> {L: {label: net_us}}
for line in open(sys.argv[1]):
    m = re.match(r"net (cpu|device|layers) (?:threads=(\d+)|(\w+)) M=(\d+) W=(\d+) L=(\d+): net_us=([\d.]+)", line)
    if not m:
        continue
    kind, th, dt, M, W, L, us = m.groups()
    label = f"CPU {th} thr" if kind == "cpu" else (f"device net {dt}" if kind == "device" else f"{dt} {kind} loop")
    rows[(int(M), int(W))].setdefault(int(L), {})[label] = float(us)

order = ["CPU 1 thr", "CPU 6 thr", "CPU 12 thr", "device net fp32", "device net bf16", "bf16 layers loop"]
for (M, W), by_l in sorted(rows.items()):
    print(f"\n### x = {M} x {W}, {W} x {W} layers (network time in us; speedup of the device network against the CPU)\n")
    print("| L | MACs | " + " | ".join(order) + " | bf16 vs CPU 1 | bf16 vs best CPU | fused vs layer loop |")
    print("|---|---|" + "---|" * (len(order) + 3))
    for L, d in sorted(by_l.items()):
        cells = [f"{d[o]:.0f}" if o in d else "n/a" for o in order]
        best = min((d[o] for o in order[:3] if o in d), default=None)
        sp = lambda a, b: f"{b / a:.1f}x" if a and b else "n/a"
        dev = d.get("device net bf16")
        print(f"| {L} | {L * M * W * W / 1e6:.1f} M | " + " | ".join(cells) + f" | {sp(dev, d.get('CPU 1 thr'))} | {sp(dev, best)} | {sp(dev, d.get('bf16 layers loop'))} |")
