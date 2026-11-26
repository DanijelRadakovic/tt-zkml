#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Markdown tables from results/nn.txt: per layer time (us) for every width, CPU threads and device, and the speedups."""
import re, sys
from collections import defaultdict

rows = defaultdict(dict)  # (M, K, L) -> {N: {label: layer_us}}
for line in open(sys.argv[1]):
    m = re.match(r"nn (cpu|device) (?:threads=(\d+)|(\w+)) M=(\d+) K=(\d+) N=(\d+) L=(\d+): layer_us=([\d.]+)", line)
    if not m:
        continue
    kind, th, dt, M, K, N, L, us = m.groups()
    label = f"CPU {th} thr" if kind == "cpu" else f"device {dt}"
    rows[(int(M), int(K), int(L))].setdefault(int(N), {})[label] = float(us)

order = ["CPU 1 thr", "CPU 6 thr", "CPU 12 thr", "CPU 16 thr", "device fp32", "device bf16"]
for (M, K, L), by_n in sorted(rows.items()):
    macs = lambda N: M * K * N
    print(f"\n### x = {M} x {K}, w = {K} x N, {L} layers (time per layer in us; speedup against 1 CPU thread)\n")
    print("| N | MACs/layer | " + " | ".join(order) + " | device fp32 vs CPU 1 | device bf16 vs CPU 1 | device bf16 vs best CPU |")
    print("|---|---|" + "---|" * (len(order) + 3))
    for N, d in sorted(by_n.items()):
        cells = [f"{d[o]:.1f}" if o in d else "n/a" for o in order]
        c1 = d.get("CPU 1 thr")
        best = min((d[o] for o in order[:4] if o in d), default=None)
        sp = lambda a, b: f"{b / a:.1f}x" if a and b else "n/a"
        print(f"| {N} | {macs(N) / 1e6:.2f} M | " + " | ".join(cells) + f" | {sp(d.get('device fp32'), c1)} | {sp(d.get('device bf16'), c1)} | {sp(d.get('device bf16'), best)} |")
