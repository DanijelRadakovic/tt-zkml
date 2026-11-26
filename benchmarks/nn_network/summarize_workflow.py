#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Markdown tables from results/net.txt for the whole workflow: load the model once (quantize the weights), then run N forward passes.

Strategies (times in us; every CPU entry is the best of the plain and the VNNI implementation and of 1, 6 and 12 threads, chosen separately for
every column):
  device per call   execute_network: the weights are quantized inside the kernel on every call; there is no load step
  device pre-quant  load = one pot_quantize_per_tensor per weight (and the wait for the device), then execute_network_quantized
  CPU per call      the weights are quantized inside every forward pass
  CPU pre-quant     load = quantize + pack the weights once, then forward passes without it
"""
import re, sys
from collections import defaultdict

KINDS = r"cpu-vnniload|cpu-vnniq|cpu-vnni|cpuload|cpuq|cpu|deviceload|deviceq|device|layers"
rows = defaultdict(lambda: defaultdict(dict))  # (M, W) -> L -> {key: us}
for line in open(sys.argv[1]):
    m = re.match(rf"net ({KINDS}) (?:threads=(\d+)|(\w+)) M=(\d+) W=(\d+) L=(\d+): net_us=([\d.]+)", line)
    if not m:
        continue
    kind, th, dt, M, W, L, us = m.groups()
    key = f"{kind}:{th}" if th else f"{kind}:{dt}"
    rows[(int(M), int(W))][int(L)][key] = float(us)


def best(d, kinds):
    c = [v for k, v in d.items() if k.split(":")[0] in kinds]
    return min(c) if c else None


def best_total(d, load_kinds, fwd_kinds, n):
    """min over thread counts and implementations of load + n * forward (the load and the forward pass of one implementation at one thread count)."""
    out = None
    for lk, fk in zip(load_kinds, fwd_kinds):
        for k, lv in d.items():
            if k.split(":")[0] != lk:
                continue
            th = k.split(":")[1]
            fv = d.get(f"{fk}:{th}")
            if fv is not None:
                t = lv + n * fv
                out = t if out is None else min(out, t)
    return out


def fmt(x):
    return "n/a" if x is None else (f"{x:.0f}" if x < 10000 else f"{x / 1000:.1f} ms")


for (M, W), by_l in sorted(rows.items()):
    print(f"\n### x = {M} x {W}, {W} x {W} layers\n")
    print("Forward pass and load (us). `pre-quant` forward passes do not quantize the weights; `load` quantizes them once.\n")
    print("| L | CPU per call | device per call | CPU pre-quant forward | device pre-quant forward | CPU load | device load | pre-quant forward: device vs CPU |")
    print("|---|---|---|---|---|---|---|---|")
    for L, d in sorted(by_l.items()):
        cpu_call = best(d, ["cpu", "cpu-vnni"])
        dev_call = d.get("device:bf16")
        cpu_q = best(d, ["cpuq", "cpu-vnniq"])
        dev_q = d.get("deviceq:bf16")
        cpu_load = best(d, ["cpuload", "cpu-vnniload"])
        dev_load = d.get("deviceload:bf16")
        sp = f"{cpu_q / dev_q:.2f}x" if cpu_q and dev_q else "n/a"
        print(f"| {L} | {fmt(cpu_call)} | {fmt(dev_call)} | {fmt(cpu_q)} | {fmt(dev_q)} | {fmt(cpu_load)} | {fmt(dev_load)} | {sp} |")
    print("\nThe whole workflow: load once, then N forward passes (total us; lower is better).\n")
    print("| L | N | device per call | device pre-quant | CPU per call | CPU pre-quant | best |")
    print("|---|---|---|---|---|---|---|")
    for L, d in sorted(by_l.items()):
        for n in (1, 10, 100):
            dev_call = d.get("device:bf16")
            dev_q = d.get("deviceq:bf16")
            dl = d.get("deviceload:bf16")
            t_dev_call = dev_call * n if dev_call else None
            t_dev_q = dl + n * dev_q if dl is not None and dev_q else None
            cc = best(d, ["cpu", "cpu-vnni"])
            t_cpu_call = cc * n if cc else None
            t_cpu_q = best_total(d, ["cpuload", "cpu-vnniload"], ["cpuq", "cpu-vnniq"], n)
            cand = {"device per call": t_dev_call, "device pre-quant": t_dev_q, "CPU per call": t_cpu_call, "CPU pre-quant": t_cpu_q}
            win = min((v, k) for k, v in cand.items() if v is not None)[1]
            print(f"| {L} | {n} | {fmt(t_dev_call)} | {fmt(t_dev_q)} | {fmt(t_cpu_call)} | {fmt(t_cpu_q)} | {win} |")
