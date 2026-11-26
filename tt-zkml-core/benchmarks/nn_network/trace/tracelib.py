# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""Reads a device-profiler CSV (TT_METAL_DEVICE_PROFILER=1) of `zkml_nn_network_bench device|deviceq` into the zones of one run."""
import csv, re
from collections import defaultdict

RISCS = ["NCRISC", "BRISC", "TRISC_0", "TRISC_1", "TRISC_2"]


def load_run(path, run=-1):
    """-> (mhz, {risc: [(name, start, end, depth)]}, t0) for the last complete run (run = -1) with all zones relative to the kernel start t0."""
    with open(path) as f:
        mhz = float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", f.readline()).group(1))
        rows = [(r["RISC processor type"], r["zone name"], r["type"], int(r["time[cycles since reset]"])) for r in csv.DictReader(f, skipinitialspace=True)]
    stacks, nrun, runs = defaultdict(list), defaultdict(lambda: -1), defaultdict(lambda: defaultdict(list))
    for risc, name, kind, t in rows:
        if kind == "ZONE_START":
            if name.endswith("-KERNEL"):
                nrun[risc] += 1
            stacks[risc].append((name, t, nrun[risc], len(stacks[risc])))
        elif kind == "ZONE_END" and stacks[risc]:
            n, t0, r, depth = stacks[risc].pop()
            if r >= 0:
                runs[r][risc].append((n, t0, t, depth))
    # complete runs: all five RISCs present and a MATMUL zone on TRISC_1
    ok = [r for r in sorted(runs) if all(x in runs[r] for x in RISCS) and any(z[0] == "MATMUL" for z in runs[r]["TRISC_1"])]
    r = ok[run]
    z = runs[r]
    t0 = min(s for risc in z for n, s, e, d in z[risc] if n.endswith("-KERNEL"))
    out = {risc: sorted(((n, s - t0, e - t0, d) for n, s, e, d in z[risc]), key=lambda x: (x[1], x[3])) for risc in z}
    return mhz, out, t0


def seq(zones, risc, name):
    """The zones called `name` on `risc`, in time order."""
    return [(s, e) for n, s, e, d in zones[risc] if n == name]
