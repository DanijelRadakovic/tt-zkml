#!/usr/bin/env python3
"""Joins the CPU/SIMD benchmark (results/cpu_simd.csv) with the device benchmark
(../quant_tensor_trace_bench/results/device.csv) into results/comparison.md.

Same job on both sides: the whole tensor of `tiles` tiles (32 x 32*tiles elements, 1 tile = 1024 elements) from the
input values to the quantized values `q`, the scaler and the M31 residues of `q`, in milliseconds; the number in
parentheses is microseconds per tile (the time divided by the number of tiles). Lower is faster.
  CPU 1 thread    std::simd on one CPU thread
  CPU 16 threads  std::simd on all 16 threads of the CPU (rayon)
  device          one Tensix core: pot_quantize_per_tensor(QuantRange::M31) followed by pot_quant_m31
Two more columns: the CPU deriving the rest of the witness itself (the remainder and the argmax, 1 and 16 threads), and the
cost of copying the M31 tensor from the device to the host.
"""
import csv, pathlib

here = pathlib.Path(__file__).resolve().parent
cpu = {}
for r in csv.DictReader(open(here / "results" / "cpu_simd.csv")):
    cpu[(int(r["bits"]), int(r["tiles"]), r["impl"], int(r["trace"]))] = float(r["ms"])
dev = {}
for r in csv.DictReader(open(here.parent / "quant_tensor_trace_bench" / "results" / "device.csv")):
    dev[(int(r["bits"]), int(r["tiles"]), r["op"])] = float(r["ms"])

out = []
for bits in (8, 16, 32):
    out.append(f"#### {bits} bits\n")
    out.append("| tiles | CPU 1 thread | CPU 16 threads | device | fastest | device vs CPU 1 thread | CPU + remainder and argmax: 1 thread / 16 threads | copy M31 to host |")
    out.append("|---|---|---|---|---|---|---|---|")
    for tiles in sorted({k[1] for k in dev if k[0] == bits}):
        s1, sn = cpu[(bits, tiles, "simd-1", 0)], cpu[(bits, tiles, "simd-N", 0)]
        w1, wn = cpu[(bits, tiles, "simd-1", 1)], cpu[(bits, tiles, "simd-N", 1)]
        d = dev[(bits, tiles, "quantize")] + dev[(bits, tiles, "to_m31")]
        best = min((s1, "CPU 1 thread"), (sn, "CPU 16 threads"), (d, "device"))[1]
        ratio = d / s1
        vs = f"{ratio:.1f}x slower" if ratio > 1 else f"{1 / ratio:.1f}x faster"
        pt = lambda ms: f"{ms:.3f} ms ({ms * 1e3 / tiles:.2f} us/tile)"
        out.append(f"| {tiles} | {pt(s1)} | {pt(sn)} | {pt(d)} | {best} | {vs} | {w1:.3f} / {wn:.3f} ms | {dev[(bits, tiles, 'readback_m31')]:.3f} ms |")
    out.append("")
(here / "results" / "comparison.md").write_text("\n".join(out))
print("\n".join(out))
