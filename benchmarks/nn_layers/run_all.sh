#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
#
# A network of L identical quantized layers for several output widths N: CPU with 1, 6, 12 and 16 threads, and the device (fp32 and bf16).
# Run from the repo root, inside the IRD container:   TT_METAL_HOME=... benchmarks/nn_layers/run_all.sh
set -euo pipefail
: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_LOGGER_LEVEL=error
BIN=${BIN:-.build/default/zkml_nn_bench}
OUT=benchmarks/nn_layers/results
K=${K:-256}
L=${L:-8}
REPS=${REPS:-20}
BATCHES=${BATCHES:-"64 128"}
WIDTHS=${WIDTHS:-"32 64 128 256 512"}
THREADS=${THREADS:-"1 6 12 16"}
mkdir -p "$OUT"
{
  echo "# correctness: the CPU layer against the device op (fp32), every output byte and the three scalers"
  "$BIN" check 64 "$K" 128
  "$BIN" check 128 "$K" 512
} > "$OUT/check.txt"
: > "$OUT/nn.txt"
for m in $BATCHES; do
  for n in $WIDTHS; do
    for t in $THREADS; do
      "$BIN" cpu "$t" "$m" "$K" "$n" "$L" "$REPS" >> "$OUT/nn.txt"
    done
    for dt in fp32 bf16; do
      "$BIN" device $dt "$m" "$K" "$n" "$L" "$REPS" 2>&1 | grep "^nn " >> "$OUT/nn.txt" || echo "nn device $dt M=$m K=$K N=$n L=$L: does not fit in L1" >> "$OUT/nn.txt"
    done
  done
done
python3 -I benchmarks/nn_layers/summarize.py "$OUT/nn.txt" | tee "$OUT/nn.md"
cat "$OUT/check.txt"
