#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
#
# A chained network of L square layers (up to 32; the default sweep stops at 6) for several batch sizes M, widths W and depths L: CPU (plain
# C++ and AVX-512 VNNI) with 1, 6 and 12 threads, the
# device (one execute_network call, fp32 and bf16) and, for
# comparison, L separate execute_layer calls (bf16).
# Run from the repo root, inside the IRD container:   TT_METAL_HOME=... benchmarks/nn_network/run_all.sh
set -euo pipefail
: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_LOGGER_LEVEL=error
BIN=${BIN:-.build/default/zkml_nn_network_bench}
OUT=${OUT:-benchmarks/nn_network/results}
REPS=${REPS:-20}
BATCHES=${BATCHES:-"32 128"}
WIDTHS=${WIDTHS:-"64 128 256 512"}
DEPTHS=${DEPTHS:-"1 2 4 6"}
THREADS=${THREADS:-"1 6 12"}
mkdir -p "$OUT"
{
  echo "# correctness: the CPU network against execute_network (fp32): output bytes, every accumulator, every scaler"
  "$BIN" check 32 64 3
  "$BIN" check 128 256 6
  "$BIN" check 64 128 4
  echo "# correctness: the VNNI CPU network against the plain C++ one (no device)"
  "$BIN" vnnicheck
} 2>&1 | grep -E "^#|^check|^vnnicheck" > "$OUT/check.txt"
: > "$OUT/net.txt"
for m in $BATCHES; do
  for w in $WIDTHS; do
    for l in $DEPTHS; do
      for t in $THREADS; do
        "$BIN" cpu "$t" "$m" "$w" "$l" "$REPS" >> "$OUT/net.txt"
        "$BIN" cpuvnni "$t" "$m" "$w" "$l" "$REPS" >> "$OUT/net.txt"
      done
      for dt in fp32 bf16; do
        "$BIN" device $dt "$m" "$w" "$l" "$REPS" 2>&1 | grep "^net " >> "$OUT/net.txt" || echo "net device $dt M=$m W=$w L=$l: does not fit in L1" >> "$OUT/net.txt"
      done
      "$BIN" layers bf16 "$m" "$w" "$l" "$REPS" 2>&1 | grep "^net " >> "$OUT/net.txt" || echo "net layers bf16 M=$m W=$w L=$l: does not fit in L1" >> "$OUT/net.txt"
    done
  done
done
python3 -I benchmarks/nn_network/summarize.py "$OUT/net.txt" | tee "$OUT/net.md"
cat "$OUT/check.txt"
