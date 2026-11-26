#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
#
# Zone cycles and wall time of execute_layer for several shapes. Run from the repo root, on the machine with the device
# (inside the IRD container):   TT_METAL_HOME=... benchmarks/execute_layer/run_all.sh
set -euo pipefail
: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_LOGGER_LEVEL=error
BIN=${BIN:-.build/default/zkml_execute_layer_bench}
OUT=benchmarks/execute_layer/results
LOG=$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
SHAPES=${SHAPES:-"1,1,1 2,4,2 4,8,4 4,16,4 2,32,2 1,8,8 8,8,1"}   # Mt,Kt,Nt
mkdir -p "$OUT"
: > "$OUT/cycles.txt"; : > "$OUT/wall.txt"
for act in relu none; do
  for shape in $SHAPES; do
    IFS=, read -r mt kt nt <<< "$shape"
    rm -f "$LOG"
    TT_METAL_DEVICE_PROFILER=1 "$BIN" profile "$mt" "$kt" "$nt" $act 8 > /dev/null
    echo "activation $act" >> "$OUT/cycles.txt"
    python3 -I benchmarks/execute_layer/analyze.py "$LOG" "$mt" "$kt" "$nt" >> "$OUT/cycles.txt"
    "$BIN" wall "$mt" "$kt" "$nt" $act 50 | tail -1 >> "$OUT/wall.txt"
  done
done
cat "$OUT/cycles.txt" "$OUT/wall.txt"
