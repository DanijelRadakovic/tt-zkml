#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
#
# twos / sm_float / sm_int accumulator of execute_layer: outputs checked against each other, zone cycles and wall time.
# Run from the repo root, on the machine with the device (inside the IRD container):
#   TT_METAL_HOME=... benchmarks/execute_layer_2s_vs_sm_acc/run_all.sh
set -euo pipefail
: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_LOGGER_LEVEL=error
BIN=${BIN:-.build/default/zkml_execute_layer_2s_vs_sm_acc_bench}
D=benchmarks/execute_layer_2s_vs_sm_acc
OUT=$D/results
LOG=$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
SHAPES=${SHAPES:-"1,1,1 2,4,2 4,8,4 4,16,4 2,32,2 1,8,8 8,8,1"}   # Mt,Kt,Nt (Kt <= 32: the float variants are exact below 2^24)
mkdir -p "$OUT"
{
  echo "# outputs of the variants against each other (out bytes, decoded acc, scaler_out)"
  for act in relu none; do for shape in $SHAPES; do
    IFS=, read -r mt kt nt <<< "$shape"
    echo "## $shape $act"; "$BIN" check "$mt" "$kt" "$nt" $act
  done; done
} > "$OUT/check.txt"
: > "$OUT/cycles.txt"; : > "$OUT/wall.txt"
echo "act shape variant QUANT_X QUANT_W MATMUL REQUANT QX_P1 QX_P2" >> "$OUT/cycles.txt"
for act in relu none; do
  for shape in $SHAPES; do
    IFS=, read -r mt kt nt <<< "$shape"
    for v in twos sm_float sm_int sm_int_sfpi sm_int_sfpi_absmax; do
      rm -f "$LOG"
      TT_METAL_DEVICE_PROFILER=1 "$BIN" profile $v "$mt" "$kt" "$nt" $act 8 > /dev/null
      echo "$act $shape $v $(python3 -I $D/analyze.py "$LOG")" >> "$OUT/cycles.txt"
      "$BIN" wall $v "$mt" "$kt" "$nt" $act 50 | tail -1 >> "$OUT/wall.txt"
    done
  done
done
cat "$OUT/check.txt" "$OUT/cycles.txt" "$OUT/wall.txt"
