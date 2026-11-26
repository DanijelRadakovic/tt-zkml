#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
#
# Pass 2 of the x / W quantization: pipelined hand-written instructions (ZKML_QLM=0, quant_sm8) against SFPLOADMACRO (ZKML_QLM=1,
# quant_sm8_lm), with the `sm_int` accumulator variant. Zone cycles and wall time. Run from the repo root, inside the IRD container:
#   TT_METAL_HOME=... benchmarks/execute_layer_2s_vs_sm_acc/run_quant_lm.sh
set -euo pipefail
: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_LOGGER_LEVEL=error
BIN=${BIN:-.build/default/zkml_execute_layer_2s_vs_sm_acc_bench}
D=benchmarks/execute_layer_2s_vs_sm_acc
OUT=$D/results/quant_lm.txt
LOG=$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
SHAPES=${SHAPES:-"1,1,1 2,4,2 4,8,4 4,16,4 2,32,2 1,8,8 8,8,1"}
rm -f "$TT_METAL_HOME/generated/profiler/.logs/zone_src_locations.log"
echo "shape qlm QUANT_X QUANT_W MATMUL REQUANT QX_P1 QX_P2 wall_us" > "$OUT"
for shape in $SHAPES; do
  IFS=, read -r mt kt nt <<< "$shape"
  for qlm in 0 1; do
    ZKML_QLM=$qlm "$BIN" check "$mt" "$kt" "$nt" relu > /dev/null   # all variants agree with each other
    rm -f "$LOG"
    ZKML_QLM=$qlm TT_METAL_DEVICE_PROFILER=1 "$BIN" profile sm_int "$mt" "$kt" "$nt" relu 8 > /dev/null
    cyc=$(python3 -I $D/analyze.py "$LOG")
    wall=$(ZKML_QLM=$qlm "$BIN" wall sm_int "$mt" "$kt" "$nt" relu 50 | tail -1 | sed 's/.*: \([0-9.]*\) us.*/\1/')
    echo "$shape $qlm $cyc $wall" >> "$OUT"
  done
done
cat "$OUT"
