#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
#
# Runs the L1-fit benchmark over a matrix of configurations with the device profiler on and keeps
# one profile_log_device.csv per configuration in results/raw/. Run from the repo root:
#   TT_METAL_HOME=... tools/quant_tensor_fit_in_l1_bench/run_all.sh
set -euo pipefail
: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
BIN=${BIN:-.build/default/zkml_quant_l1_bench}
OUT=tools/quant_tensor_fit_in_l1_bench/results
ITERS=${ITERS:-11}   # run 0 is warm-up and dropped by analyze.py
LOG=$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
mkdir -p "$OUT/raw"
: > "$OUT/checks.txt"

for in_dt in f32 bf16; do
 for bits in 8 32; do
  for tiles in 16 64 128; do
   for block in 4 16; do
    for mode in stream resident; do
     rm -f "$LOG"
     name="${mode}_${in_dt}_${bits}b_${tiles}t_${block}blk"
     TT_METAL_DEVICE_PROFILER=1 TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_LOGGER_LEVEL=error \
       "$BIN" $mode $in_dt $bits $tiles $block $ITERS 2>&1 | grep '^config' | tee -a "$OUT/checks.txt"
     cp "$LOG" "$OUT/raw/$name.csv"
    done
   done
  done
 done
done
