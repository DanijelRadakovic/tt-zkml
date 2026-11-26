#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
#
# Runs the SFPU-vs-FPU benchmark matrix. Run from the repo root, on the machine that has the device (inside
# the IRD container):
#   TT_METAL_HOME=... tools/quant_tensor_fpu_vs_sfpu/run_all.sh [cycles|wall|power|all]
#
#   cycles  device-profiler logs, one per configuration  -> results/raw/prof_*.csv   (~6 min)
#   wall    host wall time without the profiler          -> results/wall.txt         (~6 min)
#   power   sustained multi-core runs with power sampling -> results/power.txt       (~15 min)
set -euo pipefail
: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_LOGGER_LEVEL=error
BIN=${BIN:-.build/default/zkml_quant_fpu_vs_sfpu_bench}
OUT=tools/quant_tensor_fpu_vs_sfpu/results
LOG=$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
STEP=${1:-all}

# bits:out pairs. 8-bit also runs with INT8 output (the UInt8 packer path).
CONFIGS="8:int8 8:int32 16:int32 32:int32"
TILES=${TILES:-"16 64 256 1024"}
UNITS="sfpu fpu"
mkdir -p "$OUT/raw"

if [[ $STEP == cycles || $STEP == all ]]; then
  rm -f "$OUT"/raw/prof_*.csv
  for tiles in $TILES; do
    for cfg in $CONFIGS; do
      bits=${cfg%%:*}; out=${cfg##*:}
      for unit in $UNITS; do
        rm -f "$LOG"
        TT_METAL_DEVICE_PROFILER=1 "$BIN" profile bf16 $unit $bits $out $tiles 11 | tail -1
        cp "$LOG" "$OUT/raw/prof_bf16_${unit}_${bits}b_${out}_${tiles}t.csv"
      done
    done
  done
  python3 -I tools/quant_tensor_fpu_vs_sfpu/analyze.py "$OUT" > "$OUT/cycles.md"
fi

if [[ $STEP == wall || $STEP == all ]]; then
  : > "$OUT/wall.txt"
  for tiles in $TILES; do
    for cfg in $CONFIGS; do
      bits=${cfg%%:*}; out=${cfg##*:}
      for unit in $UNITS; do
        "$BIN" wall bf16 $unit $bits $out $tiles 50 | tail -1 | tee -a "$OUT/wall.txt"
      done
    done
  done
fi

if [[ $STEP == power || $STEP == all ]]; then
  CORES=${CORES:-16}; SECONDS_LOAD=${SECONDS_LOAD:-15}; ROUNDS=${ROUNDS:-3}; PTILES=${PTILES:-1024}
  : > "$OUT/power.txt"
  # Alternate the units every round, so a drift of the board (temperature, clocks) hits both equally.
  for round in $(seq 1 "$ROUNDS"); do
    for cfg in $CONFIGS; do
      bits=${cfg%%:*}; out=${cfg##*:}
      for unit in $UNITS; do
        echo -n "round $round | " | tee -a "$OUT/power.txt"
        "$BIN" power bf16 $unit $bits $out $PTILES $CORES $SECONDS_LOAD | tail -1 | tee -a "$OUT/power.txt"
      done
    done
  done
fi
