#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
#
# Runs the whole matrix of the two's complement vs sign-magnitude benchmark. Run from the repo root, on the machine that
# has the device (inside the IRD container):
#   TT_METAL_HOME=... benchmarks/sign_mag_fpu_compatibillity/run_all.sh [check|compat|cycles|wall|all]
#
#   check   every variant against the host answer          -> results/check.md
#   compat  which input formats and which CB formats build and give correct bytes, one process each -> results/compat.md
#   cycles  device-profiler logs, one per configuration    -> results/raw/prof_*.csv, results/cycles.{csv,md}
#   wall    host wall time without the profiler            -> results/wall.txt
set -euo pipefail
: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_LOGGER_LEVEL=error
BIN=${BIN:-.build/default/zkml_sign_mag_fpu_compat_bench}
OUT=benchmarks/sign_mag_fpu_compatibillity/results
LOG=$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
STEP=${1:-all}

TILES=${TILES:-"48 192"}
WORK=${WORK:-6144}          # tiles * reps per run
MODES="full sfpu"
mkdir -p "$OUT/raw"
variants_for() { if [[ $1 == bf16 ]]; then echo "twos_lib twos_own sm_uint8pack"; else echo "twos_lib twos_own sm_int8pack sm_uint8pack"; fi; }

if [[ $STEP == check || $STEP == all ]]; then
  : > "$OUT/check.md"
  for inp in fp32 bf16; do
    echo "### input $inp" >> "$OUT/check.md"; echo >> "$OUT/check.md"
    echo "| variant | packer | wrong bytes of 49152 | first wrong |" >> "$OUT/check.md"
    echo "|---|---|---|---|" >> "$OUT/check.md"
    for v in $(variants_for $inp); do "$BIN" check 48 $v --input $inp | grep "^| $v" >> "$OUT/check.md"; done
    echo >> "$OUT/check.md"
  done
  cat "$OUT/check.md"
fi

if [[ $STEP == compat || $STEP == all ]]; then
  # One process per case: a program the JIT rejects aborts the process.
  one() {  # args: variant, extra bench args...; prints "builds, N wrong" or the failure
    local v=$1; shift
    local out; out=$(timeout 300 "$BIN" check 48 $v "$@" 2>&1 || true)
    local row; row=$(echo "$out" | grep "^| $v" || true)
    if [[ -n $row ]]; then
      local wrong; wrong=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$4); print $4}')
      if [[ $wrong == 0 ]]; then echo "builds, all bytes correct"; else
        local ms; ms=$(echo "$row" | sed -n 's/.*wrong m values: *\([^|]*[^| ]\).*/\1/p')
        echo "builds, **$wrong wrong bytes** (inputs m = $ms: x * 32 = m / 2 is a tie rounded the wrong way)"; fi
    elif echo "$out" | grep -q "same exponent format"; then echo "**does not build** (JIT: exponent format)"
    else echo "**does not build**: $(echo "$out" | grep -m1 -oE 'what\(\): .*|TT_THROW: No valid conversion[^(]*' | cut -c1-110)"; fi
  }
  {
    echo "# What builds and what is correct"; echo
    echo "## Input format of the quantizer against the output byte encoding"; echo
    echo "Pass 2 of the 8-bit SFPU quantizer (see README.md), 48 tiles, every output byte compared with the host answer."; echo
    echo "| input CB | twos_lib (UInt8 packer) | sm_int8pack (Int8 packer) | sm_uint8pack (UInt8 packer) |"
    echo "|---|---|---|---|"
    for inp in fp32 bf16 fp16; do
      echo "| $inp | $(one twos_lib --input $inp) | $(one sm_int8pack --input $inp) | $(one sm_uint8pack --input $inp) |"
    done
    echo; echo "## Which other CB formats may share a program with an Int8 CB"; echo
    echo "The sm_int8pack program with fp32 input plus one unused CB (c_5) of the given format."; echo
    echo "| extra CB format | result |"; echo "|---|---|"
    for f in fp32 bf16 fp16 tf32 bfp8_b bfp4_b bfp8_a int8 uint8 int16 uint16 int32 uint32; do
      echo "| $f | $(one sm_int8pack --extra-cb $f) |"
    done
  } | tee "$OUT/compat.md"
fi

if [[ $STEP == cycles || $STEP == all ]]; then
  rm -f "$OUT"/raw/prof_*.csv
  for inp in fp32 bf16; do
    for tiles in $TILES; do
      reps=$((WORK / tiles))
      for mode in $MODES; do
        for v in $(variants_for $inp); do
          rm -f "$LOG"
          TT_METAL_DEVICE_PROFILER=1 "$BIN" profile $v $mode $tiles $reps 11 --input $inp | tail -1
          cp "$LOG" "$OUT/raw/prof_${inp}_${v}_${mode}_${tiles}t_r${reps}.csv"
        done
      done
    done
  done
  python3 -I benchmarks/sign_mag_fpu_compatibillity/analyze.py "$OUT"
fi

if [[ $STEP == wall || $STEP == all ]]; then
  : > "$OUT/wall.txt"
  for inp in fp32 bf16; do
    for tiles in $TILES; do
      reps=$((WORK / tiles))
      for mode in $MODES; do
        for v in $(variants_for $inp); do
          echo -n "input $inp: " | tee -a "$OUT/wall.txt" >/dev/null
          "$BIN" wall $v $mode $tiles $reps --input $inp | tail -1 | tee -a "$OUT/wall.txt"
        done
      done
    done
  done
fi
