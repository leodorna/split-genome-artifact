#!/usr/bin/env bash
# EvoX baseline on GPU 2 (RTX A6000), the paper's design: nine cells, de-gen/ga-gen/ux-bf,
# island-local and panmictic, 10 seeds, 20 generations after 5 warmup, compiled only.
# Environment: conda env evox_gpu2 = Python 3.14, torch 2.14.0+cu126, evox 1.4.0 (pip_freeze.txt);
# the paper used torch 2.14.0 built for CUDA 13.0, which driver 570 here cannot run.
# Waits until the GPU has been idle 5 min; gpu_watch.log records any other user's GPU process.
set -uo pipefail
E="$(cd "$(dirname "$0")" && pwd)"
SC="$E/.."
# conda's (de)activate scripts reference unset variables, so activate without set -u
# Activate the Python environment with torch and evox before running this script.
python -c "import torch, evox" || exit 1
idle=0
while [ $idle -lt 10 ]; do
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && idle=$((idle+1)) || idle=0
  sleep 30
done
(
  while sleep 30; do
    others=""
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      # compare numeric uids: ps truncates user names longer than 8 characters
      uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
      [ -z "$uid" ] && continue
      [ "$uid" = "$(id -u)" ] && continue
      info=$(ps -o user=,args= -p "$pid" 2>/dev/null | cut -c1-120)
      others="$others$pid $info;"
    done
    [ -n "$others" ] && echo "$(date -u '+%F %T') SHARED: $others"
  done
) >> "$E/gpu_watch.log" &
watcher=$!
trap 'kill $watcher 2>/dev/null' EXIT
echo "$(date -u '+%F %T') start" >> "$E/gpu_watch.log"
# COMPILE_MODE (e.g. max-autotune-no-cudagraphs, the paper's baseline mode) is passed to
# bench_generation.py --compile-mode; its results go to $E/evox_<mode>/.
EXTRA="${COMPILE_MODE:+--compile-mode $COMPILE_MODE}" \
EVOX_SRC=/nonexistent SEEDS=10 GENS=20 WARMUP=5 OUTDIR="$E${COMPILE_MODE:+/evox_$COMPILE_MODE}" OPS="de-gen ga-gen ux-bf" \
  bash "$SC/evox_islands_sweep.sh"
echo "$(date -u '+%F %T') end" >> "$E/gpu_watch.log"
