#!/usr/bin/env bash
# A6000 (sm_86, 84 SMs) validation sweep of the launch heuristic. Pass 1: I=1 cells
# (the decisive ones), pass 2: the other six. Sphere, t_y=1, all s and t_x, 5 seeds.
# Usage: bash run_sweep.sh i1|rest [tag]
#   OPS / LAYOUTS override the operators / layouts (e.g. a rerun of one section); with
#   a tag the output is phase3_gpu2_<part>_<tag>.csv.
# While it runs, gpu_watch_<part>[_<tag>].log records every 30 s any compute process on
# the GPU that is not ours, so timings taken while the GPU was shared can be found.
set -uo pipefail
G="$(cd "$(dirname "$0")" && pwd)"
SC="$G/.."
R="$G/../../results/gpu2"
part=$1; tag=${2:+_$2}
WATCH="$R/gpu_watch_$part$tag.log"
(
  while sleep 30; do
    # nvidia-smi reports "[No data]" as the name of a process caught while starting or
    # exiting (our own short benchmark runs), so identify each PID through ps instead.
    others=""
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      # numeric uids: ps truncates user names longer than 8 characters
      uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
      [ -z "$uid" ] && continue                       # already exited: one of our short runs
      info=$(ps -o user=,args= -p "$pid" 2>/dev/null | cut -c1-120)
      [ "$uid" = "$(id -u)" ] && case "$info" in *bench_generation_v2*) continue;; esac
      others="$others$pid $info;"
    done
    [ -n "$others" ] && echo "$(date -u '+%F %T') SHARED: $others"
  done
) >> "$WATCH" &
watcher=$!
trap 'kill $watcher 2>/dev/null' EXIT
echo "$(date -u '+%F %T') start $part$tag OPS=${OPS:-all} LAYOUTS=${LAYOUTS:-both}" >> "$WATCH"
OPS="${OPS:-de-gen ga-gen ux-bf}" SWEEP="$R/sweep_$part.json" LAYOUTS="${LAYOUTS:-highdim highdim_vec4}" \
  CAPS="1 2 4 8 16 32 64 128 256 512 1024 2048" TXS="32 64 128 256 512" TYS="1" \
  SEEDS=5 REPS=10 WARMUP=5 REGS="$R/kernel_regs_v2_sm86.csv" \
  OUT="$R/phase3_gpu2_$part$tag.csv" PROGRESS="$R/progress_$part$tag.log" \
  bash "$SC/phase3_hybrid_probe_v2.sh"
echo "$(date -u '+%F %T') end" >> "$WATCH"
