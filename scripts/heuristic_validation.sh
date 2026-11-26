#!/usr/bin/env bash
# Out-of-sample validation of the revised launch heuristic: every s and t_x at t_y = 1
# on the cells of configs/sweep_validation.json (fixed, with the rule's predictions in
# configs/validation_predictions.csv, before measuring), for f2 and f12 (de-gen,
# ga-gen) and the knapsack (ux-bf). Into results/transfer/validation/<f>/.
# Env: WAIT_PID (start after that process exits), SEEDS (3). Run inside blackwell_env.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SB="$(cd "$HERE/.." && pwd)"
V="$SB/results/transfer/validation"
log() { echo "[$(date '+%F %T')] $*"; }
if [ -n "${WAIT_PID:-}" ]; then
  log "waiting for PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
mkdir -p "$V"
git -C "$SB" rev-parse HEAD > "$V/artifact_rev.txt"
for fit in cec-f2 cec-f12 knapsack; do
  F="$V/$fit"; mkdir -p "$F"
  ops="de-gen ga-gen"; [ "$fit" = knapsack ] && ops=ux-bf
  log "$fit: kernels"
  OPS="$ops" FITNESS="$fit" SWEEP="$SB/configs/sweep_validation.json" LAYOUTS="highdim highdim_vec4" \
    CAPS="1 2 4 8 16 32 64 128 256 512 1024 2048" TXS="32 64 128 256 512" TYS="1" \
    DEDUP_CAPS=1 SEEDS="${SEEDS:-3}" REPS=10 WARMUP=5 OUT="$F/phase3_$fit.csv" PROGRESS="$F/progress.log" \
    bash "$HERE/phase3_hybrid_probe_v2.sh"
done
log "ALL DONE"
