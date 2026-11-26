#!/usr/bin/env bash
# One-off (2026-11-25): extend every quality check of the paper from 10 to 30 seeds
# (adds seeds 10..29; seeds 0..9 are unchanged), after the GPU is idle.
set -uo pipefail
cd "$(dirname "$0")/../.."
log() { echo "[$(date '+%F %T')] $*"; }
idle=${SKIP_WAIT:+120}; idle=${idle:-0}
log "waiting for the GPU to be idle for 120 s"
while [ $idle -lt 120 ]; do
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then idle=$((idle + 10)); else idle=0; fi
  sleep 10
done
export EVOX_SRC="${EVOX_SRC:-}"   # an EvoX source checkout, or empty for the installed package
R="$(pwd)"; S="$R/data/cec2013lsgo"; T="$R/results/transfer"
log "Sphere/OneMax"
SEED0=10 SEEDS=30 OUT="$R/results/rerun_v3/quality" bash scripts/quality_check.sh
for f in cec-f1 cec-f2 cec-f3 cec-f12; do
  n="${f#cec-f}"; log "$f"
  FITNESS="$f" QD=1000 KARGS="--shift-file $S/F$n-xopt.txt" PARGS="--shift-file $S/F$n-xopt.txt" \
    SEED0=10 SEEDS=30 OUT="$T/$f/quality" EXTRA_CONFIGS="s4w highdim 4 128" bash scripts/quality_check.sh
done
log "knapsack"
FITNESS=knapsack QDBIN=4096 SEED0=10 SEEDS=30 OUT="$T/knapsack/quality" \
  EXTRA_CONFIGS="s2w highdim 2 64" bash scripts/quality_check.sh
log "analysis"
python scripts/analyze_quality.py --dir results/rerun_v3/quality > results/rerun_v3/quality/quality.txt 2>&1
for f in cec-f1 cec-f2 cec-f3 cec-f12 knapsack; do
  python scripts/analyze_quality.py --dir "$T/$f/quality" --fitness "$f" > "$T/$f/quality/quality.txt" 2>&1
done
(cd scripts && python quality_table.py) > "$T/table_quality_supp.txt"
python scripts/plot_quality_paper.py > /dev/null
log "ALL DONE"
