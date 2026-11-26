#!/usr/bin/env bash
# The paper's baselines with torch.compile in max-autotune-no-cudagraphs mode (the
# fastest mode in every cell of the compile-mode probe, evox_modes.sh), 10 seeds:
#   results/rerun_v3/evox_autotune/          main comparison, panmictic and island-local
#   results/transfer/<f>/evox_autotune/      CEC f1, f2, f3, f12 and the knapsack
set -uo pipefail
cd "$(dirname "$0")/../.."
R="$(pwd)"
export EVOX_SRC="${EVOX_SRC:-}"   # an EvoX source checkout, or empty for the installed package
M="--compile-mode max-autotune-no-cudagraphs"
log() { echo "[$(date '+%F %T')] $*"; }
log "main comparison"
OPS="de-gen ga-gen ux-bf" EXTRA="$M" SEEDS=10 GENS=20 WARMUP=5 \
  SWEEP="$R/configs/sweep_realistic.json" OUTDIR="$R/results/rerun_v3/evox_autotune" \
  bash scripts/evox_islands_sweep.sh
for fit in cec-f1 cec-f2 cec-f3 cec-f12 knapsack; do
  ops="de-gen ga-gen"; [ "$fit" = knapsack ] && ops=ux-bf
  log "$fit"
  OPS="$ops" ONLY_ISLANDS=1 EXTRA="--problem $fit $M" SEEDS=10 GENS=20 WARMUP=5 \
    SWEEP="$R/configs/sweep_transfer.json" OUTDIR="$R/results/transfer/$fit/evox_autotune" \
    bash scripts/evox_islands_sweep.sh
done
log "ALL DONE"
