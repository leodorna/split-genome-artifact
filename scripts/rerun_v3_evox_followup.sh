#!/usr/bin/env bash
# Follow-up to the v3 rerun, on the same machine state, one job at a time:
#   1. EvoX de-gen and ga-gen baselines (island-local and panmictic, 10 seeds)
#   2. EvoX ga-gen with tournament_selection (panmictic, 10 seeds)
#   3. evox.utils.clamp vs torch.clamp and StdWorkflow vs step (3 seeds)
#   4. restore deferral probe (defer_probe_v2.py on the v3 tuned configs)
# Outputs under results/rerun_v3/. Run inside blackwell_env.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SB="$(cd "$HERE/.." && pwd)"
R3="$SB/results/rerun_v3"
log() { echo "[$(date '+%F %T')] $*"; }

log "evox de-gen ga-gen"
OPS="de-gen ga-gen" SEEDS=10 GENS=20 WARMUP=5 OUTDIR="$R3/evox" bash "$HERE/evox_islands_sweep.sh"
log "evox ga-gen tournament_selection"
OPS=ga-gen ONLY_PANMICTIC=1 EXTRA="--ga-select evox" SEEDS=10 GENS=20 WARMUP=5 \
  OUTDIR="$R3/evox/tournament_selection" bash "$HERE/evox_islands_sweep.sh"
log "evox clamp and workflow"
(cd "$SB/python" && PYTHONPATH=.${EVOX_SRC:+:$EVOX_SRC} python -m step_bench.bench_evox_variants \
  --sweep "$SB/configs/sweep_realistic.json" --out "$R3/evox/evox_variants.csv" 2>/dev/null)
log "defer probe"
python "$HERE/defer_probe_v2.py" --final "$R3/phase3_unified_final_v3.csv" --out "$R3/defer_probe_v3.csv" 2>/dev/null
log "ALL DONE"
