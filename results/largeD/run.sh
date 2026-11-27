#!/usr/bin/env bash
# Equivalence and time-to-quality at a large-D cell: I = 1, N = 64, D = 2^20 (64M genes,
# ux-bf 32 D bits), below one couple per SM. Per operator: 30 seeds of 1000-generation
# fitness traces of our kernel, unsplit (s = 1, t_x = 512) and at the heuristic's split
# configuration, and of the island-local baseline (default compile mode, as the other
# quality runs); then the generation time of both configurations and of the baseline
# (max-autotune-no-cudagraphs, as all timings of the paper) with check_speedup.py.
# Run inside blackwell_env with an idle GPU. About an hour on the RTX PRO 6000.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
export I=1 N=64 QD=1048576 QDBIN=33554432 SEEDS="${SEEDS:-30}" GENS="${GENS:-1000}"
export OUT="$HERE/quality"
# heuristic (Equation 14, 188 SMs): de (4, 512, 1), sbx-pm and ux-bf (8, 512, 1)
for spec in "de-gen 4" "ga-gen 8" "ux-bf 8"; do
  set -- $spec
  ONLY_OPS=$1 CONFIGS="s1 highdim 1 512
heur highdim $2 512" bash "$ROOT/scripts/quality_check.sh"
done
for spec in "de sphere 4" "sbx-pm sphere 8" "ux-bf onemax 8"; do
  set -- $spec
  for cfg in "$3,512,1" "1,512,1"; do
    python3 "$ROOT/scripts/check_speedup.py" --op $1 --function $2 --I 1 --N 64 --D 1048576 \
      --config "$cfg" --seeds 10 --outdir "$HERE/timing" || true
  done
done
echo "[$(date '+%T')] all done"
