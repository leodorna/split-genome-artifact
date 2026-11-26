#!/usr/bin/env bash
# One-off (2026-11-25): the island-local baselines of the paper's EvoX comparison
# (de-gen, ga-gen, ux-bf; nine cells) under other torch.compile modes, 3 seeds each:
# max-autotune-no-cudagraphs (autotuned Triton kernels) and reduce-overhead (CUDA
# graphs, with the generation cloning and step marking they need, see
# bench_generation.py). The default mode is results/rerun_v3/evox.
set -uo pipefail
cd "$(dirname "$0")/../.."
R="$(pwd)"
export EVOX_SRC="${EVOX_SRC:-}"   # an EvoX source checkout, or empty for the installed package
for m in max-autotune-no-cudagraphs reduce-overhead; do
  echo "[$(date '+%F %T')] $m"
  OPS="de-gen ga-gen ux-bf" ONLY_ISLANDS=1 EXTRA="--compile-mode $m" SEEDS=3 GENS=20 WARMUP=5 \
    SWEEP="$R/configs/sweep_realistic.json" OUTDIR="$R/results/rerun_v3/evox_modes/$m" \
    bash scripts/evox_islands_sweep.sh
done
echo "[$(date '+%F %T')] ALL DONE"
