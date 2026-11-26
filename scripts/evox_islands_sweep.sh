#!/usr/bin/env bash
# EvoX generation reference with repeats, panmictic and island-local.
#
# For each realistic triple (I, N, D) and op, runs bench_generation.py on
# I*N rows twice -- --islands 1 (panmictic, one EvoX population) and
# --islands I (island-local parent/donor draws, as in the fused kernel) --
# for SEEDS seeds, compiled only.  Rows go to OUTDIR/evox_gen_<op>.csv.
# Env: SWEEP, SEEDS (10), GENS (20), WARMUP (5), OUTDIR, OPS,
#      EXTRA (extra bench flags, e.g. "--ga-select evox" or "--problem rastrigin"),
#      ONLY_PANMICTIC (0/1), ONLY_ISLANDS (0/1: skip the panmictic run when I > 1),
#      EVOX_SRC (EvoX source checkout on PYTHONPATH; default: the first machine's path).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYDIR="$ROOT/python"
SWEEP="${SWEEP:-$ROOT/configs/sweep_realistic.json}"
OUT="${OUTDIR:-$ROOT/results/rerun_v3/evox}"
SEEDS="${SEEDS:-10}"; GENS="${GENS:-20}"; WARMUP="${WARMUP:-5}"
OPS="${OPS:-de-gen ga-gen ux-bf}"; EXTRA="${EXTRA:-}"; ONLY_PANMICTIC="${ONLY_PANMICTIC:-0}"
ONLY_ISLANDS="${ONLY_ISLANDS:-0}"
# EVOX_SRC: an EvoX source checkout to use instead of the installed package (first machine:
# EVOX_SRC); a missing path is harmless.
export PYTHONPATH="$PYDIR${EVOX_SRC:+:$EVOX_SRC}"
mkdir -p "$OUT"

mapfile -t TUPLES < <(python3 "$ROOT/scripts/sweep_tuples.py" --config "$SWEEP")

cd "$PYDIR"
for op in $OPS; do
  OUTFILE="$OUT/evox_gen_${op}.csv"
  echo "op,N,D,mode,per_gen_mean_us,per_gen_min_us,islands,seed,per_gen_median_us" > "$OUTFILE"
  for t in "${TUPLES[@]}"; do
    I=${t%% *}; rest=${t#* }; N=${rest% *}; D=${rest#* }
    # Same population size in memory as the kernels: ux-bf's D counts bits,
    # packed 32 per word, so it is 32x the tuple's.
    [ "$op" = ux-bf ] && D=$((D * 32))
    total=$((I * N))
    for isl in 1 "$I"; do
      [ "$ONLY_PANMICTIC" = 1 ] && [ "$isl" != 1 ] && continue
      [ "$ONLY_ISLANDS" = 1 ] && [ "$isl" = 1 ] && [ "$I" != 1 ] && continue
      [ "$isl" = 1 ] && [ "$I" = 1 ] && [ "${done1:-}" = "$op $t" ] && continue
      for ((s = 0; s < SEEDS; s++)); do
        python -m step_bench.bench_generation --op "$op" --n "$total" --d "$D" \
          --islands "$isl" --seed "$s" --generations "$GENS" --warmup "$WARMUP" \
          --compiled-only $EXTRA 2>/dev/null | tail -n +2 >> "$OUTFILE"
      done
      done1="$op $t"
    done
    echo "$op I=$I N=$N D=$D done"
  done
  echo "wrote $OUTFILE"
done
