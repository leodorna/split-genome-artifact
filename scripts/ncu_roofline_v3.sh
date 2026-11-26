#!/usr/bin/env bash
# ncu profiles for the instruction roofline: every kernel of one generation.
# stdin lines: tags op layout I N D tx ty cap   (from roofline_configs_v3.py)
#
# A generation is FusedGeneration_HighDim plus ParallelRestore when I*N <= 64
# (the deferred restore, DEFER_ROWS in bench_generation_v2); both are profiled
# for generation WARMUP (0-based), after WARMUP unprofiled generations.
# --clock-control none keeps the boost clocks of the timing runs (ncu's default
# locks the base clock). Configs whose raw file exists are skipped (resumable).
#
# Env: RAWDIR, WARMUP (3), SEED (42), METRICS (default below), FITNESS (--fitness, if set).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/code/bin/bench_generation_v2"
RAWDIR="${RAWDIR:-$ROOT/results/rerun_v3/roofline/raw}"
WARMUP="${WARMUP:-3}"; SEED="${SEED:-42}"
DEFER_ROWS="${DEFER_ROWS:-64}"
source "$(dirname "$0")/roofline_metrics_v3.sh"
FIT_ARGS=(); [ -n "${FITNESS:-}" ] && FIT_ARGS=(--fitness "$FITNESS")
mkdir -p "$RAWDIR"

while read -r tags op lay I N D tx ty cap; do
  [ -z "${op:-}" ] && continue
  case "$tags" in \#*) continue;; esac
  out="$RAWDIR/${op}_${lay}_I${I}_N${N}_D${D}_tx${tx}_ty${ty}_cap${cap}.csv"
  if [ -s "$out" ] && grep -q '^"ID"' "$out"; then echo "skip (done) $out"; continue; fi
  k=1; [ $(( I * N )) -le "$DEFER_ROWS" ] && k=2
  printf '== %s %s I=%s N=%s D=%s tx=%s ty=%s cap=%s (%s kernel/gen) ==\n' "$op" "$lay" "$I" "$N" "$D" "$tx" "$ty" "$cap" "$k"
  ncu --clock-control none --cache-control all --replay-mode kernel \
      --launch-skip $(( WARMUP * k )) --launch-count "$k" \
      -k "regex:FusedGeneration_HighDim|ParallelRestore" \
      --page raw --csv --print-units base --metrics "$METRICS" \
      "$BIN" --mode generation --algo "$op" --layout "$lay" \
      --islands "$I" --n "$N" --d "$D" --xthreads "$tx" --yblock "$ty" --hd-cap "$cap" \
      --reps 1 --warmup "$WARMUP" --seed "$SEED" "${FIT_ARGS[@]}" > "$out.tmp" 2> "${out%.csv}.log"
  rc=$?
  grep '^"' "$out.tmp" > "$out"; rm -f "$out.tmp"
  n=$(( $(grep -c '^"' "$out") - 2 ))
  echo "   rc=$rc kernels=$n"
  [ "$n" -eq "$k" ] || echo "   WARNING: expected $k kernels, see ${out%.csv}.log"
done
