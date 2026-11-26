#!/usr/bin/env bash
# Roofline ceilings: roofline_ceilings_v3 timed with CUDA events, then each of its
# kernels under ncu with the metrics of the generation profiles.
#   $RAWDIR/ceilings_events.csv   kernel,best_ms,work,work_unit
#   $RAWDIR/ceil_<kernel>.csv     ncu raw page, one kernel launch (after WARMUP)
# Env: RAWDIR, WARMUP (3), KERNELS.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/code/bin/roofline_ceilings_v3"
RAWDIR="${RAWDIR:-$ROOT/results/rerun_v3/roofline/raw}"
WARMUP="${WARMUP:-3}"
KERNELS="${KERNELS:-copy_dram copy_l2 issue_fp32 issue_int issue_imad issue_mix philox_draws binary_rng binary_rng_vec4}"
source "$(dirname "$0")/roofline_metrics_v3.sh"
mkdir -p "$RAWDIR"

"$BIN" --kernel all --warmup "$WARMUP" --reps 10 > "$RAWDIR/ceilings_events.csv" || exit 1
cat "$RAWDIR/ceilings_events.csv"

for kern in $KERNELS; do
  fn=$kern; [ "$kern" = binary_rng_vec4 ] && fn=binary_rng   # template, same function name
  out="$RAWDIR/ceil_${kern}.csv"
  echo "== ceiling $kern =="
  ncu --clock-control none --cache-control all --replay-mode kernel \
      --launch-skip "$WARMUP" --launch-count 1 -k "regex:^${fn}\$" \
      --page raw --csv --print-units base --metrics "$METRICS" \
      "$BIN" --kernel "$kern" --warmup "$WARMUP" --reps 1 > "$out.tmp" 2> "${out%.csv}.log"
  grep '^"' "$out.tmp" > "$out"; rm -f "$out.tmp"
  echo "   kernels=$(( $(grep -c '^"' "$out") - 2 ))"
done
