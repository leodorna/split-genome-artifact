#!/usr/bin/env bash
# Small speedup tests: six cases of the paper, each timed on both sides with
# scripts/check_speedup.py, which prints the exact commands and every timed generation of
# our kernel and of the baseline, and compares the speedup with the paper's. The cases
# cover the three operators, both variants, Sphere/OneMax and the transfer functions, a
# D = 1000 cell, the tuned configuration, and one case where the heuristic is slower than
# the baseline (the paper reports 0.81x there).
# About 10 minutes on the RTX PRO 6000, most of it the baseline's autotuning.
# Env: SEEDS (2 per side), TOL (0.15), EVOX_SRC (optional), PYTHON.
# Needs an idle GPU, the built kernels (make -C code bench) and torch + evox.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
SEEDS="${SEEDS:-2}"; TOL="${TOL:-0.15}"
CASES=(
  "--op sbx-pm --variant vec4   --function sphere   --I 32   --N 256  --D 131072"
  "--op de     --variant scalar --function sphere   --I 1    --N 1024 --D 1048576"
  "--op ux-bf  --variant scalar --function onemax   --I 1024 --N 1024 --D 1024"
  "--op de     --variant scalar --function cec-f12  --I 1024 --N 64   --D 1000"
  "--op sbx-pm --variant scalar --function cec-f3   --I 1024 --N 256  --D 4096 --config tuned"
  "--op ux-bf  --variant vec4   --function knapsack --I 32   --N 256  --D 131072"
)
mkdir -p check_speedup
summary=check_speedup/summary.txt; : > "$summary"
fail=0
for c in "${CASES[@]}"; do
  echo; echo "==================== $c"
  out=$($PY scripts/check_speedup.py $c --seeds "$SEEDS" --tol "$TOL" 2>&1); rc=$?
  echo "$out"
  sp=$(echo "$out" | grep -m1 '^speedup =' | sed 's/.*= //')
  pp=$(echo "$out" | grep -m1 '^paper (' | sed 's/.*speedup \([0-9.]*x\).*/\1/')
  verdict=$(echo "$out" | grep -E -m1 '^(PASS|FAIL)|reported only' | cut -d: -f1)
  [ $rc -eq 2 ] || [ -z "$sp" ] && verdict="ERROR"
  [ $rc -ne 0 ] && fail=1
  printf '%-90s measured %-7s paper %-7s %s\n' "$(echo $c | tr -s ' ')" "${sp:--}" "${pp:--}" "${verdict:-}" >> "$summary"
done
echo; echo "==================== summary (raw times and commands in check_speedup/)"
cat "$summary"
exit $fail
