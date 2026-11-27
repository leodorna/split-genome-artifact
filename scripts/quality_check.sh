#!/usr/bin/env bash
# Solution-quality check: does the fused kernel run the same algorithm as the EvoX
# baseline? Best and mean fitness per generation (--trace) for one small cell per
# operator, I=4 islands of N=64, D=1024 genes (4096 bits for ux-bf), SEEDS seeds,
# GENS generations. Ours in five launch configurations, EvoX island-local:
#   s1        scalar, unsplit           (s=1,  tx=128)
#   s8        scalar, split             (s=8,  tx=32)
#   s8defer   scalar, split, restore in a separate kernel (--defer-always)
#   v4s1      vec4,   unsplit           (s=1,  tx=32)
#   v4s8      vec4,   split             (s=8,  tx=32; ux-bf vec4 has 32 units, so one block)
# FITNESS=rastrigin|rosenbrock|cec-f1|cec-f2|cec-f3|cec-f12 evaluates that function
# instead (de-gen and ga-gen only; EvoX uses its own Rastrigin/Rosenbrock, the CEC
# functions come from step_bench/cec.py; both sides the function's standard bounds).
# FITNESS=knapsack runs ux-bf on the 0/1 knapsack instead of OneMax.
# Transfer-campaign overrides: QD (real-valued D, 1024), QDBIN (bits, 4096),
# KARGS / PARGS (extra kernel / baseline flags, e.g. the same --shift-file for both),
# EXTRA_CONFIGS (more "cfg layout cap tx [flags]" lines), CONFIGS (replaces the five),
# I and N (the cell, 4 and 64), ONLY_OPS (a subset of the operators), SEED0 (first seed index, 0:
# with SEED0=10 SEEDS=30, adds seeds 10..29 to an existing run of 10).
# Output: OUT/<op>_<config>_<seed>.csv (gen,best,mean). Run inside blackwell_env.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SB="$(cd "$HERE/.." && pwd)"
ROOT="$SB"
BIN="$ROOT/code/bin/bench_generation_v2"
OUT="${OUT:-$SB/results/rerun_v3/quality}"
SEEDS="${SEEDS:-10}"; SEED0="${SEED0:-0}"; GENS="${GENS:-1000}"; I="${I:-4}"; N="${N:-64}"
FITNESS="${FITNESS:-}"; OPS="de-gen ga-gen ux-bf"; FIT_ARGS=(); PROB_ARGS=()
[ -n "$FITNESS" ] && { OPS="de-gen ga-gen"; FIT_ARGS=(--fitness "$FITNESS"); PROB_ARGS=(--problem "$FITNESS"); }
[ "$FITNESS" = knapsack ] && OPS="ux-bf"
OPS="${ONLY_OPS:-$OPS}"
QD="${QD:-1024}"; QDBIN="${QDBIN:-4096}"
read -r -a KARGS <<< "${KARGS:-}"; read -r -a PARGS <<< "${PARGS:-}"
CONFIGS="${CONFIGS:-s1 highdim 1 128
s8 highdim 8 32
s8defer highdim 8 32 --defer-always
v4s1 highdim_vec4 1 32
v4s8 highdim_vec4 8 32${EXTRA_CONFIGS:+
$EXTRA_CONFIGS}}"
mkdir -p "$OUT"

for op in $OPS; do
  D=$QD; [ "$op" = ux-bf ] && D=$QDBIN
  for ((sd = SEED0; sd < SEEDS; sd++)); do
    seed=$(( 42 + sd * 1000003 ))
    while read -r cfg lay cap tx extra; do
      "$BIN" --mode generation --algo "$op" --layout "$lay" --islands $I --n $N --d $D \
        --xthreads "$tx" --yblock 1 --hd-cap "$cap" $extra "${FIT_ARGS[@]}" "${KARGS[@]}" --generations "$GENS" --seed "$seed" \
        --trace "$OUT/${op}_${cfg}_${sd}.csv" > /dev/null || echo "FAIL $op $cfg $sd"
    done <<< "$CONFIGS"
    (cd "$SB/python" && PYTHONPATH=.${EVOX_SRC:+:$EVOX_SRC} python -m step_bench.bench_generation \
      --op "$op" --n $(( I * N )) --islands $I --d $D "${PROB_ARGS[@]}" "${PARGS[@]}" --generations "$GENS" --seed "$sd" \
      --trace "$OUT/${op}_evox_${sd}.csv" 2>/dev/null) || echo "FAIL $op evox $sd"
  done
  echo "[$(date '+%T')] $op done"
done
