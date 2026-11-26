#!/usr/bin/env bash
# Short GPU check of the artifact (about 5 minutes): builds the kernels, runs their tests,
# times one configuration of the paper and runs a tiny version of the kernel sweep and,
# if torch and evox are importable, of the EvoX baseline. Outputs go to smoke/.
# Env: CUDA_ARCH (120), MATHDX_DIR (code/external/mathdx/25.12/include), EVOX_SRC (optional),
#      PYTHON (python3).
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${CUDA_ARCH:-120}"
PY="${PYTHON:-python3}"
export MATHDX_DIR="${MATHDX_DIR:-$PWD/code/external/mathdx/25.12/include}"
mkdir -p smoke

# the kernels run only on the architecture they are built for
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.[:space:]')
if [ -n "$cc" ] && [ "$cc" != "$ARCH" ]; then
  echo "CUDA_ARCH=$ARCH, but this GPU is sm_$cc: run with CUDA_ARCH=$cc"; exit 1
fi

echo "== build and test the kernels"
make -C code CUDA_ARCH="$ARCH" MATHDX_DIR="$MATHDX_DIR" bench roofline
make -C code CUDA_ARCH="$ARCH" MATHDX_DIR="$MATHDX_DIR" test | tail -3

echo; echo "== one configuration of the main ablation (ga-gen vec4, I=32, N=256, D=2^17, s=1, t_x=512, t_y=1)"
code/bin/bench_generation_v2 --mode generation --algo ga-gen --layout highdim_vec4 \
  --islands 32 --n 256 --d 131072 --xthreads 512 --yblock 1 --hd-cap 1 --reps 10 --warmup 5 --seed 42 \
  | tail -1 | awk -F, '{printf "median time per generation: %s us (paper, RTX PRO 6000: 7134 us)\n", $NF}'

echo; echo "== tiny kernel sweep (one cell, two split factors, one seed)"
cat > smoke/sweep_one_cell.json <<'J'
{"description": "smoke test", "invariant": 1073741824, "tuples": [[32, 256, 131072]]}
J
OPS="ga-gen" LAYOUTS="highdim highdim_vec4" SWEEP=smoke/sweep_one_cell.json CAPS="1 8" TXS="256 512" TYS="1" \
  SEEDS=1 REPS=3 WARMUP=2 OUT=smoke/sweep.csv PROGRESS=smoke/progress.log \
  bash scripts/phase3_hybrid_probe_v2.sh
grep -c ",ok," smoke/sweep.csv | xargs echo "timed repetitions:"

echo; echo "== tiny EvoX baseline (same cell, island-local, one seed)"
if PYTHONPATH="python${EVOX_SRC:+:$EVOX_SRC}" "$PY" -c "import torch, evox" 2>/dev/null; then
  (cd python && PYTHONPATH=".${EVOX_SRC:+:$EVOX_SRC}" "$PY" -m step_bench.bench_generation --op ga-gen \
     --n 8192 --d 131072 --islands 32 --seed 0 --generations 5 --warmup 2 --compiled-only) | tee smoke/evox.csv
else
  echo "torch or evox not importable: skipped (see README, 'Python environment')"
fi
echo; echo "smoke test done"
