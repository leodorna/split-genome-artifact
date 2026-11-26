#!/usr/bin/env bash
# Extract register counts for the v2 kernels by compiling the benchmark app with
# `-Xptxas -v` and parsing the output into results/kernel_regs_v2.csv.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/code/apps/bench_generation_v2.cpp"
# CUDA_ARCH (default 120) selects the target; REGS_OUT (default results/kernel_regs_v2.csv)
# the output table, so a second GPU does not overwrite the sm_120 table of the paper.
CUDA_ARCH="${CUDA_ARCH:-120}"
REGS_OUT="${REGS_OUT:-$ROOT/results/kernel_regs_v2.csv}"
RAW="${REGS_OUT%.csv}.raw.txt"
mkdir -p "$(dirname "$REGS_OUT")"
cd "$ROOT"

nvcc -std=c++20 -x cu -diag-suppress 3056 -O3 -arch=sm_${CUDA_ARCH} -DTARGET_SM=${CUDA_ARCH}0 \
  -Xptxas -v \
  -Icode/include -I"${MATHDX_DIR:-code/external/mathdx/25.12/include}" \
  -c "$SRC" -o "${TMPDIR:-/tmp}/step_bench_v2_regs_sm${CUDA_ARCH}.o" 2> "$RAW"

python3 "$ROOT/scripts/parse_kernel_regs_v2.py" "$RAW" "$REGS_OUT"
