#!/usr/bin/env bash
# Donor probe: where does the time of de-gen go?
# Builds bench_generation_v2 three times (as is; with constant bounds; with constant
# bounds and one donor row instead of three) and times unsplit launches (s = 1, t_y = 1)
# of de-gen and ga-gen on the nine cells, scalar and vec4.
#   results/donor_probe/donor_probe.csv
#   variant,op,layout,I,N,D,tx,seed,rep,time_us
# Env: CUDA_ARCH (120), MATHDX_DIR (code/external/mathdx/25.12/include), SEEDS (3), REPS (10),
#      WARMUP (5), TXS ("64 512"), OUT.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SB="$ROOT"
ARCH="${CUDA_ARCH:-120}"; MATHDX="${MATHDX_DIR:-$ROOT/code/external/mathdx/25.12/include}"
SEEDS="${SEEDS:-3}"; REPS="${REPS:-10}"; WARMUP="${WARMUP:-5}"; TXS="${TXS:-64 512}"
OUTDIR="$SB/results/donor_probe"; OUT="${OUT:-$OUTDIR/donor_probe.csv}"
mkdir -p "$OUTDIR" "$ROOT/code/bin"
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "GPU is in use; set FORCE=1 to run anyway" >&2; exit 1
fi
FL="-std=c++20 -x cu -diag-suppress 3056 --compiler-options -O3 -arch=sm_$ARCH -DTARGET_SM=${ARCH}0 \
  -I$ROOT/code/include -I$MATHDX"
SRC="$ROOT/code/apps/bench_generation_v2.cpp"
nvcc $FL -o "$ROOT/code/bin/donor_probe_base" "$SRC" &
nvcc $FL -DPROBE_CONST_BOUNDS -o "$ROOT/code/bin/donor_probe_constb" "$SRC" &
nvcc $FL -DPROBE_CONST_BOUNDS -DPROBE_ONE_DONOR -o "$ROOT/code/bin/donor_probe_onedonor" "$SRC" &
wait
for b in base constb onedonor; do [ -x "$ROOT/code/bin/donor_probe_$b" ] || { echo "build failed: $b" >&2; exit 1; }; done
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvcc --version | tail -1; date -u '+%F %T UTC'; } > "$OUTDIR/RUN.txt"
echo "variant,op,layout,I,N,D,tx,seed,rep,time_us" > "$OUT"
TMP="$(mktemp)"
for cell in "1 64 16777216" "1 256 4194304" "1 1024 1048576" "32 64 524288" "32 256 131072" \
            "32 1024 32768" "1024 64 16384" "1024 256 4096" "1024 1024 1024"; do
  set -- $cell; I=$1; N=$2; D=$3
  for op in de-gen ga-gen; do for lay in highdim highdim_vec4; do for tx in $TXS; do
    for v in base constb onedonor; do
      [ "$v" = onedonor ] && [ "$op" != de-gen ] && continue
      for sd in $(seq 0 $((SEEDS - 1))); do
        : > "$TMP"
        "$ROOT/code/bin/donor_probe_$v" --mode generation --algo "$op" --layout "$lay" --islands "$I" --n "$N" \
          --d "$D" --xthreads "$tx" --yblock 1 --hd-cap 1 --reps "$REPS" --warmup "$WARMUP" \
          --seed $((42 + sd * 1000003)) --dump-times "$TMP" >/dev/null 2>&1 || { echo "$v,$op,$lay,$I,$N,$D,$tx,$sd,fail," >> "$OUT"; continue; }
        awk -v p="$v,$op,$lay,$I,$N,$D,$tx,$sd" '{print p "," NR-1 "," $1}' "$TMP" >> "$OUT"
      done
    done
  done; done; done
  echo "$(date -u '+%T') done I=$I N=$N" >> "$OUTDIR/progress.log"
done
rm -f "$TMP"; date -u '+%F %T UTC end' >> "$OUTDIR/RUN.txt"
