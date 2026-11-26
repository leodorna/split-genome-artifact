#!/usr/bin/env bash
# Transfer campaign: do the ablation's guidelines and the launch heuristic (derived on
# Sphere and OneMax) carry over to harder functions? Per function, into
# results/transfer/<f>/:
#   phase3_<f>.csv   every s and tx at ty = 1 on the nine ablation cells plus three
#                    D = 1000 cells (configs/sweep_transfer.json), SEEDS seeds
#   evox/            island-local compiled baseline on the same cells
#   quality/         convergence traces, ours vs the baseline (quality_check.sh); the
#                    CEC functions at D = 1000 with the suite's official shifts
#   roofline/        ncu profiles at the heuristic's configurations (STEPS ncu)
# Functions: cec-f1, cec-f2, cec-f3, cec-f12 (de-gen, ga-gen) and knapsack (ux-bf,
# strongly correlated instance). The sweeps use a generated shift (timing does not
# depend on it); the quality runs use data/cec2013lsgo (scripts/fetch_cec2013_shifts.sh).
#
# The steps run in phases across functions, so that the timing data (kernels, evox)
# of every function is complete before the slower quality and profiling steps.
# Env: FITS, STEPS, TR (output dir), SWEEP_JSON, (kernels evox quality ncu), SEEDS (5), EVOX_SEEDS (10),
#      QSEEDS (10), QGENS (1000), BIN (default code/bin/bench_generation_v2).
# Run inside blackwell_env, idle GPU.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SB="$(cd "$HERE/.." && pwd)"
TR="${TR:-$SB/results/transfer}"
SWEEP_JSON="${SWEEP_JSON:-$SB/configs/sweep_transfer.json}"
FITS="${FITS:-cec-f1 cec-f2 cec-f3 cec-f12 knapsack}"
STEPS="${STEPS:-kernels evox quality ncu}"
SHIFTS="$SB/data/cec2013lsgo"
export BIN="${BIN:-$SB/code/bin/bench_generation_v2}"
log() { echo "[$(date '+%F %T')] $*"; }
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
ops_for() { [ "$1" = knapsack ] && echo ux-bf || echo "de-gen ga-gen"; }

[ -x "$BIN" ] || { echo "missing $BIN (make -C code bench)"; exit 2; }
if has quality; then
  for f in 1 2 3 12; do
    [ -s "$SHIFTS/F$f-xopt.txt" ] || { echo "missing $SHIFTS/F$f-xopt.txt (scripts/fetch_cec2013_shifts.sh)"; exit 2; }
  done
fi
mkdir -p "$TR"
log "transfer campaign: FITS=($FITS) STEPS=($STEPS) BIN=$BIN"
git -C "$SB" rev-parse HEAD > "$TR/artifact_rev.txt" 2>/dev/null
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem --format=csv > "$TR/gpu.txt" 2>/dev/null

for fit in $FITS; do
  F="$TR/$fit"; mkdir -p "$F"
  if has kernels && [ ! -s "$F/phase3_$fit.csv.done" ]; then
    log "$fit: kernels"
    OPS="$(ops_for "$fit")" FITNESS="$fit" SWEEP="$SWEEP_JSON" LAYOUTS="highdim highdim_vec4" \
      CAPS="${CAPS:-1 2 4 8 16 32 64 128 256 512 1024 2048}" TXS="${TXS:-32 64 128 256 512}" TYS="1" \
      DEDUP_CAPS=1 SEEDS="${SEEDS:-5}" REPS=10 WARMUP=5 OUT="$F/phase3_$fit.csv" PROGRESS="$F/progress.log" \
      bash "$HERE/phase3_hybrid_probe_v2.sh" && date > "$F/phase3_$fit.csv.done"
  fi
  if has evox && [ ! -s "$F/evox/.done" ]; then
    log "$fit: evox"
    OPS="$(ops_for "$fit")" ONLY_ISLANDS=1 EXTRA="--problem $fit" SEEDS="${EVOX_SEEDS:-10}" GENS=20 WARMUP=5 \
      SWEEP="$SWEEP_JSON" OUTDIR="$F/evox" bash "$HERE/evox_islands_sweep.sh" \
      && date > "$F/evox/.done"
  fi
done

if has quality; then
  for fit in $FITS; do
    F="$TR/$fit"
    [ -s "$F/quality/.done" ] && continue
    log "$fit: convergence"
    # one more configuration than the paper's quality check: split rows with
    # multi-warp blocks (s = 4, tx = 128; knapsack's 128 words: s = 2, tx = 64), which
    # exercise the cross-block finalize between warps
    extra="s4w highdim 4 128"
    if [ "$fit" = knapsack ]; then
      FITNESS=knapsack QDBIN=4096 SEEDS="${QSEEDS:-10}" GENS="${QGENS:-1000}" OUT="$F/quality" \
        EXTRA_CONFIGS="s2w highdim 2 64" bash "$HERE/quality_check.sh"
    else
      n="${fit#cec-f}"
      FITNESS="$fit" QD=1000 KARGS="--shift-file $SHIFTS/F$n-xopt.txt" PARGS="--shift-file $SHIFTS/F$n-xopt.txt" \
        SEEDS="${QSEEDS:-10}" GENS="${QGENS:-1000}" OUT="$F/quality" EXTRA_CONFIGS="$extra" \
        bash "$HERE/quality_check.sh"
    fi
    date > "$F/quality/.done"
  done
fi

if has ncu; then
  for fit in $FITS; do
    F="$TR/$fit"; mkdir -p "$F/roofline/raw"
    [ -s "$F/roofline/.done" ] && continue
    log "$fit: ncu at the heuristic's configurations"
    if [ "$fit" = knapsack ]; then sel='$2 == "ux-bf"'; else sel='$2 != "ux-bf"'; fi
    python "$HERE/roofline_configs_v3.py" --traj "" | awk "$sel" > "$F/roofline/configs.txt"
    FITNESS="$fit" RAWDIR="$F/roofline/raw" bash "$HERE/ncu_roofline_v3.sh" < "$F/roofline/configs.txt"
    cp "$SB"/results/rerun_v3/roofline/raw/ceil_*.csv "$SB/results/rerun_v3/roofline/raw/ceilings_events.csv" \
      "$F/roofline/raw/" 2>/dev/null
    python "$HERE/parse_roofline_v3.py" --dir "$F/roofline" && date > "$F/roofline/.done"
  done
fi
log "ALL DONE"
