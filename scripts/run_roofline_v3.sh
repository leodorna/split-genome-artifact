#!/usr/bin/env bash
# Roofline experiment (instruction roofline), all steps. Run inside blackwell_env
# with nothing else on the GPU (checked; FORCE=1 skips the check).
#   ceilings  roofline_ceilings_v3 timed and profiled      (ncu_ceilings_v3.sh)
#   configs   heuristic, split-factor paths, best configs  (roofline_configs_v3.py)
#   profiles  every config under ncu, resumable            (ncu_roofline_v3.sh)
#   parse     roofline_v3.csv, roofline_ceilings_v3.csv    (parse_roofline_v3.py)
#   plot      fig_roofline_v3.{pdf,png}, roofline_table_v3.csv (plot_roofline_v3.py)
# Env: OUT (results/rerun_v3/roofline), STEPS (all five), BETA, USTAR (heuristic
#      constants), TRAJ ("1,64 1,256"), BEST (ablation CSV for the best configs, or
#      empty), WARMUP (3), FORCE.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="${OUT:-$ROOT/results/rerun_v3/roofline}"
STEPS="${STEPS:-ceilings configs profiles parse plot}"
BETA="${BETA:-1}"; USTAR="${USTAR:-16}"; TRAJ="${TRAJ:-1,64 1,256}"
BEST="${BEST-$ROOT/results/rerun_v3/phase3_unified_final_v3.csv}"
export WARMUP="${WARMUP:-3}"
export RAWDIR="$OUT/raw"
mkdir -p "$RAWDIR"
log() { echo "[$(date '+%F %T')] $*"; }
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }

if [ "${FORCE:-0}" != 1 ] && [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "another process is using the GPU; refusing to profile (FORCE=1 to override)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  exit 1
fi
[ -x "$ROOT/code/bin/roofline_ceilings_v3" ] || (cd "$ROOT/code" && make roofline) || exit 1

{
  echo "date: $(date '+%F %T')"
  echo "bench_generation_v2: $(sha256sum "$ROOT/code/bin/bench_generation_v2" | cut -c1-16) $(stat -c %y "$ROOT/code/bin/bench_generation_v2")"
  echo "ncu: $(ncu --version | tail -1)"
  echo "heuristic: beta=$BETA ustar=$USTAR; traj cells: $TRAJ; best from: ${BEST:-none}"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem --format=csv,noheader
} > "$OUT/RUN.txt"

if has ceilings; then log "ceilings"; bash "$HERE/ncu_ceilings_v3.sh"; fi
if has configs; then
  log "configs"
  bargs=(); [ -n "$BEST" ] && bargs=(--best "$BEST")
  python "$HERE/roofline_configs_v3.py" --beta "$BETA" --ustar "$USTAR" --traj "$TRAJ" "${bargs[@]}" \
    > "$OUT/configs.txt" || exit 1
  log "$(wc -l < "$OUT/configs.txt") configs"
fi
if has profiles; then log "profiles"; bash "$HERE/ncu_roofline_v3.sh" < "$OUT/configs.txt"; fi
if has parse; then log "parse"; python "$HERE/parse_roofline_v3.py" --dir "$OUT"; fi
if has plot; then log "plot"; python "$HERE/plot_roofline_v3.py" --dir "$OUT"; fi
log "done"
