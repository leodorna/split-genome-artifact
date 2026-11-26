#!/usr/bin/env bash
# Kernel sweep over the split factor s (cap), t_x and t_y, with the block-per-individual
# layouts as an option. The cells come from SWEEP (default configs/sweep_realistic.json,
# I*N*D = 2^30); REPRODUCE.md gives the settings of the paper's ablation. Writes OUT
# (default results/phase3_sweep.csv):
#   op,layout,cap,I,N,D,total,rows,NO,nUnits,nCouples,tx,ty,threads,regs,smem,status,seed,rep,time_us
#
# highdim* layouts sweep ty in {1,2,4,8} and cap in CAPS; individual* layouts use
# cap=0 and sweep ty. Feasibility mirrors phase3_full_ablation_v2.sh plus
# threads*regs <= 65536 (raised highdim register counts after U1).
#
# Env: REGS (register table, default results/kernel_regs_v2.csv), SWEEP, CAPS, TXS, TYS, SEEDS, REPS, WARMUP, OUT, OPS (de-gen ga-gen ux-bf),
#      FITNESS (passed as --fitness when set: rastrigin | rosenbrock).
#      DEDUP_CAPS=1 skips (status skip_dup) a highdim cap whose half already reaches
#      ceil(nUnits/tx): the kernel launches min(cap, ceil(nUnits/tx)) blocks per row, so
#      every larger cap repeats the launch of the first cap >= ceil(nUnits/tx).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/code/bin/bench_generation_v2}"
RES="$ROOT/results"
SWEEP="${SWEEP:-$ROOT/configs/sweep_realistic.json}"
REGS="${REGS:-$RES/kernel_regs_v2.csv}"
CAPS="${CAPS:-1 2 4 8 16 32 64 128 256}"
TXS="${TXS:-128 256 512}"
TYS="${TYS:-1 2 4 8}"
LAYOUTS="${LAYOUTS:-individual individual_vec4 highdim highdim_vec4}"
SEEDS="${SEEDS:-10}"; REPS="${REPS:-10}"; WARMUP="${WARMUP:-5}"; SEED="${SEED:-42}"
SEED_STEP="${SEED_STEP:-1000003}"
OUT="${OUT:-$RES/phase3_sweep.csv}"
ITER_CAP="${ITER_CAP:-262144}"; CFG_TIMEOUT="${CFG_TIMEOUT:-180}"
PROGRESS="${PROGRESS:-${OUT%.csv}.progress.log}"
OPS="${OPS:-de-gen ga-gen ux-bf}"
FIT_ARGS=(); [ -n "${FITNESS:-}" ] && FIT_ARGS=(--fitness "$FITNESS")
TMP="$(mktemp "${TMPDIR:-/tmp}/hybrid_times.XXXXXX")"
mkdir -p "$RES"

mapfile -t TUPLES < <(python3 "$ROOT/scripts/sweep_tuples.py" --config "$SWEEP")

regs_for() { # op layout
  local tok; case $1 in de-gen) tok=DE1bin;; ga-gen) tok=SBXCrossover;; ux-bf) tok=UX;; esac
  local lay; case $2 in highdim*) lay=highdim;; *) lay=individual;; esac
  local v4=0; case $2 in *_vec4) v4=1;; esac
  awk -F, -v k="$tok" -v l="$lay" -v v="$v4" '$1==k && $2==l && $3==v {print $4; exit}' "$REGS" | tr -d "\r"
}

HDR="op,layout,cap,I,N,D,total,rows,NO,nUnits,nCouples,tx,ty,threads,regs,smem,status,seed,rep,time_us"
echo "$HDR" > "$OUT"
emit_skip() { # op lay cap I N D total rows NO nu nc tx ty threads R smem status
  echo "$1,$2,$3,$4,$5,$6,$7,$8,$9,${10},${11},${12},${13},${14},${15},${16},${17},,," >> "$OUT"
}

: > "$PROGRESS"
for op in $OPS; do
  case $op in de-gen) NO=1;; *) NO=2;; esac
  case $op in ux-bf) binary=1;; *) binary=0;; esac
  for lay in $LAYOUTS; do
    vec4=0; case $lay in *_vec4) vec4=1;; esac
    hd=0; case $lay in highdim*) hd=1;; esac
    if [ $vec4 -eq 1 ]; then
      if [ $binary -eq 1 ]; then req=128; else req=4; fi
    else req=0; fi
    R=$(regs_for "$op" "$lay"); [ -z "$R" ] && R=255
    if [ $hd -eq 1 ]; then CAPLIST="$CAPS"; else CAPLIST="0"; fi
    printf '== %s / %s (R=%s) ==\n' "$op" "$lay" "$R" >> "$PROGRESS"
    for t in "${TUPLES[@]}"; do
      I=${t%% *}; rest=${t#* }; N=${rest% *}; D=${rest#* }
      # Tuples fix 2^30 32-bit words per population; a binary genome packs 32
      # genes per word, so its D (in bits) is 32x the tuple's.
      [ $binary -eq 1 ] && D=$(( D * 32 ))
      total=$(( I * N * D )); rows=$(( I * N ))
      if [ $binary -eq 1 ]; then
        if [ $vec4 -eq 1 ]; then nu=$(( D / 128 )); else nu=$(( D / 32 )); fi
      else
        if [ $vec4 -eq 1 ]; then nu=$(( D / 4 )); else nu=$(( D )); fi
      fi
      nc=$(( N / NO ))
      for cap in $CAPLIST; do
        for tx in $TXS; do
          for ty in $TYS; do
            threads=$(( tx * ty )); smem=$(( threads * 4 ))
            if [ $vec4 -eq 1 ] && [ $(( D % req )) -ne 0 ]; then
              emit_skip "$op" "$lay" "$cap" "$I" "$N" "$D" "$total" "$rows" "$NO" "$nu" "$nc" "$tx" "$ty" "$threads" "$R" "$smem" skip_vec4; continue
            fi
            if [ $(( N % NO )) -ne 0 ] || [ "$nc" -gt 65535 ] || [ "$I" -gt 65535 ]; then
              emit_skip "$op" "$lay" "$cap" "$I" "$N" "$D" "$total" "$rows" "$NO" "$nu" "$nc" "$tx" "$ty" "$threads" "$R" "$smem" skip_grid; continue
            fi
            [ "$tx" -gt "$nu" ] && { emit_skip "$op" "$lay" "$cap" "$I" "$N" "$D" "$total" "$rows" "$NO" "$nu" "$nc" "$tx" "$ty" "$threads" "$R" "$smem" skip_tx; continue; }
            [ "$ty" -gt "$nc" ] && { emit_skip "$op" "$lay" "$cap" "$I" "$N" "$D" "$total" "$rows" "$NO" "$nu" "$nc" "$tx" "$ty" "$threads" "$R" "$smem" skip_ty; continue; }
            [ "$threads" -gt 1024 ] && { emit_skip "$op" "$lay" "$cap" "$I" "$N" "$D" "$total" "$rows" "$NO" "$nu" "$nc" "$tx" "$ty" "$threads" "$R" "$smem" skip_threads; continue; }
            [ $(( threads * R )) -gt 65536 ] && { emit_skip "$op" "$lay" "$cap" "$I" "$N" "$D" "$total" "$rows" "$NO" "$nu" "$nc" "$tx" "$ty" "$threads" "$R" "$smem" skip_regs; continue; }
            if [ $hd -eq 1 ] && [ "${DEDUP_CAPS:-0}" = 1 ] && [ "$cap" -gt 1 ] && [ $(( cap / 2 )) -ge $(( (nu + tx - 1) / tx )) ]; then
              emit_skip "$op" "$lay" "$cap" "$I" "$N" "$D" "$total" "$rows" "$NO" "$nu" "$nc" "$tx" "$ty" "$threads" "$R" "$smem" skip_dup; continue
            fi
            if [ $hd -eq 1 ]; then
              gx=$(( (nu + tx - 1) / tx )); [ "$gx" -gt "$cap" ] && gx=$cap
              iters=$(( (nu + gx * tx - 1) / (gx * tx) ))
            else
              iters=$(( (nu + tx - 1) / tx ))
            fi
            [ "$iters" -gt "$ITER_CAP" ] && { emit_skip "$op" "$lay" "$cap" "$I" "$N" "$D" "$total" "$rows" "$NO" "$nu" "$nc" "$tx" "$ty" "$threads" "$R" "$smem" skip_iters; continue; }

            BASE="$op,$lay,$cap,$I,$N,$D,$total,$rows,$NO,$nu,$nc,$tx,$ty,$threads,$R,$smem"
            for (( sd=0; sd<SEEDS; sd++ )); do
              seed=$(( SEED + sd * SEED_STEP ))
              : > "$TMP"
              timeout "$CFG_TIMEOUT" "$BIN" --mode generation --algo "$op" --layout "$lay" --islands "$I" --n "$N" --d "$D" \
                     --xthreads "$tx" --yblock "$ty" --hd-cap "$cap" --reps "$REPS" --warmup "$WARMUP" \
                     --seed "$seed" --dump-times "$TMP" "${FIT_ARGS[@]}" >/dev/null 2>&1
              rc=$?
              if [ $rc -eq 124 ]; then echo "$BASE,timeout,$sd,," >> "$OUT"; continue; fi
              if [ $rc -ne 0 ] || [ ! -s "$TMP" ]; then echo "$BASE,fail,$sd,," >> "$OUT"; continue; fi
              rep=0
              while IFS= read -r us; do
                echo "$BASE,ok,$sd,$rep,$us" >> "$OUT"; rep=$(( rep + 1 ))
              done < "$TMP"
            done
          done
        done
      done
    done
    echo "$op $lay done: $(wc -l < "$OUT") rows"
  done
done

rm -f "$TMP"
echo "wrote $OUT ($(wc -l < "$OUT") rows)"
