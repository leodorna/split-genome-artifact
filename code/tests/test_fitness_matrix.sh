#!/usr/bin/env bash
# Functional check of every fitness of the split-genome kernel: Sphere, Rastrigin,
# Rosenbrock, CEC 2013 LSGO f1, f2, f3, f12 (de-gen, ga-gen), OneMax and the 0/1
# knapsack (ux-bf).
#
# Every run uses `--check`: after each generation the host recomputes the fitness of the
# whole population in double precision and compares it with the kernel's value, and
# the best fitness must not get worse (elitist replacement). The matrix covers one
# block per row (s = 1), many blocks per row, a grid-stride cap that does not divide
# the work, t_y > 1, dimensions whose unit count does not fill the last block (and,
# for the scalar layout, is not a multiple of 4), and a large dimension. Layout
# constraints: vec4 needs D % 4 == 0; binary D % 32 == 0, binary vec4 D % 128 == 0.
#
# Env: SHIFT_DIR=<dir with F1-xopt.txt ... F12-xopt.txt> adds D = 1000 runs with the
#      official shifts.  GENS=10 ISLANDS=2 N=64
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin/bench_generation_v2"
GENS="${GENS:-10}"; ISLANDS="${ISLANDS:-2}"; N="${N:-64}"
SHIFT_DIR="${SHIFT_DIR:-}"
pass=0; fail=0

run() {  # run <label> <args...>
  local label="$1"; shift
  local out
  out="$("$BIN" --mode generation --islands "$ISLANDS" --n "$N" --generations "$GENS" --warmup 0 --check "$@" 2>&1)"
  case "$out" in
    *"check PASS"*) pass=$((pass + 1)) ;;
    *) fail=$((fail + 1)); echo "FAIL $label: $*"; echo "$out" | tail -3 | sed 's/^/    /' ;;
  esac
}

# name|args shared by the continuous and binary matrices
CONFIGS=(
  "s=1|--hd-cap 1"
  "many-blocks|--xthreads 32"
  "grid-stride|--xthreads 32 --hd-cap 3"
  "ty=4|--xthreads 64 --yblock 4"
)

for fit in sphere rastrigin rosenbrock cec-f1 cec-f2 cec-f3 cec-f12; do
  for algo in de-gen ga-gen; do
    for lay in highdim highdim_vec4; do
      dims="1000 1002"; [ "$lay" = highdim_vec4 ] && dims="1000 1004"
      for d in $dims; do
        for cfg in "${CONFIGS[@]}"; do
          run "$fit $algo $lay d=$d ${cfg%%|*}" --fitness "$fit" --algo "$algo" --layout "$lay" --d "$d" ${cfg#*|}
        done
      done
      run "$fit $algo $lay d=2^20 default" --fitness "$fit" --algo "$algo" --layout "$lay" --d 1048576 --generations 3
      if [ -n "$SHIFT_DIR" ] && [ "${fit#cec-}" != "$fit" ]; then
        n="${fit#cec-f}"
        run "$fit $algo $lay d=1000 official" --fitness "$fit" --algo "$algo" --layout "$lay" --d 1000 \
            --shift-file "$SHIFT_DIR/F$n-xopt.txt" --xthreads 32
      fi
    done
  done
  echo "$fit done: pass=$pass fail=$fail"
done

for kp in onemax strong uncorrelated; do
  if [ "$kp" = onemax ]; then FIT=(); else FIT=(--fitness knapsack --kp-type "$kp"); fi
  for lay in highdim highdim_vec4; do
    dims="1056 4096"; [ "$lay" = highdim_vec4 ] && dims="1152 4096"
    for d in $dims; do
      for cfg in "${CONFIGS[@]}"; do
        run "ux-bf/$kp $lay d=$d ${cfg%%|*}" "${FIT[@]}" --algo ux-bf --layout "$lay" --d "$d" ${cfg#*|}
      done
    done
    run "ux-bf/$kp $lay d=2^22 default" "${FIT[@]}" --algo ux-bf --layout "$lay" --d 4194304 --generations 3
  done
done
echo "ux-bf done: pass=$pass fail=$fail"

if [ "$fail" = 0 ]; then echo "ALL PASS ($pass runs)"; else echo "FAILURES: $fail of $((pass + fail))"; fi
exit $((fail > 0))
