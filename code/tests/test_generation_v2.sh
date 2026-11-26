#!/usr/bin/env bash
# Test the decomposed generation pipeline against the fully-fused one.
#
#   decomposed = SelectionKernel | FusedVariation_* | ReplacementKernel
#
# For every (algo, layout): functional `--check` on both, then a fused-vs-
# decomposed comparison of the final fitness (`--dump-fit`). The equality is
# reported (per the "functional gate + reported equality" policy), not hard-failed
# if a future selection kernel changes its draws.
#
# Env: ISLANDS=4 N=512 D=256 GENS=20
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin/bench_generation_v2"
ISLANDS="${ISLANDS:-4}"; N="${N:-512}"; D="${D:-256}"; GENS="${GENS:-20}"
TMP="$(mktemp -d)"
fail=0

echo "decomposed-pipeline test: islands=$ISLANDS n=$N d=$D gens=$GENS"
for algo in de-gen ga-gen ux-bf; do
  for lay in individual individual_vec4 highdim highdim_vec4; do
    base=(--algo "$algo" --layout "$lay" --islands "$ISLANDS" --n "$N" --d "$D")

    fchk="$("$BIN" --mode generation "${base[@]}" --generations "$GENS" --warmup 2 --check 2>&1)"
    dchk="$("$BIN" --mode decomposed "${base[@]}" --generations "$GENS" --warmup 2 --check 2>&1)"
    f_ok=0; d_ok=0
    case "$fchk" in *"check PASS"*) f_ok=1;; esac
    case "$dchk" in *"check PASS"*) d_ok=1;; esac
    [ "$f_ok" = 1 ] || { echo "  FAIL fused check: $algo $lay"; fail=1; }
    [ "$d_ok" = 1 ] || { echo "  FAIL decomposed check: $algo $lay"; fail=1; }

    "$BIN" --mode generation "${base[@]}" --generations "$GENS" --warmup 0 --dump-fit "$TMP/a" >/dev/null 2>&1
    "$BIN" --mode decomposed "${base[@]}" --generations "$GENS" --warmup 0 --dump-fit "$TMP/b" >/dev/null 2>&1
    eq="DIFF"
    if cmp -s "$TMP/a" "$TMP/b"; then eq="IDENTICAL"; fi

    printf "  %-7s %-16s fused=%-4s decomposed=%-4s equality=%s\n" \
      "$algo" "$lay" "$([ "$f_ok" = 1 ] && echo ok || echo FAIL)" \
      "$([ "$d_ok" = 1 ] && echo ok || echo FAIL)" "$eq"
  done
done

rm -rf "$TMP"
if [ "$fail" = 0 ]; then echo "ALL PASS"; else echo "FAILURES"; fi
exit "$fail"
