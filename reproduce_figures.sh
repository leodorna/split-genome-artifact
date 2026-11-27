#!/usr/bin/env bash
# Regenerate every figure, table and number of the paper's results section from the raw
# data in results/. Needs Python 3 with numpy, scipy, matplotlib, seaborn and statsmodels
# (requirements-analysis.txt); no GPU. Takes a few minutes, most of it the main ablation
# (1.1 million timings).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
R3=results/rerun_v3; A=$R3/analysis; G2=results/gpu2
mkdir -p "$A" "$G2/analysis"

step() { echo; echo "== $*"; }

step "unpack the timing tables"
[ -f $R3/phase3_unified_final_v3.csv ] || gunzip -k $R3/phase3_unified_final_v3.csv.gz
[ -f $G2/phase3_gpu2_all.csv ]        || gunzip -k $G2/phase3_gpu2_all.csv.gz
F=$R3/phase3_unified_final_v3.csv

step "Figure 3 and Equation 13: penalties P_k of the 54 cases"
$PY scripts/sensitivity_pmnf.py --final $F --dir $A > $A/sensitivity.txt
$PY scripts/plot_sensitivity_heatmap.py --coeffs $A/sensitivity_coeffs_pmnf.csv --outdir $A

step "Figure 4: near-optimal levels and the heuristic's choices"
$PY scripts/plot_sensitivity_intervals.py --final $F --out $A/fig_sensitivity_intervals | tee $A/sensitivity_intervals.txt

step "Table 3 (left): regret of the launch heuristic (Equation 14); Section S5"
$PY scripts/launch_heuristic.py --final $F --sens $A/sensitivity_coeffs_pmnf.csv --evox $R3/evox_autotune --dir $A \
  > $A/launch_heuristic.txt
$PY scripts/heuristic_loo.py --final $F --dir $A > $A/heuristic_loo.txt

step "Figure 5: instruction roofline"
$PY scripts/plot_roofline_v3.py --dir $R3/roofline --no-l2 --out fig_roofline_v3_nol2

step "Solution quality against the baselines, Sphere/OneMax (Figure 7 is drawn with the transfer step below)"
$PY scripts/analyze_quality.py --dir $R3/quality > $A/quality_sphere_onemax.txt

step "Figure 8 and the large-D block of Table S3: time to quality at I=1, N=64, D=2^20"
$PY scripts/analyze_quality.py --dir results/largeD/quality > results/largeD/quality_stdout.txt
$PY scripts/plot_time_to_quality.py --dir results/largeD > results/largeD/time_to_quality.txt
# search progress (share of the total improvement in the last 100/500 generations), Section 4.2
$PY scripts/quality_progress.py > $A/quality_progress.txt

step "Figure 6, Table 4 and Table S1: comparison with EvoX (baselines: torch.compile max-autotune-no-cudagraphs)"
$PY scripts/plot_evox_comparison.py --final $F --evox $R3/evox_autotune --outdir $A > $A/evox_comparison.txt
$PY scripts/gen_evox_config_table.py --csv $A/augmented_vs_evox.csv --tex $A/evox_launch_configs.tex
# the same comparison with the default compile mode, for Table S4
mkdir -p $A/default_mode
$PY scripts/plot_evox_comparison.py --final $F --evox $R3/evox --outdir $A/default_mode > $A/default_mode/evox_comparison.txt
$PY scripts/analyze_compile_modes.py --dir $R3 > $A/compile_modes.txt

step "Table 3 (right) and Table S5: the second GPU (RTX A6000, 84 SMs)"
$PY scripts/score_gpu2.py --gpu2 $G2/phase3_gpu2_all.csv --ref $F --out $G2 > $G2/analysis/score_gpu2.txt
# autotuned baselines on both GPUs when the A6000 run exists (scripts/gpu2/run_evox.sh with
# COMPILE_MODE=max-autotune-no-cudagraphs), otherwise the default mode on both
EV2=$G2/evox_max-autotune-no-cudagraphs
if [ -d $EV2 ]; then EV1=$R3/evox_autotune; PAPER=$A/augmented_vs_evox.csv
else EV2=$G2/evox; EV1=$R3/evox; PAPER=$A/default_mode/augmented_vs_evox.csv; fi
$PY scripts/plot_evox_comparison.py --final $G2/phase3_gpu2_all.csv --evox $EV2 --outdir $G2/analysis --sm 84 \
  > $G2/analysis/evox_comparison.txt
$PY scripts/compare_evox_gpu2.py --evox2 $EV2 --evox1 $EV1 --paper $PAPER --out $G2/analysis --bw-ratio 2.33 \
  > $G2/analysis/evox_gpu2_vs_gpu1.txt

step "Probes and remaining numbers of Section 4"
$PY scripts/analyze_donor_probe.py results/donor_probe/donor_probe.csv > $A/donor_probe.txt
$PY scripts/analyze_defer_probe.py $R3/defer_probe_v3.csv > $A/defer_probe.txt
$PY scripts/analyze_evox_variants.py --dir $R3/evox > $A/evox_variants.txt
$PY scripts/paper_numbers_v3.py --dir $R3 > $A/paper_numbers_v3.txt

step "Table 5, Figures 7 and 9, Table S3: transfer to CEC 2013 LSGO f1-f3, f12 and the knapsack; validation cells"
T=results/transfer
for z in $(find $T -name 'phase3_*.csv.gz'); do [ -f "${z%.gz}" ] || gunzip -k "$z"; done
$PY scripts/analyze_transfer.py --dir $T --ref $R3 > $T/transfer_summary_stdout.txt
$PY scripts/heuristic_variants.py --ref $R3 --transfer $T > $T/heuristic_variants.txt
$PY scripts/plot_quality_paper.py --ref $R3 --transfer $T --out $T/fig_quality_paper
(cd scripts && $PY quality_table.py --ref ../$R3 --transfer ../$T --out ../$T/table_quality_supp.tex) > $T/table_quality_supp.txt
for f in cec-f1 cec-f2 cec-f3 cec-f12 knapsack; do
  $PY scripts/analyze_quality.py --dir $T/$f/quality --fitness $f > $T/$f/quality/quality.txt
done

echo; echo "done: figures and numbers in $A, $R3/roofline, $R3/quality and $G2/analysis; transfer in $T"
