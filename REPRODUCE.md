# Reproducing the paper: details

Reference for the artifact of the paper (see `README.md` for the quick start): which
script produces each figure and table, how to re-run each experiment, the data files
and the hardware and software of the paper.

Naming: the paper calls the kernels `de`, `sbx-pm` and `ux-bf`. The code calls the first
two `de-gen` and `ga-gen` (`--algo ga-gen`), and so do the data files
(`evox_gen_ga-gen.csv`, the `op` column), the sweep drivers and the text reports that the
analysis scripts write next to the figures (`results/*/analysis/*.txt`, `*.csv`). The
figures, the LaTeX tables and the speedup check (`check_speedup.py`, `speedup_tests.sh`)
use the paper's names; the speedup check also accepts the code's names and prints the
exact commands, which use them.

## 1. Regenerate the results from the shipped data (no GPU, a few minutes)

    python3 -m pip install -r requirements-analysis.txt     # numpy scipy matplotlib seaborn statsmodels
    ./reproduce_figures.sh

Analysis outputs go to `results/rerun_v3/analysis/`, `results/gpu2/analysis/` and
`results/transfer/`.

| Paper item | Script | Output |
|---|---|---|
| Figure 3, penalties $P_k$ (Equation 13) | `sensitivity_pmnf.py`, `plot_sensitivity_heatmap.py` | `results/rerun_v3/analysis/fig_sensitivity_heatmap.pdf` |
| Figure 4, near-optimal levels | `plot_sensitivity_intervals.py` | `results/rerun_v3/analysis/fig_sensitivity_intervals.pdf` |
| Table 3 (left), regret of the launch heuristic (Equation 14) | `launch_heuristic.py` | `results/rerun_v3/analysis/launch_heuristic.txt` |
| Section S5 (supplement), constants of the heuristic; leave-one-cell-out regret | `heuristic_loo.py` | `results/rerun_v3/analysis/heuristic_loo.txt` |
| Section 4.1.2, DE donor probe | `analyze_donor_probe.py` | `results/rerun_v3/analysis/donor_probe.txt` |
| Table 3 (right), the heuristic on the second GPU | `score_gpu2.py` | `results/gpu2/analysis/score_gpu2.txt` |
| Figure 5, instruction roofline | `plot_roofline_v3.py --no-l2` | `results/rerun_v3/roofline/fig_roofline_v3_nol2.pdf` |
| Figure 7, best fitness per generation, ours and the baselines | `plot_quality_paper.py`, `analyze_quality.py` | `results/transfer/fig_quality_paper.pdf`, `results/*/quality/quality_summary.csv`, `results/transfer/*/quality/quality.txt` |
| Table S3 (supplement), equivalence tests of the fitness | `quality_table.py` | `results/transfer/table_quality_supp.tex`, `table_quality_supp.txt` |
| Figure 8, best fitness against time at the large-D cell; large-D block of Table S3 | `analyze_quality.py --dir results/largeD/quality`, `plot_time_to_quality.py` | `results/largeD/fig_time_to_quality.pdf`, `time_to_quality.csv`, `quality/quality_summary.csv` |
| Section 4.2, search progress (share of the improvement in the last 100/500 generations) | `quality_progress.py` | `results/rerun_v3/analysis/quality_progress.txt` |
| Figure 6, Table 4, speedups over EvoX | `plot_evox_comparison.py --evox results/rerun_v3/evox_autotune` | `results/rerun_v3/analysis/fig_evox_comparison.pdf`, `evox_comparison.txt`, `augmented_vs_evox.csv` |
| Table S1 (supplement), launch configurations | `gen_evox_config_table.py` | `results/rerun_v3/analysis/evox_launch_configs.tex` |
| Table S4 (supplement), `torch.compile` modes of the baselines | `analyze_compile_modes.py` | `results/rerun_v3/analysis/compile_modes.txt`, `table_compile_modes_supp.tex` |
| Table S5 (supplement), the comparison with EvoX on two GPUs | `plot_evox_comparison.py --sm 84`, `compare_evox_gpu2.py` | `results/gpu2/analysis/evox_gpu2_vs_gpu1.txt` |
| Table 5, Figure 9, transfer to the CEC functions and the knapsack; validation cells | `analyze_transfer.py`, `heuristic_variants.py` | `results/transfer/transfer_summary.txt`, `transfer_cells.csv`, `fig_transfer.pdf` |
| Deferred restore, `StdWorkflow` overhead, remaining numbers of Section 4 | `analyze_defer_probe.py`, `analyze_evox_variants.py`, `paper_numbers_v3.py` | `results/rerun_v3/analysis/{defer_probe,evox_variants,paper_numbers_v3}.txt` |

Figures 1 and 2 are diagrams; Table 1 surveys related work; Table 2 lists the cells of
`configs/`. Table S2 (hardware and software) is summarized at the end of this file.
`sensitivity_pmnf.py` also fits performance models that the paper no longer uses; only
its penalty columns enter Figure 3. `validation_predictions.py` wrote
`configs/validation_predictions.csv`, the heuristic's choices on the validation cells,
before those cells were measured (its "revised" columns are the heuristic of the paper).

## 2. Check the kernels on a GPU (a few minutes)

Requirements: an NVIDIA GPU, the CUDA Toolkit (12.9 or newer; 13.2 for Blackwell, `sm_120`)
and NVIDIA MathDx 25.12, whose cuRANDDx header provides the Philox generator. MathDx is not
redistributed: download it from NVIDIA and unpack it into `code/external/mathdx/25.12/`, or
set `MATHDX_DIR` to its `include/` directory.

    CUDA_ARCH=120 MATHDX_DIR=/path/to/mathdx/25.12/include ./smoke_test.sh

`CUDA_ARCH` is the SM architecture of the GPU that runs the kernels (120: RTX PRO 6000
Blackwell, 86: RTX A6000; `nvidia-smi --query-gpu=compute_cap --format=csv` prints it). The script
builds the kernels and runs their tests: random-stream independence; the fused kernel
against a decomposed pipeline, which must agree bit for bit; and the fitness of every
function, recomputed on the host, for split and unsplit launches with one and several
warps per block. It then times one configuration of the ablation (7165 µs on the paper's
GPU, against 7134 µs in the ablation) and, if `torch` and `evox` are importable, one
baseline cell.

## 3. Check a speedup over the baseline (GPU)

`scripts/check_speedup.py` times one case on both sides and compares the speedup with the
paper's. A case is an operator, a variant, an objective function and a cell $(I, N, D)$,
with $D$ as in Table 2 (`ux-bf` runs $32D$ bits, the same memory):

    python3 scripts/check_speedup.py --op sbx-pm --variant vec4 --function sphere --I 32 --N 256 --D 131072
    python3 scripts/check_speedup.py --op de --function cec-f12 --I 1024 --N 64 --D 1000 --config tuned
    python3 scripts/check_speedup.py --op ux-bf --function knapsack --I 32 --N 256 --D 131072 --config 4,512,1

Functions: `sphere` (`de`, `sbx-pm`), `onemax` (`ux-bf`), `cec-f1`, `cec-f2`, `cec-f3`,
`cec-f12` (`de`, `sbx-pm`) and `knapsack` (`ux-bf`). `--config` is `heuristic` (the
default: Equation 14 with this GPU's SM count), `tuned` (the paper's best configuration
of the case) or an explicit `s,tx,ty`. `--seeds` sets the seeds per side (default 3).

Nothing is hidden behind the speedup. The script prints the exact command of each side,
then the time of every timed generation of every seed, for our kernel and for the
baseline, and writes them to `check_speedup/<case>.times.csv` (columns `side, seed,
generation, time_us`) and `check_speedup/<case>.commands.txt`. Both sides are timed alike:
CUDA events around one generation after warm-up, excluding compilation; ours is one
launch of the fused kernel, the baseline one call of the compiled step (island-local,
`torch.compile` in `max-autotune-no-cudagraphs` mode, as in the paper). Each side's time is
the median over seeds of the per-seed medians, and the speedup is the baseline's time over
ours. The paper's times and speedup for the same case are printed next to the measured
ones (from `results/speedup_reference.csv`, all 270 timed cases). On the paper's GPU the
check passes when the measured speedup is within `--tol` (default 15%) of the paper's; on
another GPU it reports only.

`./speedup_tests.sh` runs six such cases (about 10 minutes): the three operators, both
variants, Sphere, OneMax, $f_3$, $f_{12}$ and the knapsack, a $D = 1000$ cell, a tuned
configuration, and the case where the heuristic is slower than the baseline (`de`,
$I = 1$, $N = 1024$, $D = 2^{20}$: $0.81\times$ in the paper). The output ends with a
summary of measured against published speedups (`check_speedup/summary.txt`). On the
paper's GPU, with 2 seeds per side:

| Case | Measured | Paper |
|---|---|---|
| `sbx-pm` `vec4`, Sphere, $I = 32$, $N = 256$, $D = 2^{17}$ | 4.00× | 4.01× |
| `de` scalar, Sphere, $I = 1$, $N = 1024$, $D = 2^{20}$ | 0.81× | 0.81× |
| `ux-bf` scalar, OneMax, $I = N = D = 1024$ | 6.87× | 6.57× |
| `de` scalar, $f_{12}$, $I = 1024$, $N = 64$, $D = 1000$ | 2.72× | 2.71× |
| `sbx-pm` scalar, $f_3$, $I = 1024$, $N = 256$, $D = 4096$, tuned | 2.14× | 2.06× |
| `ux-bf` `vec4`, knapsack, $I = 32$, $N = 256$, $D = 2^{17}$ | 4.37× | 4.26× |

### Python environment of the baselines

The baselines need Python 3.14, PyTorch 2.14.0 with CUDA and EvoX 1.4.0 (the versions of
the paper). Install PyTorch first, from the index that matches your CUDA driver, then the
rest:

    conda create -n splitgenome python=3.14 && conda activate splitgenome   # or: python3.14 -m venv .venv && . .venv/bin/activate
    pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130    # driver 580 or newer (the first GPU)
    # pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126  # older drivers (the second GPU, driver 570)
    pip install -r requirements-baselines.txt     # evox==1.4.0, numpy
    pip install -r requirements-analysis.txt      # for reproduce_figures.sh
    python -c "import importlib.metadata as m, torch; print(m.version('torch'), m.version('evox'), torch.cuda.is_available())"

The last line should print `2.14.0+cu130 1.4.0 True` (or `+cu126`). `torch.compile` also
needs a C compiler on the path (for Triton). The complete environment of the second GPU is
in `results/gpu2/evox/pip_freeze.txt`. Instead of the installed package, `EVOX_SRC` may
point to the `src/` directory of an EvoX source checkout (tag `v1.4.0`), which is put on
`PYTHONPATH`; the paper's runs on the first GPU used such a checkout.

## 4. Re-run the experiments

Every driver writes into `results/`, over the shipped data; `git status` shows what changed,
and `./reproduce_figures.sh` then rebuilds the figures from the new data. Run each on an
otherwise idle GPU; times are those of the RTX PRO 6000.

| Experiment (paper item) | Command | Time |
|---|---|---|
| register table, needed by the sweeps on a new GPU | `CUDA_ARCH=86 REGS_OUT=results/kernel_regs_sm86.csv bash scripts/gen_kernel_regs_v2.sh` | 1 min |
| main ablation (Figures 3, 4; Tables 3, 4) | see below | 16 h |
| baselines, autotuned, of the main comparison and the transfer functions (Figures 6, 9; Tables 4, 5) | `bash results/rerun_v3/evox_autotune.sh` | 3 h |
| baselines under the other compile modes (Table S4) | `bash results/rerun_v3/evox_modes.sh` (CUDA Graphs); `OUTDIR=results/rerun_v3/evox bash scripts/evox_islands_sweep.sh` (default mode) | 25 min + 1 h |
| baseline variants and deferred restore | `bash scripts/rerun_v3_evox_followup.sh` | 40 min |
| transfer: kernel sweeps, fitness traces and profiles on the CEC functions and the knapsack (Table 5, Figures 7, 9) | `bash scripts/fetch_cec2013_shifts.sh && STEPS="kernels quality ncu" bash scripts/transfer_campaign.sh` | 7.5 h |
| validation cells (Table 5) | `bash scripts/heuristic_validation.sh` | 20 min |
| fitness equivalence, seeds 10–29 (Figure 7, Table S3) | `bash results/transfer/quality30.sh`, after the campaign's quality step (seeds 0–9) | 20 min |
| fitness equivalence and time to quality at the large-D cell (Figure 8, Table S3) | `bash results/largeD/run.sh` | 20 min |
| roofline (Figure 5), needs Nsight Compute and access to performance counters | `bash scripts/run_roofline_v3.sh` | 10 min |
| DE donor probe | `bash scripts/donor_probe.sh` | 5 min |
| second GPU (Tables 3, S5), after building with `CUDA_ARCH=86` | `bash scripts/gpu2/run_sweep.sh i1 && bash scripts/gpu2/run_sweep.sh rest`, then `bash scripts/gpu2/run_evox.sh` (`COMPILE_MODE=max-autotune-no-cudagraphs` for autotuned baselines) | 7 h + 2 h (RTX A6000) |

The main ablation sweeps every split factor $s \in \{1, 2, \dots, 2048\}$, block width
$t_x \in \{32, \dots, 512\}$ and couples per block $t_y \in \{1, \dots, 16\}$ that fits the
thread and register limits, for the three operators in both variants, with 10 seeds of 10
repetitions:

    OPS="de-gen ga-gen ux-bf" LAYOUTS="highdim highdim_vec4" SWEEP=configs/sweep_realistic.json \
    CAPS="1 2 4 8 16 32 64 128 256 512 1024 2048" TXS="32 64 128 256 512" TYS="1 2 4 8 16" \
    SEEDS=10 REPS=10 WARMUP=5 OUT=results/rerun_v3/phase3_unified_final_v3.csv \
    PROGRESS=results/rerun_v3/progress.log bash scripts/phase3_hybrid_probe_v2.sh

For a shorter check of the same claims, reduce `SEEDS` (each seed is a separate run) or
restrict `OPS` and `LAYOUTS`; the analysis scripts accept any subset of the grid.

The fitness traces of the CEC functions use the suite's official shift vectors at
$D = 1000$, which are GPL-3.0 and not redistributed: `scripts/fetch_cec2013_shifts.sh`
downloads them into `data/cec2013lsgo/`. The timing runs use a generated shift (timing
does not depend on the shift values), the same in the kernels and the baselines.

## Data

| File | Content |
|---|---|
| `results/rerun_v3/phase3_unified_final_v3.csv.gz` | main ablation, RTX PRO 6000: one row per timed generation |
| `results/rerun_v3/evox_autotune/` | the paper's baselines, `torch.compile` `max-autotune-no-cudagraphs`, panmictic and island-local, 10 seeds (`evox_gen_<op>.csv`) |
| `results/rerun_v3/evox/` | the same in the default compile mode; variants and tournament selection |
| `results/rerun_v3/evox_modes/` | the compile-mode probe (`reduce-overhead`, 3 seeds) and its summary |
| `results/rerun_v3/quality/` | best and mean fitness per generation on Sphere and OneMax, ours and the baselines, 30 seeds |
| `results/transfer/<f>/` | per transfer function: kernel sweep (`phase3_<f>.csv.gz`), baselines (`evox_autotune/`; `evox/`: default mode), fitness traces at $D = 1000$ with the official shifts or with 4096 items, 30 seeds (`quality/`), Nsight Compute profiles (`roofline/`) |
| `results/transfer/validation/` | sweeps of the seven validation cells (`configs/sweep_validation.json`) |
| `results/transfer/transfer_cells.csv` | per case of the transfer: the configurations, regrets, baseline time and speedups |
| `results/speedup_reference.csv` | the paper's times and speedups of all 270 timed cases, used by `check_speedup.py` |
| `results/rerun_v3/roofline/` | measured roofs and Nsight Compute metrics per configuration |
| `results/transfer/<f>/roofline/` | Nsight Compute metrics of the transfer functions, supplementary (no figure or table uses them); for the knapsack, the six `vec4` configurations of the $I \le 32$ cells recorded no kernel |
| `results/rerun_v3/defer_probe_v3.csv` | in-kernel against deferred restore |
| `results/donor_probe/donor_probe.csv` | DE with one donor row and with constant bounds |
| `results/gpu2/` | second GPU: sweep at $t_y = 1$ (`phase3_gpu2_all.csv.gz`), autotuned baselines (`evox_max-autotune-no-cudagraphs/`, the paper's) and in the default mode (`evox/`), GPU and toolkit (`GPU.txt`) |
| `results/kernel_regs_v2.csv` | registers per thread of each kernel (`sm_120`), used for feasibility |

Columns of the sweep tables: `op, layout, cap` (= $s$), `I, N, D` (`D` in bits for `ux-bf`),
`total, rows, NO, nUnits, nCouples, tx, ty, threads, regs, smem`, `status` (`ok` or the reason
a configuration was skipped: `skip_threads`, `skip_regs`, `skip_tx`, `skip_iters`,
`skip_dup`), `seed, rep` and `time_us`, the time of one generation in microseconds.
Baseline tables: `op, N` (all rows, $I \cdot N$), `D, mode, per_gen_mean_us,
per_gen_min_us, islands, seed, per_gen_median_us`. Each case is summarized as the median
over seeds of the per-seed medians.

Notes on the data:
- The transfer sweeps and the validation use the kernels with the corrected cross-block
  finalize (one thread decides the replacement); the main ablation predates it. The
  correction changes the time of the heuristic's configurations on the nine cells by 0.1%
  (geometric mean; range −0.6% to +0.8%).
- `results/transfer/campaign.log` starts with a profiling step that failed (a script was
  missing at the time); the profiles were taken again in the second run of the log.
- One configuration of the main ablation failed once and keeps 9 seeds
  (`de-gen`, scalar, $s = 64$, $I = 32$, $N = 256$, $t_x = 128$, $t_y = 4$).
- On the second GPU, the `ux-bf` `vec4` rows with $I = 1$ were re-measured after another
  process shared the GPU; `phase3_gpu2_all.csv.gz` contains only the re-measured rows.
- On the second GPU, in the default compile mode, the baseline `ga-gen` at $I = N = D = 1024$
  is $7.1\times$ slower than on the first GPU, against $2.3$–$2.9\times$ for the other
  cells; autotuned, it is $2.4\times$ slower, in line with them.

## Hardware and software of the paper

| | First GPU | Second GPU |
|---|---|---|
| GPU | RTX PRO 6000 Blackwell Workstation (188 SMs, `sm_120`) | RTX A6000 (84 SMs, `sm_86`) |
| Driver, CUDA | 580.178.04, CUDA 13.2 | 570.211.01, CUDA 12.9 |
| PyTorch, EvoX | 2.14.0 (CUDA 13.0), EvoX 1.4.0 | 2.14.0 (CUDA 12.6), EvoX 1.4.0 |
| `torch.compile` mode of the baselines | `max-autotune-no-cudagraphs` | `max-autotune-no-cudagraphs` |
| Profiler | Nsight Compute 2026.1 | — |
