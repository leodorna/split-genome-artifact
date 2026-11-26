#!/usr/bin/env python3
"""EvoX vs unified highdim kernel (scalar and vec4), per generation.

Ours, per (op, family, I, N, D) from the augmented ablation grid (median of
per-seed medians):
  tuned      best (cap, tx, ty) of the cell
  heuristic  launch_heuristic.launch_config(I, N, D, SM count) -- the derived rule of
             the paper (no tuning); a shape above the register limit gets a smaller tx

EvoX compiled on I*N rows (results/rerun_v3/evox_autotune/, from
scripts/evox_islands_sweep.sh; median of per-seed medians):
  island  --islands I: island-local parent/donor draws, same selection as ours
  panm.   --islands 1: one panmictic population (context)

Outputs (in --outdir):
  augmented_vs_evox.csv     per op/family/triple: configs, times, speedups
  fig_evox_comparison.{pdf,png}
    top row:    time per generation (log) - EvoX island/panmictic, ours tuned
    bottom row: speedup over island EvoX; solid = heuristic config,
                light segment on top = extra from per-cell tuning
"""
import argparse
import csv
import os
import shutil
import statistics as st
import sys
from collections import defaultdict

import matplotlib
import matplotlib.ticker
import matplotlib.transforms
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_unified_final import load_final  # noqa: E402
from launch_heuristic import feasible, launch_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OPS = ["de-gen", "ga-gen", "ux-bf"]
# the paper names the kernels de, sbx-pm and ux-bf; the code and data keep de-gen and ga-gen
PAPER_NAME = {"de-gen": "de", "ga-gen": "sbx-pm"}
FAMS = {"scalar": "highdim", "vec4": "highdim_vec4"}
COLOR = {"scalar": "#2a78d6", "vec4": "#eb6834", "evox": "#6b6a64"}
LIGHT = {"scalar": "#9ec5f4", "vec4": "#f5b89c"}
MARK = {"scalar": "o", "vec4": "s"}
BITS = {"ux-bf": 32}   # genes per 32-bit word


def load_evox_seeds(evoxdir, op):
    """(total, D, islands) -> median over seeds of per-seed median (compiled, any mode)."""
    per = defaultdict(list)
    path = os.path.join(evoxdir, f"evox_gen_{op}.csv")
    with open(path) as f:
        for r in csv.DictReader(f):
            # one compile mode per directory: "compiled" (default mode) or
            # "compiled-<mode>" (bench_generation.py --compile-mode)
            if not r["mode"].startswith("compiled") or r["per_gen_median_us"] == "oom":
                continue
            per[(int(r["N"]), int(r["D"]), int(r["islands"]))].append(float(r["per_gen_median_us"]))
    return {k: (st.median(v), len(v)) for k, v in per.items()}


def cells_by_config(final):
    """(op, layout, I, N, D) -> {(cap, tx, ty): time}."""
    cell_sub = defaultdict(dict)
    for (op, layout, I, N, D, cap, tx, ty), us in final.items():
        cell_sub[(op, layout, I, N, D)][(cap, tx, ty)] = us
    return cell_sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=os.path.join(RES, "rerun_v3", "phase3_unified_final_v3.csv"))
    ap.add_argument("--evox", default=os.path.join(RES, "rerun_v3", "evox_autotune"))
    ap.add_argument("--outdir", default=os.path.join(RES, "rerun_v3", "analysis"))
    ap.add_argument("--sm", type=int, default=188, help="SM count for the heuristic")
    ap.add_argument("--paper-fig", default=None,
                    help="optional extra copy of the PDF (e.g. paper figures/ dir)")
    a = ap.parse_args()

    cell_sub = cells_by_config(load_final(a.final))
    # Cells are the (I, N, D) of the real-coded operators; ux-bf runs the same cell
    # with 32x the genes (bits), so its D is looked up as D * BITS["ux-bf"].
    cells = sorted({k[2:] for k in cell_sub if k[0] != "ux-bf"}, key=lambda c: c[2])

    rows = []
    print(f"{'op':8}{'fam':7}{'I':>6}{'N':>6}{'D':>10}{'tuned cfg':>14}{'us':>9}{'heur cfg':>14}{'us':>9}"
          f"{'ev_isl':>9}{'ev_pan':>9}{'x_tun':>7}{'x_heu':>7}")
    for op in OPS:
        ev = load_evox_seeds(a.evox, op)
        for fam, layout in FAMS.items():
            for I, N, Dc in cells:
                D = Dc * BITS.get(op, 1)
                sub = cell_sub.get((op, layout, I, N, D))
                isl = ev.get((I * N, D, I))
                pan = ev.get((I * N, D, 1))
                if not sub or isl is None:
                    continue
                tcfg = min(sub, key=sub.get)
                hcfg = feasible(sub, launch_config(op, fam, I, N, D, a.sm))
                tus, hus = sub[tcfg], sub[hcfg]
                r = dict(op=op, family=fam, I=I, N=N, D=D, D_cell=Dc,
                         cap=tcfg[0], tx=tcfg[1], ty=tcfg[2], tuned_us=tus,
                         heur_cap=hcfg[0], heur_tx=hcfg[1], heur_ty=hcfg[2], heur_us=hus,
                         evox_island_us=isl[0], evox_panmictic_us=pan[0] if pan else float("nan"),
                         evox_seeds=isl[1],
                         speedup_tuned=isl[0] / tus, speedup_heuristic=isl[0] / hus,
                         speedup_tuned_panmictic=(pan[0] / tus) if pan else float("nan"))
                rows.append(r)
                print(f"{op:8}{fam:7}{I:>6}{N:>6}{D:>10}{str(tcfg):>14}{tus:>9.0f}{str(hcfg):>14}{hus:>9.0f}"
                      f"{isl[0]:>9.0f}{r['evox_panmictic_us']:>9.0f}"
                      f"{r['speedup_tuned']:>7.2f}{r['speedup_heuristic']:>7.2f}")

    os.makedirs(a.outdir, exist_ok=True)
    out_csv = os.path.join(a.outdir, "augmented_vs_evox.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) and k.startswith("speedup")
                            else f"{v:.1f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print("wrote " + out_csv)

    sns.set_theme(context="paper", style="ticks")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.15, 1]))
    x = np.arange(len(cells))
    e = lambda v: f"{int(np.log2(v))}"  # noqa: E731
    labels = [f"{e(D)}\n{N}" for I, N, D in cells]   # log2 D and N per cell; I once per group below
    bw = 0.38
    idx = {(r["op"], r["family"], r["D_cell"]): r for r in rows}
    get = lambda op, fam, key: [idx[(op, fam, D)][key] if (op, fam, D) in idx else np.nan  # noqa: E731
                                for _, _, D in cells]
    ms = lambda v: np.array(v, dtype=float) / 1e3  # noqa: E731

    for j, op in enumerate(OPS):
        ax, bx = axes[0, j], axes[1, j]
        ref = "scalar" if any((op, "scalar", D) in idx for _, _, D in cells) else "vec4"
        ax.plot(x, ms(get(op, ref, "evox_panmictic_us")), color=COLOR["evox"], lw=1.1, ls="--",
                label="EvoX compiled, panmictic")
        ax.plot(x, ms(get(op, ref, "evox_island_us")), color=COLOR["evox"], lw=2,
                label="EvoX compiled, island-local")
        for k, fam in enumerate(FAMS):
            ax.plot(x, ms(get(op, fam, "tuned_us")), color=COLOR[fam], lw=2, marker=MARK[fam], ms=4.5,
                    markeredgecolor="white", markeredgewidth=0.8, label=f"ours ({fam}, tuned)")
            off = (k - 0.5) * (bw + 0.02)
            heur, tuned = np.array(get(op, fam, "speedup_heuristic")), np.array(get(op, fam, "speedup_tuned"))
            bx.bar(x + off, heur, width=bw, color=COLOR[fam], label=f"{fam}, heuristic")
            bx.bar(x + off, tuned - heur, bottom=heur, width=bw, color=LIGHT[fam],
                   label=f"{fam}, + per-cell tuning")
            for xi, v in zip(x + off, tuned):
                if np.isnan(v):
                    bx.text(xi, 1.3, "n/a", ha="center", va="bottom", fontsize=5.5,
                            rotation=90, color="0.35")
        ax.set_yscale("log")
        fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(PAPER_NAME.get(op, op), fontsize=9)
        bx.axhline(1, color="0.3", lw=0.8, ls=":")
        ymax = np.nanmax([r["speedup_tuned"] for r in rows if r["op"] == op])
        bx.set_ylim(0, ymax * 1.08)
        bx.set_xticks(x)
        bx.set_xticklabels(labels)
        ax.grid(axis="y", which="major", color="0.9", lw=0.6)
        bx.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True); bx.set_axisbelow(True)
        ax.tick_params(labelsize=7); bx.tick_params(axis="y", labelsize=7)
        bx.tick_params(axis="x", labelsize=5.5)
        # I is constant over consecutive cells: one label per group, under the ticks
        tr = matplotlib.transforms.blended_transform_factory(bx.transData, bx.transAxes)
        groups = []
        for xi, (I, _, _) in zip(x, cells):
            if groups and groups[-1][0] == I:
                groups[-1][2] = xi
            else:
                groups.append([I, xi, xi])
        for I, x0, x1 in groups:
            bx.text((x0 + x1) / 2, -0.30, f"{I}", transform=tr, ha="center", va="top",
                    fontsize=6.5, color="0.2", clip_on=False)
            bx.plot([x0 - 0.35, x1 + 0.35], [-0.28, -0.28], transform=tr, color="0.6", lw=0.6,
                    clip_on=False)

    axes[0, 0].set_ylabel("time / generation (ms)", fontsize=8)
    axes[1, 0].set_ylabel("speedup vs island\nEvoX compiled (×)", fontsize=8)
    axes[1, 0].annotate("$\\log_2 D$\n$N$\n$I$", xy=(0, 0), xycoords="axes fraction",
                        xytext=(-6, -4), textcoords="offset points", ha="right", va="top",
                        fontsize=6.5, color="0.3")
    fig.text(0.5, 0.005, "cells ordered by D; I·N·D = $2^{30}$ genes (ux-bf: 32× as many bits, the same memory)",
             ha="center", fontsize=7.5)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=4, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.0))
    sns.despine(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 0.91))
    for ext in ("pdf", "png"):
        p = os.path.join(a.outdir, f"fig_evox_comparison.{ext}")
        fig.savefig(p, dpi=200)
        print("wrote " + p)
    if a.paper_fig:
        shutil.copy(os.path.join(a.outdir, "fig_evox_comparison.pdf"), a.paper_fig)
        print("wrote " + a.paper_fig)


if __name__ == "__main__":
    main()
