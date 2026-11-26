#!/usr/bin/env python3
"""Launch-parameter penalty heatmap: all ops x {scalar, vec4} x D, one panel per knob.

Reads results/rerun_v3/analysis/sensitivity_coeffs_pmnf.csv (sensitivity_pmnf.py). Cell value =
penalty of the knob (slowdown of its worst setting with the other two re-tuned).
Hatched: no valid launch for that (op, family, D). Dot: the knob has a single
effective value in that cell (penalty trivially 1).

Outputs results/rerun_v3/analysis/fig_sensitivity_heatmap.{pdf,png} (+ optional paper copy).
"""
import argparse
import csv
import math
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = [(op, fam) for op in ("de-gen", "ga-gen", "ux-bf") for fam in ("scalar", "vec4")]
# the paper names the kernels de, sbx-pm and ux-bf; the code and data keep de-gen and ga-gen
PAPER_NAME = {"de-gen": "de", "ga-gen": "sbx-pm"}
KNOBS = [("cap", r"$s$"), ("tx", r"$t_x$"), ("ty", r"$t_y$")]
DS = [2 ** k for k in (10, 12, 14, 15, 17, 19, 20, 22, 24)]
BITS = {"ux-bf": 32}   # genes per 32-bit word
# single-hue sequential ramp (blue 100 -> 700), white at "no effect"
CMAP = LinearSegmentedColormap.from_list(
    "penalty", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeffs", default=os.path.join(HERE, "..", "results", "rerun_v3", "analysis",
                                                     "sensitivity_coeffs_pmnf.csv"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "results", "rerun_v3", "analysis"))
    ap.add_argument("--paper-fig", default=None)
    a = ap.parse_args()

    pen, single = {}, set()
    for r in csv.DictReader(open(a.coeffs)):
        # columns are the cells' D in 32-bit words; ux-bf runs them with 32x the genes (bits)
        k = (r["op"], r["family"], int(r["D"]) // BITS.get(r["op"], 1), r["knob"])
        pen[k] = float(r["penalty"])
        if math.isnan(float(r["elasticity"])):
            single.add(k)
    vmax = max(pen.values())
    norm = LogNorm(vmin=1.0, vmax=vmax)

    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
    for p, (kn, ktex) in enumerate(KNOBS):
        ax = axes[p]
        M = np.full((len(ROWS), len(DS)), np.nan)
        for i, (op, fam) in enumerate(ROWS):
            for j, D in enumerate(DS):
                v = pen.get((op, fam, D, kn))
                if v is not None:
                    M[i, j] = v
        mesh = ax.pcolormesh(np.ma.masked_invalid(M), cmap=CMAP, norm=norm,
                             edgecolors="white", linewidth=1.0)
        for i, (op, fam) in enumerate(ROWS):
            for j, D in enumerate(DS):
                v = M[i, j]
                if np.isnan(v):
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch="////",
                                               edgecolor="0.6", lw=0))
                elif (op, fam, D, kn) in single:
                    ax.plot(j + 0.5, i + 0.5, "o", ms=2, color="0.45")
                else:
                    dark = norm(v) > 0.55
                    ax.text(j + 0.5, i + 0.5, f"{v:.2f}" if v < 9.995 else f"{v:.1f}",
                            ha="center", va="center", fontsize=5.3,
                            color="white" if dark else "0.15")
        for y in (2, 4):  # separate the op groups
            ax.axhline(y, color="white", lw=5)
        ax.set_xticks(np.arange(len(DS)) + 0.5)
        ax.set_xticklabels([str(int(math.log2(D))) for D in DS])
        ax.set_xlabel(r"$\log_2 D$")
        ax.set_title(ktex, fontsize=9)
        ax.set_ylim(len(ROWS), 0)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(np.arange(len(ROWS)) + 0.5)
    axes[0].set_yticklabels([f"{fam} {PAPER_NAME.get(op, op)}" for op, fam in ROWS])   # variant first, as in the paper

    cax = fig.add_axes([0.925, 0.2, 0.012, 0.62])
    cb = fig.colorbar(mesh, cax=cax)
    ticks = [t for t in (1, 1.5, 2, 3, 4, 6) if t <= vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}×" for t in ticks])
    cb.minorticks_off()
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=0)
    cb.set_label("penalty", fontsize=6.5)
    fig.subplots_adjust(left=0.1, right=0.91, bottom=0.17, top=0.9, wspace=0.06)

    os.makedirs(a.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(a.outdir, f"fig_sensitivity_heatmap.{ext}")
        fig.savefig(p, dpi=220)
        print("wrote", p)
    if a.paper_fig:
        shutil.copy(os.path.join(a.outdir, "fig_sensitivity_heatmap.pdf"), a.paper_fig)
        print("wrote", a.paper_fig)


if __name__ == "__main__":
    main()
