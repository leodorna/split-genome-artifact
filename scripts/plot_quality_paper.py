#!/usr/bin/env python3
"""Figure 7 of the paper: best fitness per generation, ours against the baselines, for
every function the paper times: Sphere and the CEC 2013 LSGO f1, f2, f3, f12 (rows:
de-gen, ga-gen), and OneMax and the 0/1 knapsack (ux-bf, last column). Median and range
over the seeds of island-local compiled EvoX (or the PyTorch binary GA) and of our scalar
and vec4 kernels (unsplit), from the traces of quality_check.sh: Sphere and OneMax in
results/rerun_v3/quality (I = 4, N = 64, D = 1024, 4096 bits), the CEC functions at
D = 1000 with the official shifts and the knapsack with 4096 items in
results/transfer/<f>/quality. Statistics: analyze_quality.py.

Usage: plot_quality_paper.py [--ref results/rerun_v3] [--transfer results/transfer]
                             [--out results/transfer/fig_quality_paper]
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import seaborn as sns  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
COLOR = {"s1": "#2a78d6", "v4s1": "#eb6834", "evox": "#6b6a64"}
# the paper names the kernels de, sbx-pm and ux-bf; the code and data keep de-gen and ga-gen
PAPER_NAME = {"de-gen": "de", "ga-gen": "sbx-pm"}
LABEL = {"evox": "baseline (compiled, island-local)", "s1": "ours, scalar", "v4s1": "ours, vec4"}



def load(d):
    """Best fitness per generation of every trace in d: {(op, config, seed): array}."""
    T = {}
    for f in glob.glob(os.path.join(d, "*.csv")):
        m = re.match(r"(de-gen|ga-gen|ux-bf)_(\w+)_(\d+)\.csv$", os.path.basename(f))
        if m:
            T[(m[1], m[2], int(m[3]))] = np.genfromtxt(f, delimiter=",", names=True)["best"]
    return T

def panel(ax, T, op, title, log):
    for cfg in ("evox", "s1", "v4s1"):
        Y = np.array([v for (o, c, _), v in sorted(T.items()) if o == op and c == cfg])
        if not len(Y):
            continue
        x = np.arange(Y.shape[1])
        ax.fill_between(x, Y.min(0), Y.max(0), color=COLOR[cfg], alpha=0.18, lw=0)
        ax.plot(x, np.median(Y, 0), color=COLOR[cfg], lw=1.5 if cfg == "evox" else 1.0,
                ls="--" if cfg == "v4s1" else "-", label=LABEL[cfg])
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_xticks([0, 500, 1000])
    ax.tick_params(labelsize=6, pad=1.5)
    ax.yaxis.get_offset_text().set_fontsize(6)
    ax.grid(color="0.92", lw=0.6)
    sns.despine(ax=ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(HERE, "..", "results", "rerun_v3"))
    ap.add_argument("--transfer", default=os.path.join(HERE, "..", "results", "transfer"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "transfer", "fig_quality_paper"))
    a = ap.parse_args()

    sph = load(os.path.join(a.ref, "quality"))
    tr = {f: load(os.path.join(a.transfer, f, "quality"))
          for f in ("cec-f1", "cec-f2", "cec-f3", "cec-f12", "knapsack")}
    # (data, title, log scale) per column; rows de-gen, ga-gen
    cols = [(sph, "Sphere", False), (tr["cec-f1"], "$f_1$", True), (tr["cec-f2"], "$f_2$", False),
            (tr["cec-f3"], "$f_3$", False), (tr["cec-f12"], "$f_{12}$", True)]

    sns.set_theme(context="paper", style="ticks")
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 3.2), constrained_layout=True)
    for r, op in enumerate(("de-gen", "ga-gen")):
        for c, (T, name, log) in enumerate(cols):
            panel(axes[r][c], T, op, f"{PAPER_NAME.get(op, op)}, {name}", log)
    panel(axes[0][5], sph, "ux-bf", "ux-bf, OneMax", False)
    panel(axes[1][5], tr["knapsack"], "ux-bf", "ux-bf, knapsack", False)
    axes[1][5].set_ylabel(r"$\times 10^6$", fontsize=6, labelpad=1)
    axes[1][5].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
    for ax in axes[1]:
        ax.set_xlabel("generation", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("best fitness", fontsize=7)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3, frameon=False, fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=200)
    print("wrote", a.out + ".{pdf,png}")


if __name__ == "__main__":
    main()
