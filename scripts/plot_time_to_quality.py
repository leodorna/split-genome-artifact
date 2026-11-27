#!/usr/bin/env python3
"""Best fitness against time at the large-D cell (I = 1, N = 64, D = 2^20; ux-bf 32 D bits),
from results/largeD (run.sh there): the 1000-generation traces of the baseline, our
unsplit kernel (s = 1) and our kernel at the heuristic's split configuration, 30 seeds
each, with the time axis (log scale, from generation 1) set to the generation index times the generation time measured
at the same cell (median over seeds of per-seed medians; check_speedup.py, baseline
compiled with max-autotune-no-cudagraphs). The traced runs copy the fitness to the host
every generation, so they are not timed themselves. Median and range over the seeds.

Usage: plot_time_to_quality.py [--dir results/largeD] [--out results/largeD/fig_time_to_quality]
"""
import argparse
import csv
import glob
import os
import re
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = [("de-gen", "de", "sphere", "Sphere"), ("ga-gen", "sbx-pm", "sphere", "Sphere"),
       ("ux-bf", "ux-bf", "onemax", "OneMax")]
HEUR = {"de-gen": "4-512-1", "ga-gen": "8-512-1", "ux-bf": "8-512-1"}
COLOR = {"evox": "#6b6a64", "s1": "#2a78d6", "heur": "#eb6834"}


def traces(d, op, cfg):
    Y = []
    for f in glob.glob(os.path.join(d, f"{op}_{cfg}_*.csv")):
        if re.match(rf"{op}_{cfg}_\d+\.csv$", os.path.basename(f)):
            Y.append(np.genfromtxt(f, delimiter=",", names=True)["best"])
    return np.array(Y)


def gen_time(path, side):
    """Median over seeds of the per-seed median generation time (us)."""
    per = {}
    for r in csv.DictReader(open(path)):
        if r["side"] == side:
            per.setdefault(r["seed"], []).append(float(r["time_us"]))
    return st.median(st.median(v) for v in per.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "..", "results", "largeD"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "fig_time_to_quality")
    q, tdir = os.path.join(a.dir, "quality"), os.path.join(a.dir, "timing")

    sns.set_theme(context="paper", style="ticks")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), constrained_layout=True)
    rows = []
    for ax, (op, name, fn, fname) in zip(axes, OPS):
        tag = f"{name}_scalar_{fn}_I1_N64_D1048576"
        t_heur = gen_time(os.path.join(tdir, f"{tag}_{HEUR[op]}.times.csv"), "ours")
        t_s1 = gen_time(os.path.join(tdir, f"{tag}_1-512-1.times.csv"), "ours")
        # the baseline is timed in both runs; pool them
        t_ev = st.median([gen_time(os.path.join(tdir, f"{tag}_{c}.times.csv"), "baseline")
                          for c in (HEUR[op], "1-512-1")])
        series = [("evox", t_ev, "baseline"), ("s1", t_s1, "ours, unsplit ($s = 1$)"),
                  ("heur", t_heur, "ours, heuristic ($s = 4$ for de, 8 for sbx-pm and ux-bf)")]
        for cfg, t, label in series:
            Y = traces(q, op, cfg)
            # log time axis from generation 1: a constant speedup is a constant shift
            x = np.arange(1, Y.shape[1]) * t * 1e-6
            Z = Y[:, 1:]
            ax.fill_between(x, Z.min(0), Z.max(0), color=COLOR[cfg], alpha=0.18, lw=0)
            ax.plot(x, np.median(Z, 0), color=COLOR[cfg], lw=1.1, label=label)
            rows.append(dict(op=name, config=cfg, seeds=len(Y), gen_time_us=round(t, 1),
                             time_1000_gens_s=round(1000 * t * 1e-6, 3),
                             median_best_gen1000=float(np.median(Y[:, -1]))))
        ax.set_title(f"{name}, {fname}", fontsize=8)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.tick_params(labelsize=6, pad=1.5)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.set_xscale("log")
        sns.despine(ax=ax)
    axes[0].set_ylabel("best fitness", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    with open(os.path.join(a.dir, "time_to_quality.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", out + ".pdf")


if __name__ == "__main__":
    main()
