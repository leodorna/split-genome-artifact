#!/usr/bin/env python3
"""Analyze the solution-quality traces of quality_check.sh.

1. Same algorithm as EvoX: per operator and configuration, the median (and range over
   seeds) of the best and mean fitness at a few generations, the relative difference
   to island-local EvoX, and a two-sided Mann-Whitney U test against EvoX.
2. Layout equivalence: with the same seed the split layouts draw the same random
   numbers as the unsplit one; per seed, whether the s8 / s8defer trajectories match
   s1 (relative difference < 1e-5 in every generation) and, if not, where they part.
   With each test, a 95% percentile-bootstrap interval (10,000 resamples of the seeds of
   both sides, fixed RNG) for the relative difference of the medians, ours / EvoX - 1:
   a non-significant test alone does not show that the two are close, the interval
   bounds how far apart they can plausibly be.
3. fig_quality.{pdf,png}: median best fitness per generation with the range over seeds,
   EvoX vs ours (scalar and vec4, unsplit).
The tests are also Holm-corrected over all of them (best and mean, every generation and
configuration). --fitness names the function in the titles (Sphere/OneMax by default;
knapsack titles ux-bf). Configurations beyond the five (e.g. the transfer campaign's
multi-warp s4w / s2w) are picked up from the file names.

Usage: analyze_quality.py [--dir results/rerun_v3/quality] [--fitness rastrigin]
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = ["de-gen", "ga-gen", "ux-bf"]
CFGS = ["s1", "s8", "s8defer", "v4s1", "v4s8"]
CHECK = [10, 100, 1000]
COLOR = {"s1": "#2a78d6", "v4s1": "#eb6834", "evox": "#6b6a64"}
LABEL = {"s1": "ours, scalar", "v4s1": "ours, vec4", "evox": "EvoX compiled, island-local"}
TITLE = {"de-gen": "de-gen (Sphere)", "ga-gen": "ga-gen (Sphere)", "ux-bf": "ux-bf (OneMax, zero bits)"}


def holm(ps):
    """Holm-adjusted p values, in the order of ps."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def boot_ci(x, y, n=10000, seed=0):
    """95% percentile-bootstrap interval of median(x) / median(y) - 1."""
    rng = np.random.default_rng(seed)
    bx = np.median(rng.choice(x, (n, len(x))), axis=1)
    by = np.median(rng.choice(y, (n, len(y))), axis=1)
    r = bx / by - 1
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def load(d):
    T = {}
    for f in glob.glob(os.path.join(d, "*.csv")):
        m = re.match(r"(de-gen|ga-gen|ux-bf)_(\w+)_(\d+)\.csv$", os.path.basename(f))
        if not m:
            continue
        a = np.genfromtxt(f, delimiter=",", names=True)
        T[(m[1], m[2], int(m[3]))] = (a["best"], a["mean"])
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "..", "results", "rerun_v3", "quality"))
    ap.add_argument("--fitness", default=None, help="function of de-gen and ga-gen, for the titles")
    a = ap.parse_args()
    if a.fitness == "knapsack":
        TITLE["ux-bf"] = "ux-bf (0/1 knapsack, -profit + penalty)"
    elif a.fitness:
        for op in ("de-gen", "ga-gen"):
            TITLE[op] = f"{op} ({a.fitness.capitalize()})"
    T = load(a.dir)
    cfgs = CFGS + sorted({k[1] for k in T} - set(CFGS) - {"evox"})
    seeds = sorted({k[2] for k in T})
    ops = [op for op in OPS if any(k[0] == op for k in T)]   # FITNESS runs have no ux-bf
    checks = [g for g in CHECK if g < min(len(v[0]) for v in T.values())]   # shorter runs: fewer
    rows = []

    print("== same algorithm: ours vs island-local EvoX (10 seeds each) ==")
    for op in ops:
        for g in checks:
            ev = {w: np.array([T[(op, "evox", s)][i][g] for s in seeds if (op, "evox", s) in T])
                  for i, w in enumerate(("best", "mean"))}
            for cfg in cfgs + ["evox"]:
                for i, w in enumerate(("best", "mean")):
                    x = np.array([T[(op, cfg, s)][i][g] for s in seeds if (op, cfg, s) in T])
                    if not len(x):
                        continue
                    p = mannwhitneyu(x, ev[w]).pvalue if cfg != "evox" else float("nan")
                    rel = np.median(x) / np.median(ev[w]) - 1
                    ci = boot_ci(x, ev[w]) if cfg != "evox" else (float("nan"), float("nan"))
                    rows.append(dict(op=op, gen=g, config=cfg, metric=w, n=len(x), median=np.median(x),
                                     lo=x.min(), hi=x.max(), rel_to_evox=rel, rel_ci_lo=ci[0], rel_ci_hi=ci[1],
                                     mwu_p=p))
        for g in checks:
            line = [f"{r['config']}: {r['median']:.4g} ({100 * r['rel_to_evox']:+.1f}% "
                    f"[{100 * r['rel_ci_lo']:+.1f}, {100 * r['rel_ci_hi']:+.1f}], p={r['mwu_p']:.2f})"
                    for r in rows if r["op"] == op and r["gen"] == g and r["metric"] == "best" and r["config"] != "evox"]
            e = next(r for r in rows if r["op"] == op and r["gen"] == g and r["metric"] == "best" and r["config"] == "evox")
            print(f"  {op} gen {g:>4} best | evox {e['median']:.4g} [{e['lo']:.4g}, {e['hi']:.4g}] | " + "  ".join(line))
    ps = [r["mwu_p"] for r in rows if r["config"] != "evox"]
    print(f"  smallest p over all {len(ps)} tests: {min(ps):.3f}; tests with p < 0.05: {sum(p < 0.05 for p in ps)}; "
          f"smallest Holm-adjusted p: {min(holm(ps)):.2f}")
    for g in checks:
        cis = [max(abs(r["rel_ci_lo"]), abs(r["rel_ci_hi"])) for r in rows
               if r["config"] != "evox" and r["metric"] == "best" and r["gen"] == g]
        print(f"  gen {g:>4}: every 95% interval of (ours / EvoX - 1) for the best fitness lies within "
              f"+-{100 * max(cis):.1f}%")
    big = [r for r in rows if r["config"] != "evox" and r["metric"] == "best" and abs(r["rel_to_evox"]) > 0.02]
    print(f"  largest |median best - EvoX| / EvoX: {max(abs(r['rel_to_evox']) for r in rows):.3f} (best and mean); "
          f"best fitness beyond 2%: " + (", ".join(f"{r['op']} {r['config']} gen {r['gen']} "
                                                    f"{100 * r['rel_to_evox']:+.1f}% (p={r['mwu_p']:.2f})" for r in big)
                                         or "none"))

    print("\n== layout equivalence: split vs unsplit, same seed ==")
    eq = []
    for op in ops:
        pairs = [("s8", "s1"), ("s8defer", "s1"), ("v4s8", "v4s1")] + [(c, "s1") for c in cfgs[len(CFGS):]]
        for cfg, ref in pairs:
            same, parted = 0, []
            for s in seeds:
                if (op, cfg, s) not in T or (op, ref, s) not in T:
                    continue
                b, r = T[(op, cfg, s)][1], T[(op, ref, s)][1]
                d = np.abs(b - r) / np.maximum(np.abs(r), 1e-30)
                if (d < 1e-5).all():
                    same += 1
                else:
                    parted.append(int(np.argmax(d >= 1e-5)))
            eq.append(dict(op=op, config=cfg, reference=ref, identical=same, diverged=len(parted),
                           first_divergence=";".join(map(str, sorted(parted)))))
            print(f"  {op:7} {cfg:8} vs {ref:5}: identical trajectory in {same} of {same + len(parted)} seeds"
                  + (f"; the others part at generations {sorted(parted)}" if parted else ""))

    with open(os.path.join(a.dir, "quality_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.dir, "quality_layouts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eq[0]))
        w.writeheader()
        w.writerows(eq)

    sns.set_theme(context="paper", style="ticks")
    fig, axes = plt.subplots(1, len(ops), figsize=(2.4 * len(ops), 2.3), constrained_layout=True)
    for ax, op in zip(np.atleast_1d(axes), ops):
        for cfg in ("evox", "s1", "v4s1"):
            Y = np.array([T[(op, cfg, s)][0] for s in seeds if (op, cfg, s) in T])
            if not len(Y):
                continue
            x = np.arange(Y.shape[1])
            ax.fill_between(x, Y.min(0), Y.max(0), color=COLOR[cfg], alpha=0.18, lw=0)
            ax.plot(x, np.median(Y, 0), color=COLOR[cfg], lw=1.6 if cfg == "evox" else 1.1,
                    ls="--" if cfg == "v4s1" else "-", label=LABEL[cfg])
        ax.set_title(TITLE[op], fontsize=8.5)
        ax.set_xlabel("generation", fontsize=8)
        ax.grid(color="0.92", lw=0.6)
        ax.tick_params(labelsize=7)
        sns.despine(ax=ax)
    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("best fitness (minimized)", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3, frameon=False, fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.dir, f"fig_quality.{ext}"), dpi=200)
    print("\nwrote quality_summary.csv, quality_layouts.csv, fig_quality.{pdf,png} in", a.dir)


if __name__ == "__main__":
    main()
