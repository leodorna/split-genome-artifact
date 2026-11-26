#!/usr/bin/env python3
"""Figure 4 of the paper: near-optimal levels of the three launch parameters.

For every cell and parameter, the levels whose profile (best time with that level, the
other two parameters re-tuned, over the best time of the cell) is within 5% (dark bar) and
within 25% (light bar) of the best time; the white dot is the best level and the orange
ring the level chosen by the launch heuristic.

Usage: plot_sensitivity_intervals.py [--final results/rerun_v3/phase3_unified_final_v3.csv]
                                     [--out results/rerun_v3/analysis/fig_sensitivity_intervals]
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plot_sensitivity_profiles as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--final", default=os.path.join(HERE, "..", "results", "rerun_v3", "phase3_unified_final_v3.csv"))
ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "rerun_v3", "analysis", "fig_sensitivity_intervals"))
ap.add_argument("--sm", type=int, default=188, help="SM count for the heuristic")
args = ap.parse_args()
OUT, SM, T1, T2 = args.out, args.sm, 1.05, 1.25
os.makedirs(os.path.dirname(OUT), exist_ok=True)

cells = P.load(args.final)
prof = {k: P.profiles(t) for k, t in cells.items()}
NO = {"de-gen": 1, "ga-gen": 2, "ux-bf": 2}
# the paper names the kernels de, sbx-pm and ux-bf; the code and data keep de-gen and ga-gen
PAPER_NAME = {"de-gen": "de", "ga-gen": "sbx-pm"}
C1, C2, EDGE, HEUR, DOT = "#2a78d6", "#b7d3f6", "#0d366b", "#eb6834", "0.60"
LEVELS = {"cap": [2 ** e for e in range(12)], "tx": [32, 64, 128, 256, 512], "ty": [1, 2, 4, 8, 16]}
INK, MUTED = "0.15", "0.45"
LW = 3.9          # bar thickness in points
PAD = 0.37        # half-length of one level; the round cap adds the rest

def heur(op, fam, I, N, D):
    U = P.units_per_row(op, fam, D)
    tx = min(512, max(32, 1 << int(math.floor(math.log2(U / 16)))))
    cap = 1 << max(0, math.ceil(math.log2(max(SM / (I * (N // NO[op])), 1))))
    return {"cap": min(cap, max(1, math.ceil(U / tx))), "tx": tx, "ty": 1}

def runs(p, tol):
    """Maximal runs of adjacent levels (in log2) whose slowdown is <= tol."""
    xs = sorted(round(math.log2(l)) for l, v in p.items() if v <= tol)
    out = []
    for x in xs:
        if out and x == out[-1][1] + 1: out[-1][1] = x
        else: out.append([x, x])
    return out

ypos = lambda r: r + 0.55 * (r // 3)       # a gap between the three D regimes
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})
fig, axes = plt.subplots(6, 3, figsize=(7.2, 7.8), sharex="col", gridspec_kw={"width_ratios": [12, 5, 5]})
n1 = n2 = tot = 0
for i, (op, fam) in enumerate(P.ROWS):
    ks = sorted((k for k in cells if k[:2] == (op, fam)), key=lambda k: k[4])
    for j, (kn, title) in enumerate(P.KNOBS):
        ax = axes[i][j]
        for r, k in enumerate(ks):
            p, y = prof[k][kn], ypos(r)
            ax.plot([math.log2(l) for l in p], [y] * len(p), ls="", marker="o", ms=1.8, color=DOT, markeredgewidth=0, zorder=1)
            na = [math.log2(l) for l in LEVELS[kn] if l not in p]
            ax.plot(na, [y] * len(na), ls="", marker="x", ms=2.6, color=DOT, mew=0.7, zorder=1)
            for tol, col, z in ((T2, C2, 2), (T1, C1, 3)):
                for a, b in runs(p, tol):
                    ax.plot([a - PAD, b + PAD], [y, y], color=col, lw=LW, solid_capstyle="round", zorder=z)
            h = min(max(heur(*k)[kn], min(p)), max(p))
            ax.plot([math.log2(h)], [y], marker="o", ms=4.8, mfc="none", mec="white", mew=1.9, zorder=5)
            ax.plot([math.log2(h)], [y], marker="o", ms=4.8, mfc="none", mec=HEUR, mew=1.0, zorder=6)
            ax.plot([math.log2(min(p, key=p.get))], [y], marker="o", ms=2.8, mfc="white", mec=EDGE, mew=0.7, zorder=7)
            tot += 1; n1 += p.get(h, 9) <= T1; n2 += p.get(h, 9) <= T2
        for l in LEVELS[kn]: ax.axvline(math.log2(l), color="0.92", lw=0.5, zorder=0)
        if j == 2:
            for g, I in enumerate((1024, 32, 1)):
                assert {k[2] for k in ks[3 * g:3 * g + 3]} == {I}
                ax.text(1.03, ypos(3 * g + 1), f"$I={I}$", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, color=MUTED)
        ax.set_ylim(ypos(len(ks) - 1) + 0.75, -0.75)
        ax.set_yticks([ypos(r) for r in range(len(ks))])
        ax.set_yticklabels([f"$2^{{{int(math.log2(k[4] // P.BITS.get(op, 1)))}}}$" for k in ks] if j == 0 else [], fontsize=6.8)
        ax.tick_params(length=0, labelcolor=MUTED, pad=2)
        ax.tick_params(axis="x", length=2, pad=2, color=MUTED, labelcolor=INK)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("0.75"); ax.spines["bottom"].set_linewidth(0.6)
        if i == 0: ax.set_title(title, fontsize=9, pad=5)
        if j == 0:
            ax.text(-0.155, 0.5, fam, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=8, color=MUTED)
            ax.text(-0.115, 0.5, PAPER_NAME.get(op, op), transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=9, color=INK, fontweight="bold")
for j, lv in ((0, [2 ** e for e in range(12)]), (1, (32, 64, 128, 256, 512)), (2, (1, 2, 4, 8, 16))):
    a = axes[-1][j]
    a.set_xticks([math.log2(v) for v in lv])
    a.set_xticklabels([f"$2^{{{int(math.log2(v))}}}$" for v in lv] if j == 0 else [str(v) for v in lv])
    a.set_xlim(math.log2(lv[0]) - 0.75, math.log2(lv[-1]) + 0.75)
axes[0][0].text(-0.012, 1.03, "$D$", transform=axes[0][0].transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
L = plt.Line2D
h = [L([], [], color=C1, lw=LW, solid_capstyle="round", label="within 5% of the best time"),
     L([], [], color=C2, lw=LW, solid_capstyle="round", label="within 25%"),
     L([], [], ls="", marker="o", ms=2.6, color=DOT, label="tested, slower"),
     L([], [], ls="", marker="x", ms=3, color=DOT, mew=0.8, label="not applicable"),
     L([], [], ls="", marker="o", ms=3.4, mfc="white", mec=EDGE, mew=0.8, label="best level"),
     L([], [], ls="", marker="o", ms=4.8, mfc="none", mec=HEUR, mew=1.0, label="chosen by the heuristic")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7.8, columnspacing=1.6, handlelength=1.6, handletextpad=0.6)
fig.subplots_adjust(left=0.105, right=0.925, bottom=0.088, top=0.968, wspace=0.04, hspace=0.12)
fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=200)
print("wrote", OUT, f"| heuristic within 5%: {n1}/{tot}, within 25%: {n2}/{tot}")
for i, (op, fam) in enumerate(P.ROWS):
    for k in sorted((k for k in cells if k[:2] == (op, fam)), key=lambda k: k[4]):
        for kn in ("cap", "tx", "ty"):
            p = prof[k][kn]; h = min(max(heur(*k)[kn], min(p)), max(p))
            if p[h] > T2: print("  >25%:", fam, op, "I=%d N=%d D=%d" % k[2:], kn, h, f"{p[h]:.2f}")
