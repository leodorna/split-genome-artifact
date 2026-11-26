#!/usr/bin/env python3
"""Profile curves of the launch parameters: all ops x {scalar, vec4}, one column per knob.

For every level v of a knob k in (cap, tx, ty), the other two knobs are re-tuned:
profile_k(v) = min { t(c) : c_k = v } / min_c t(c). The maximum of a profile is the
penalty of sensitivity_pmnf.py; the curve also shows which levels are slow and where
the best level lies, with no model.

Each panel holds the nine (I, N, D) cells of one kernel, coloured by D (light -> dark).

Outputs (--outdir): sensitivity_profiles.csv, fig_sensitivity_profiles.{pdf,png}
Also prints the numbers behind the tx rule of the launch heuristic: the cost of a
512-thread block against the units each of its threads would process.

Usage: plot_sensitivity_profiles.py [--final results/rerun_v3/phase3_unified_final_v3.csv]
"""
import argparse
import csv
import math
import os
import shutil
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "..", "results", "rerun_v3")
ROWS = [(op, fam) for op in ("de-gen", "ga-gen", "ux-bf") for fam in ("scalar", "vec4")]
KNOBS = [("cap", r"split factor $s$"), ("tx", r"threads per couple $t_x$"), ("ty", r"couples per block $t_y$")]
BITS = {"ux-bf": 32}   # genes per 32-bit word
# single-hue sequential ramp (the heatmap's blue), starting dark enough to read on white
CMAP = LinearSegmentedColormap.from_list("cells", ["#a6c9f4", "#3987e5", "#144a8c", "#04142e"])
INK, MUTED = "0.15", "0.45"


def units_per_row(op, fam, D):
    binary = op == "ux-bf"
    if fam == "vec4":
        return D // 128 if binary else D // 4
    return D // 32 if binary else D


def load(path):
    """(op, fam, I, N, D) -> {(cap, tx, ty): median over seeds of the per-seed medians}."""
    per = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok":
            continue
        fam = "vec4" if r["layout"].endswith("vec4") else "scalar"
        k = (r["op"], fam, int(r["I"]), int(r["N"]), int(r["D"]), int(r["cap"]), int(r["tx"]), int(r["ty"]))
        per[k][r["seed"]].append(float(r["time_us"]))
    cells = defaultdict(dict)
    for k, d in per.items():
        cells[k[:5]][k[5:]] = st.median(st.median(v) for v in d.values())
    return cells


def profiles(t):
    """{knob: {level: best time with that level / best time of the cell}}."""
    best = min(t.values())
    out = {}
    for i, (kn, _) in enumerate(KNOBS):
        p = defaultdict(lambda: math.inf)
        for c, v in t.items():
            p[c[i]] = min(p[c[i]], v)
        out[kn] = {lv: v / best for lv, v in sorted(p.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=os.path.join(V3, "phase3_unified_final_v3.csv"))
    ap.add_argument("--outdir", default=os.path.join(V3, "analysis"))
    ap.add_argument("--paper-fig", default=None)
    a = ap.parse_args()

    cells = load(a.final)
    prof = {k: profiles(t) for k, t in cells.items()}
    with open(os.path.join(a.outdir, "sensitivity_profiles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "family", "I", "N", "D", "knob", "level", "slowdown"])
        for k in sorted(prof):
            for kn, p in prof[k].items():
                for lv, v in p.items():
                    w.writerow(list(k) + [kn, lv, f"{v:.4f}"])

    words = sorted({k[4] // BITS.get(k[0], 1) for k in cells})   # D of the real-coded operators
    color = {d: CMAP(i / (len(words) - 1)) for i, d in enumerate(words)}
    vmax = max(v for p in prof.values() for q in p.values() for v in q.values())

    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif"})
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(7.2, 6.1), sharex="col", sharey=True)
    for i, (op, fam) in enumerate(ROWS):
        ks = sorted((k for k in cells if k[:2] == (op, fam)), key=lambda k: k[4])
        for j, (kn, title) in enumerate(KNOBS):
            ax = axes[i][j]
            ax.axhline(1, color="0.8", lw=0.6, zorder=1)
            for k in ks:
                p = prof[k][kn]
                ax.plot(list(p), list(p.values()), color=color[k[4] // BITS.get(op, 1)], lw=1.0,
                        marker="o", ms=2.2, markeredgewidth=0, zorder=3)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_ylim(0.93, vmax * 1.08)
            ax.set_yticks([1, 1.5, 2, 3, 4, 6])
            ax.set_yticklabels(["1×", "1.5×", "2×", "3×", "4×", "6×"])
            ax.minorticks_off()
            ax.grid(axis="y", color="0.93", lw=0.5)
            ax.tick_params(length=2, color=MUTED, labelcolor=INK)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(MUTED)
                ax.spines[s].set_linewidth(0.6)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{fam}\n{op}", fontsize=7.5)   # variant first, as in the paper
    axes[-1][0].set_xticks([2 ** e for e in (0, 2, 4, 6, 8, 10)])
    axes[-1][0].set_xticklabels([f"$2^{{{e}}}$" for e in (0, 2, 4, 6, 8, 10)])
    for j, lv in ((1, (32, 64, 128, 256, 512)), (2, (1, 2, 4, 8, 16))):
        axes[-1][j].set_xticks(lv)
        axes[-1][j].set_xticklabels([str(v) for v in lv])
        axes[-1][j].set_xlim(lv[0] / 1.25, lv[-1] * 1.25)
    fig.supylabel("slowdown against the best configuration of the cell", fontsize=7.5, x=0.005)

    handles = [plt.Line2D([], [], color=color[d], lw=1.6, label=f"$2^{{{int(math.log2(d))}}}$") for d in words]
    fig.legend(handles=handles, title="$D$ of the cell", loc="center right", frameon=False,
               fontsize=7, title_fontsize=7.5, handlelength=1.4, labelspacing=0.5)
    fig.subplots_adjust(left=0.115, right=0.895, bottom=0.05, top=0.955, wspace=0.07, hspace=0.16)

    for ext in ("pdf", "png"):
        p = os.path.join(a.outdir, f"fig_sensitivity_profiles.{ext}")
        fig.savefig(p, dpi=220)
        print("wrote", p)
    if a.paper_fig:
        shutil.copy(os.path.join(a.outdir, "fig_sensitivity_profiles.pdf"), a.paper_fig)
        print("wrote", a.paper_fig)

    # ---- numbers behind the tx rule: cost of 512-thread blocks vs units per thread ----
    print("\n== cost of tx = 512 (other knobs re-tuned) by units per thread U/512 ==")
    by = defaultdict(list)
    for k in sorted(cells, key=lambda k: (k[0], k[1], k[4])):
        op, fam, I, N, D = k
        p = prof[k]["tx"]
        if 512 in p:
            by[units_per_row(op, fam, D) / 512].append((p[512], op, fam))
    for u in sorted(by):
        print(f"  {u:8g} units/thread: " + "  ".join(f"{op} {fam} {v:.2f}" for v, op, fam in sorted(by[u], key=lambda x: x[1:])))
    print("\n== cost of the heuristic's tx (other knobs re-tuned) ==")
    cost = []
    for k in cells:
        op, fam, I, N, D = k
        U = units_per_row(op, fam, D)
        tx = min(512, max(32, 1 << int(math.floor(math.log2(U / 16)))))
        cost.append((prof[k]["tx"][tx], k))
    cost.sort()
    print(f"  median {st.median(c for c, _ in cost):.3f}; <= 1.05 in {sum(c <= 1.05 for c, _ in cost)}/{len(cost)}; worst:")
    for c, k in cost[-8:]:
        print(f"    {c:.2f}  {k}")
    print("\n== cost of ty = 1 and of cap = 1 (other knobs re-tuned) ==")
    ty1 = [prof[k]["ty"][1] for k in cells]
    print(f"  ty = 1: median {st.median(ty1):.3f}, max {max(ty1):.3f}, <= 1.05 in {sum(v <= 1.05 for v in ty1)}/{len(ty1)}")
    c1 = sorted((prof[k]["cap"][1], k) for k in cells)
    print(f"  cap = 1: median {st.median(c for c, _ in c1):.3f}, max {c1[-1][0]:.2f}; >= 1.05 in {sum(c >= 1.05 for c, _ in c1)}/{len(c1)}")
    for c, k in c1:
        if c >= 1.05:
            print(f"    {c:.2f}  {k}")


if __name__ == "__main__":
    main()
