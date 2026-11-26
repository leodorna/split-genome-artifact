#!/usr/bin/env python3
"""Instruction roofline (Ding & Williams 2019) of the split-genome kernels.

x = warp instructions per DRAM byte, y = warp instructions per second, both summed
over the kernels of one generation (parse_roofline_v3.py). Roofs are the measured
ceilings (roofline_ceilings_v3.csv): DRAM and L2 bandwidth slopes, the warp issue
rate, and the issue rate of the binary operator's random-number code alone.

  (a) the launch heuristic's config in every cell (tag heur)
  (b, c) the split factor s from 1 to fully split for one cell (tag traj,
         --traj-cell): DRAM traffic and the share of the DRAM roof reached, whose
         ratio gives the time (on the roofline the points only slide along the roof)

Colour = scalar / vec4 (as in the EvoX figure), marker = operator.
Also writes roofline_table_v3.csv: the heuristic points with their
speed-of-light fractions.

Usage: plot_roofline_v3.py [--dir results/rerun_v3/roofline] [--traj-cell 1,64]
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(context="paper", style="ticks", font_scale=0.95)
HERE = os.path.dirname(os.path.abspath(__file__))
DEF_DIR = os.path.join(HERE, "..", "results", "rerun_v3", "roofline")
COLOR = {"scalar": "#2a78d6", "vec4": "#eb6834"}
MARK = {"de-gen": "o", "ga-gen": "s", "ux-bf": "^"}
OP_NAME = {"de-gen": "DE", "ga-gen": "real-coded GA", "ux-bf": "binary GA"}
INK, MUTED = "0.15", "0.45"


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def roofs(ceil):
    c = {r["kernel"]: r for r in ceil}
    issue = max(c[k]["warp_GIPS"] for k in ("issue_fp32", "issue_int", "issue_imad", "issue_mix") if k in c)
    return {"dram": c["copy_dram"]["dram_GBps"], "l2": c["copy_l2"]["l2_GBps"], "issue": issue,
            "rng": c.get("binary_rng_vec4", {}).get("warp_GIPS")}


def draw_roofs(ax, rf, xlim):
    x = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 400)
    ax.plot(x, np.minimum(x * rf["dram"], rf["issue"]), color=INK, lw=1.3, zorder=1)
    if rf.get("show_l2", True):
        ax.plot(x, np.minimum(x * rf["l2"], rf["issue"]), color=MUTED, lw=0.9, ls="--", zorder=1)
    if show_rng(rf):
        ax.axhline(rf["rng"], color=MUTED, lw=0.9, ls=":", zorder=1)


def show_rng(rf):
    return bool(rf["rng"]) and rf["rng"] < 0.95 * rf["issue"]


def style(ax, xlim, ylim):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(which="major", color="0.92", lw=0.6)
    ax.set_xlabel("Warp instructions per DRAM byte")
    sns.despine(ax=ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEF_DIR)
    ap.add_argument("--traj-cell", default="1,64")
    ap.add_argument("--out", default="fig_roofline_v3")
    ap.add_argument("--no-l2", action="store_true", help="omit the L2 bandwidth roof")
    a = ap.parse_args()
    rows = load(os.path.join(a.dir, "roofline_v3.csv"))
    rf = roofs(load(os.path.join(a.dir, "roofline_ceilings_v3.csv")))
    rf["show_l2"] = not a.no_l2
    heur = [r for r in rows if "heur" in r["tags"]]
    tI, tN = (float(v) for v in a.traj_cell.split(","))
    traj = [r for r in rows if "traj" in r["tags"] and r["I"] == tI and r["N"] == tN]

    xs = [r["inst_per_dram_byte"] for r in rows]
    ys = [r["warp_GIPS"] for r in rows]
    xlim = (min(xs) / 3, max(xs) * 3)
    ylim = (min(ys) / 3, rf["issue"] * 8)   # headroom for the roof legend

    fig = plt.figure(figsize=(7.2, 3.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[:, 0])
    draw_roofs(ax, rf, xlim)
    for r in heur:
        ax.scatter(r["inst_per_dram_byte"], r["warp_GIPS"], s=26, marker=MARK[r["op"]],
                   color=COLOR[r["family"]], edgecolor="white", linewidth=0.6, zorder=3)
    style(ax, xlim, ylim)
    ax.set_ylabel("Warp instructions / s ($\\times 10^9$)")
    ax.set_title("(a) launch heuristic, all cells", fontsize=9, loc="left")
    roof_ax = ax

    # Along s the points slide on the DRAM roof, so (b, c) show the two factors of
    # the time instead: time = DRAM bytes / (DRAM ceiling * fraction reached).
    bx = fig.add_subplot(gs[0, 1])
    cx = fig.add_subplot(gs[1, 1], sharex=bx)
    comp = None
    for op in MARK:
        for fam in COLOR:
            p = sorted((r for r in traj if r["op"] == op and r["family"] == fam), key=lambda r: r["cap"])
            if not p:
                continue
            comp = p[0]["compulsory_bytes"] / 1e9
            s = [r["cap"] for r in p]
            for axx, y in ((bx, [r["dram_bytes"] / 1e9 for r in p]), (cx, [r["sol_dram"] for r in p])):
                axx.plot(s, y, color=COLOR[fam], lw=1.0, marker=MARK[op], ms=3.5,
                         markeredgecolor="white", markeredgewidth=0.4)
    if comp:
        bx.axhline(comp, color=MUTED, lw=0.8, ls=":")
        bx.text(max(r["cap"] for r in traj), comp * 0.93, "read + write once", fontsize=6.5,
                color=MUTED, ha="right", va="top")
    cx.axhline(1, color=INK, lw=0.8)
    for axx in (bx, cx):
        axx.set_xscale("log", base=2)
        axx.grid(which="major", color="0.92", lw=0.6)
        sns.despine(ax=axx)
    bx.set_ylim(bottom=0)
    bx.tick_params(labelbottom=False)
    bx.set_ylabel("DRAM traffic (GB)")
    cx.set_ylabel("Share of DRAM roof")
    cx.set_ylim(0, 1.15)
    cx.set_xlabel("Split factor $s$")
    bx.set_title(f"(b, c) split factor, $I={tI:g}$, $N={tN:g}$", fontsize=9, loc="left")

    handles = [plt.Line2D([], [], ls="", marker=MARK[o], color="0.35", label=OP_NAME[o]) for o in MARK]
    handles += [plt.Line2D([], [], ls="", marker="o", color=COLOR[f], label=f) for f in COLOR]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=7.5)
    # no point can lie above the roofs, so their upper-left corner is free
    roof = [plt.Line2D([], [], color=INK, lw=1.3, label=f"DRAM {rf['dram'] / 1e3:.2f} TB/s")]
    if rf["show_l2"]:
        roof.append(plt.Line2D([], [], color=MUTED, lw=0.9, ls="--", label=f"L2 {rf['l2'] / 1e3:.1f} TB/s"))
    roof.append(plt.Line2D([], [], color=INK, lw=0, label=f"issue {rf['issue'] / 1e3:.2f} T warp-inst/s"))
    if show_rng(rf):
        roof.append(plt.Line2D([], [], color=MUTED, lw=0.9, ls=":", label="binary RNG code alone"))
    roof_ax.legend(handles=roof, loc="upper left", frameon=False, fontsize=7, handlelength=1.8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.dir, f"{a.out}.{ext}"), dpi=200)
    print("wrote", os.path.join(a.dir, a.out + ".{pdf,png}"))

    keys = ["op", "family", "I", "N", "D", "cap", "tx", "ty", "time_ns", "dram_bytes", "compulsory_bytes",
            "l2_hit_pct", "dram_GBps", "warp_GIPS", "sol_dram", "sol_issue", "sol_rng", "bound",
            "sol_compulsory"]
    with open(os.path.join(a.dir, "roofline_table_v3.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(keys)
        for r in sorted(heur, key=lambda r: (list(MARK).index(r["op"]), r["family"], r["I"], r["N"])):
            w.writerow([r[k] if isinstance(r[k], str) else f"{r[k]:.4g}" for k in keys])
    print("wrote", os.path.join(a.dir, "roofline_table_v3.csv"))


if __name__ == "__main__":
    main()
