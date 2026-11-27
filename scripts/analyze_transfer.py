#!/usr/bin/env python3
"""Transfer of the ablation's launch guidance to harder functions (transfer_campaign.sh).

The launch heuristic and the fixed configurations were derived on Sphere and OneMax.
Here they are evaluated, unchanged, on CEC 2013 LSGO f1, f2, f3, f12 (de-gen, ga-gen)
and the 0/1 knapsack (ux-bf), within the campaign's launch space (every s and tx,
ty = 1), on the nine ablation cells and three D = 1000 cells. Per cell and variant:

  regret_first    t(first rule: u* = 16, no occupancy floor) / t(best config of the cell)
  regret_heur     the same for the revised rule (paper Eq. 7: u* = 9, occupancy floor),
                  which is also the configuration of speedup_heur and cost_vs_sphere
  regret_sphere   t(best config of the cell on Sphere/OneMax) / t(best): per-cell
                  tuning done on Sphere, reused (nine grid cells only)
  regret_fixed    t(Sphere's minimax-regret fixed config of the op/variant) / t(best)
  speedup_heur/tuned   island-local compiled baseline / ours (baselines in --evox-sub:
                       evox_autotune, max-autotune-no-cudagraphs, by default)
  cost_vs_sphere  t_f(heuristic) / t_Sphere(heuristic), the price of the fitness

Caps the campaign skipped as repeats of a launch (DEDUP_CAPS) take the time of the
launch they repeat. Times are medians over seeds of per-seed medians; an infeasible heuristic choice
(register limit) is replaced by a smaller tx, as in the paper (launch_heuristic.feasible).
Also scores both rules on the validation cells (DIR/validation/<f>/, scripts/
heuristic_validation.sh). Writes OUT/transfer_cells.csv, transfer_summary.txt and fig_transfer.{pdf,png}.

Usage: analyze_transfer.py [--dir results/transfer] [--ref results/rerun_v3] [--sm 188]
"""
import argparse
import csv
import itertools
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_heuristic as lh  # noqa: E402
import sensitivity_pmnf as sp  # noqa: E402
from plot_evox_comparison import load_evox_seeds  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FITS = ["cec-f1", "cec-f2", "cec-f3", "cec-f12", "knapsack"]
NAME = {"cec-f1": "f1", "cec-f2": "f2", "cec-f3": "f3", "cec-f12": "f12", "knapsack": "knapsack"}
OPS = {"knapsack": ["ux-bf"]}
FAMS = ["scalar", "vec4"]
COLOR = {"scalar": "#2a78d6", "vec4": "#eb6834"}
MARK = {"de-gen": "o", "ga-gen": "^", "ux-bf": "s"}
# the paper names the kernels de, sbx-pm and ux-bf; the code and data keep de-gen and ga-gen
PAPER_NAME = {"de-gen": "de", "ga-gen": "sbx-pm"}


def ops_for(fit):
    return OPS.get(fit, ["de-gen", "ga-gen"])


def cells_ty1(path):
    """(op, fam, I, N, D) -> {(cap, tx, ty): median over seeds}, ty = 1 only."""
    cells = defaultdict(dict)
    for (op, fam, I, N, D, cap, tx, ty), per_seed in sp.load(path).items():
        if ty == 1:
            cells[(op, fam, I, N, D)][(cap, tx, ty)] = st.median(per_seed.values())
    return cells


REV_USTAR = 9   # revised rule (paper Eq. 7): derived u*, occupancy floor
CAPS = [1 << k for k in range(12)]   # 1 ... 2048, the sweep's caps


def fill_dup_caps(cells):
    """The campaign skips caps that repeat a launch (DEDUP_CAPS): the kernel launches
    min(cap, ceil(U/tx)) blocks per row, so every cap above the first power of two
    >= ceil(U/tx) has that cap's time. Fill them in, so that configurations chosen
    on the full Sphere sweep can be looked up."""
    for (op, fam, I, N, D), t in cells.items():
        U = sp.units_per_row(op, fam, D)
        for (cap, tx, ty) in list(t):
            c0 = lh.p2c(-(-U // tx))
            if cap == c0:
                for c in CAPS:
                    if c > c0:
                        t.setdefault((c, tx, ty), t[(cap, tx, ty)])
    return cells


def evox(path, op):
    f = os.path.join(path, f"evox_gen_{op}.csv")
    return load_evox_seeds(path, op) if os.path.exists(f) else {}


def minimax_fixed(cells):
    """Sphere's minimax-regret fixed config per (op, fam), within ty = 1."""
    out = {}
    for op, fam in itertools.product(["de-gen", "ga-gen", "ux-bf"], FAMS):
        ks = [k for k in cells if k[:2] == (op, fam)]
        if not ks:
            continue
        common = set.intersection(*[set(cells[k]) for k in ks])
        out[(op, fam)] = min(common, key=lambda c: max(cells[k][c] / min(cells[k].values()) for k in ks))
    return out


def q(xs):
    """median, 90th percentile, max."""
    s = sorted(xs)
    return st.median(s), s[min(len(s) - 1, int(0.9 * len(s)))], s[-1]


def fmt_q(xs):
    if not xs:
        return "-"
    m, p, w = q(xs)
    return f"{m:.2f}/{p:.2f}/{w:.2f} (n={len(xs)})"


def fmt_r(xs):
    if not xs:
        return "-"
    return f"{min(xs):.1f}-{max(xs):.1f} (median {st.median(xs):.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "..", "results", "transfer"))
    ap.add_argument("--ref", default=os.path.join(HERE, "..", "results", "rerun_v3"))
    ap.add_argument("--sm", type=int, default=188)
    ap.add_argument("--evox-sub", default="evox_autotune",
                    help="baseline directory inside DIR/<f>/ (evox: default compile mode)")
    ap.add_argument("--ref-evox", default=None,
                    help="Sphere/OneMax baselines for the price of the fitness (default: REF/evox_autotune)")
    a = ap.parse_args()

    ref_path = os.path.join(a.ref, "phase3_unified_final_v3.csv")
    if not os.path.exists(ref_path):
        sys.exit(f"missing {ref_path} (gunzip -k {ref_path}.gz)")
    sph = cells_ty1(ref_path)
    sph_fixed = minimax_fixed(sph)
    ref_evox = a.ref_evox or os.path.join(a.ref, "evox_autotune")
    sph_ev = {op: evox(ref_evox, op) for op in ("de-gen", "ga-gen", "ux-bf")}

    rows = []
    for fit in FITS:
        kpath = os.path.join(a.dir, fit, f"phase3_{fit}.csv")
        if not os.path.exists(kpath):
            print("missing", kpath)
            continue
        cells = fill_dup_caps(cells_ty1(kpath))
        ev = {op: evox(os.path.join(a.dir, fit, a.evox_sub), op) for op in ops_for(fit)}
        for (op, fam, I, N, D), t in sorted(cells.items()):
            cfg0 = lh.feasible(t, lh.launch_config(op, fam, I, N, D, a.sm))
            cfg = lh.feasible(t, lh.launch_config(op, fam, I, N, D, a.sm, 1.0, REV_USTAR, True))
            if cfg is None or cfg0 is None:
                continue
            best = min(t, key=t.get)
            grid = (I * N * D == (1 << 30)) or (op == "ux-bf" and I * N * D == (1 << 35))
            r = dict(fitness=fit, op=op, family=fam, I=I, N=N, D=D, grid=int(grid),
                     heur_cfg="/".join(map(str, cfg)), tuned_cfg="/".join(map(str, best)),
                     heur_us=t[cfg], tuned_us=t[best], regret_heur=t[cfg] / t[best],
                     first_cfg="/".join(map(str, cfg0)), regret_first=t[cfg0] / t[best])
            s = sph.get((op, fam, I, N, D))
            if s:
                sb = min(s, key=s.get)
                if sb in t:
                    r["regret_sphere"] = t[sb] / t[best]
                scfg = lh.feasible(s, lh.launch_config(op, fam, I, N, D, a.sm))
                if scfg:
                    r["cost_vs_sphere"] = t[cfg] / s[scfg]
            fx = sph_fixed.get((op, fam))
            if fx and fx in t:
                r["regret_fixed"] = t[fx] / t[best]
            e = ev[op].get((I * N, D, I))
            if e:
                r["evox_us"] = e[0]
                r["speedup_heur"] = e[0] / t[cfg]
                r["speedup_tuned"] = e[0] / t[best]
                se = sph_ev[op].get((I * N, D, I))
                if se:
                    r["evox_cost_vs_sphere"] = e[0] / se[0]
            rows.append(r)

    if not rows:
        sys.exit("no transfer data yet")
    lines = ["== launch guidance derived on Sphere/OneMax, evaluated on each function (ty = 1)",
             "   regret = t(choice) / t(best config of the cell); median/p90/max over cells and variants"]
    for fit in FITS:
        for part, sel in (("grid", 1), ("D=1000", 0)):
            rs = [r for r in rows if r["fitness"] == fit and r["grid"] == sel]
            if not rs:
                continue
            lines.append(f"  {NAME[fit]:8} {part:6}  first rule {fmt_q([r['regret_first'] for r in rs])}"
                         f" | revised {fmt_q([r['regret_heur'] for r in rs])}"
                         f" | Sphere-tuned per cell {fmt_q([r['regret_sphere'] for r in rs if 'regret_sphere' in r])}"
                         f" | Sphere's fixed config {fmt_q([r['regret_fixed'] for r in rs if 'regret_fixed' in r])}")
    allh = [r["regret_heur"] for r in rows]
    lines.append(f"  all functions: first rule {fmt_q([r['regret_first'] for r in rows])}; revised {fmt_q(allh)}; "
                 f"cells within 10% of the best (revised): {sum(x <= 1.10 for x in allh)} of {len(allh)}")

    lines.append("\n== validation cells (configs/sweep_validation.json, chosen and predicted before measuring)")
    vall = {"first": [], "rev": [], "fixed": []}
    for fit in FITS:
        vpath = os.path.join(a.dir, "validation", fit, f"phase3_{fit}.csv")
        if not os.path.exists(vpath):
            continue
        vr = {"first": [], "rev": [], "fixed": []}
        for (op, fam, I, N, D), t in sorted(fill_dup_caps(cells_ty1(vpath)).items()):
            b = min(t.values())
            c0 = lh.feasible(t, lh.launch_config(op, fam, I, N, D, a.sm))
            c1 = lh.feasible(t, lh.launch_config(op, fam, I, N, D, a.sm, 1.0, REV_USTAR, True))
            vr["first"].append(t[c0] / b)
            vr["rev"].append(t[c1] / b)
            fx = sph_fixed.get((op, fam))
            if fx and fx in t:
                vr["fixed"].append(t[fx] / b)
        for k in vall:
            vall[k] += vr[k]
        lines.append(f"  {NAME[fit]:8} first rule {fmt_q(vr['first'])} | revised {fmt_q(vr['rev'])}"
                     f" | Sphere's fixed config {fmt_q(vr['fixed'])}")
    if vall["rev"]:
        d = [r / f for r, f in zip(vall["rev"], vall["first"])]
        lines.append(f"  all      first rule {fmt_q(vall['first'])} | revised {fmt_q(vall['rev'])}"
                     f" | Sphere's fixed config {fmt_q(vall['fixed'])}")
        lines.append(f"  revised vs first: faster in {sum(x < 0.98 for x in d)}, slower in {sum(x > 1.02 for x in d)} "
                     f"(at most {max(d):.2f}x), within 2% in {sum(0.98 <= x <= 1.02 for x in d)} of {len(d)}")

    lines.append("\n== speedup over the island-local compiled baseline: heuristic | tuned (min-max, median)")
    for fit in FITS:
        for op in ops_for(fit):
            for fam in FAMS:
                rs = [r for r in rows if (r["fitness"], r["op"], r["family"]) == (fit, op, fam) and "speedup_heur" in r]
                for part, sel in (("grid", 1), ("D=1000", 0)):
                    p = [r for r in rs if r["grid"] == sel]
                    if p:
                        lines.append(f"  {NAME[fit]:8} {op:7} {fam:6} {part:6} {fmt_r([r['speedup_heur'] for r in p])}"
                                     f" | {fmt_r([r['speedup_tuned'] for r in p])}")

    lines.append("\n== price of the fitness: time / Sphere's at the heuristic (ours) and baseline / its Sphere")
    for fit in FITS:
        for op in ops_for(fit):
            rs = [r for r in rows if (r["fitness"], r["op"]) == (fit, op) and r["grid"] == 1]
            lines.append(f"  {NAME[fit]:8} {op:7} ours {fmt_r([r['cost_vs_sphere'] for r in rs if 'cost_vs_sphere' in r])}"
                         f" | baseline {fmt_r([r['evox_cost_vs_sphere'] for r in rs if 'evox_cost_vs_sphere' in r])}")

    lines.append("\n== cells where the heuristic is more than 10% off: heuristic vs best (cap/tx/ty)")
    for r in sorted(rows, key=lambda r: -r["regret_heur"]):
        if r["regret_heur"] <= 1.10:
            break
        lines.append(f"  {NAME[r['fitness']]:8} {r['op']:7} {r['family']:6} I={r['I']:<5} N={r['N']:<5} D={r['D']:<9} "
                     f"{r['regret_heur']:.2f}  {r['heur_cfg']:>11} vs {r['tuned_cfg']}  (first rule {r['regret_first']:.2f})")

    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(a.dir, "transfer_summary.txt"), "w").write(txt + "\n")
    keys = ["fitness", "op", "family", "I", "N", "D", "grid", "first_cfg", "heur_cfg", "tuned_cfg", "heur_us",
            "tuned_us", "regret_first", "regret_heur", "regret_sphere", "regret_fixed", "evox_us", "speedup_heur", "speedup_tuned",
            "cost_vs_sphere", "evox_cost_vs_sphere"]
    with open(os.path.join(a.dir, "transfer_cells.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(keys)
        for r in rows:
            w.writerow([f"{r[k]:.4g}" if isinstance(r.get(k), float) else r.get(k, "") for k in keys])

    # figure: (a) heuristic regret, (b) speedup at the heuristic; one column per function,
    # scalar and vec4 side by side, D = 1000 cells hollow
    fits = [f for f in FITS if any(r["fitness"] == f for r in rows)]
    sns.set_theme(context="paper", style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, metric, ylab in ((axes[0], "regret_heur", "heuristic / best time"),
                             (axes[1], "speedup_heur", "speedup over baseline (heuristic)")):
        for i, fit in enumerate(fits):
            for j, fam in enumerate(FAMS):
                x0 = i + (j - 0.5) * 0.32
                for k, op in enumerate(ops_for(fit)):
                    dx = (k - 0.5) * 0.1 if len(ops_for(fit)) > 1 else 0.0
                    for sel, face in ((1, COLOR[fam]), (0, "none")):
                        y = [r[metric] for r in rows if r["fitness"] == fit and r["family"] == fam
                             and r["op"] == op and r["grid"] == sel and metric in r]
                        if y:
                            ax.scatter(x0 + dx + np.linspace(-0.03, 0.03, len(y)), y, s=16, marker=MARK[op],
                                       facecolors=face, edgecolors=COLOR[fam], linewidths=0.9, zorder=3)
        ax.set_xticks(range(len(fits)))
        ax.set_xticklabels([NAME[f] for f in fits], fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="0.92", lw=0.6)
        sns.despine(ax=ax)
    axes[0].axhline(1.0, color="0.6", lw=0.8, zorder=1)
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].axhline(1.0, color="0.6", lw=0.8, zorder=1)
    h = [plt.Line2D([], [], ls="", marker="o", color=COLOR["scalar"], label="scalar"),
         plt.Line2D([], [], ls="", marker="o", color=COLOR["vec4"], label="vec4"),
         plt.Line2D([], [], ls="", marker="o", color="0.4", label="de"),
         plt.Line2D([], [], ls="", marker="^", color="0.4", label="sbx-pm"),
         plt.Line2D([], [], ls="", marker="s", color="0.4", label="ux-bf"),
         plt.Line2D([], [], ls="", marker="o", mfc="none", color="0.4", label="D = 1000 (hollow)")]
    fig.legend(handles=h, loc="outside upper center", ncol=6, frameon=False, fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.dir, f"fig_transfer.{ext}"), dpi=200)
    print("\nwrote transfer_cells.csv, transfer_summary.txt, fig_transfer.{pdf,png} in", a.dir)


if __name__ == "__main__":
    main()
