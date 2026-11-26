#!/usr/bin/env python3
"""Every data-dependent number of the paper's results section, recomputed.

Reads the rerun outputs (results/rerun_v3/):
  phase3_unified_final_v3.csv            ablation grid
  analysis/sensitivity_coeffs_pmnf.csv   sensitivity_pmnf.py
  analysis/launch_heuristic_*.csv        launch_heuristic.py
  analysis/augmented_vs_evox.csv         plot_evox_comparison.py
  roofline/roofline_v3.csv               run_roofline_v3.sh
and prints the numbers grouped by the paragraph that uses them, in the paper's
order. Cells are named by the D of the real-coded operators (32-bit words per
genome); ux-bf runs the same cell with 32x the genes (bits).
"""
import argparse
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_heuristic as lh  # noqa: E402
import sensitivity_pmnf as sp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R3 = os.path.join(HERE, "..", "results", "rerun_v3")
OPS, FAMS, KNOBS = lh.OPS, lh.FAMS, ["cap", "tx", "ty"]
BITS = {"ux-bf": 32}
NO = {"de-gen": 1, "ga-gen": 2, "ux-bf": 2}


def wD(op, D):
    """Cell D in 32-bit words (the real-coded operators' D)."""
    return D // BITS.get(op, 1)


def l2(D):
    return f"2^{D.bit_length() - 1}"


def rng(xs, f="{:.2f}"):
    return f"{f.format(min(xs))}-{f.format(max(xs))} (median {f.format(st.median(xs))})"


def head(t):
    print(f"\n=== {t} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=R3)
    ap.add_argument("--sm", type=int, default=188)
    a = ap.parse_args()
    A = os.path.join(a.dir, "analysis")

    # ---------- inputs ----------
    co = defaultdict(dict)            # (op, fam, wD) -> knob -> row
    for r in csv.DictReader(open(os.path.join(A, "sensitivity_coeffs_pmnf.csv"))):
        co[(r["op"], r["family"], wD(r["op"], int(r["D"])))][r["knob"]] = r
    data = sp.load(os.path.join(a.dir, "phase3_unified_final_v3.csv"))
    cells = defaultdict(dict)         # (op, fam, I, N, D) -> (cap, tx, ty) -> median of seed medians
    for k, v in data.items():
        cells[k[:5]][k[5:8]] = st.median(v.values())
    ev = {(r["op"], r["family"], int(r["D_cell"])): r
          for r in csv.DictReader(open(os.path.join(A, "augmented_vs_evox.csv")))}
    heur = list(csv.DictReader(open(os.path.join(A, "launch_heuristic_cells.csv"))))
    roof = list(csv.DictReader(open(os.path.join(a.dir, "roofline", "roofline_v3.csv"))))
    P = lambda k, kn: float(co[k][kn]["penalty"])  # noqa: E731
    Ds = sorted({k[2] for k in co})

    head("Ablation: grid")
    print("cells (op, family, D) with a valid launch:", len(co))
    for op in OPS:
        print(f"  {op}: D (genes) {[l2(d * BITS.get(op, 1)) for d in Ds]}")

    # ---------- Table 2 and its paragraph: ga-gen vec4 ----------
    head("Table 2 (ga-gen vec4)")
    g = [("ga-gen", "vec4", D) for D in Ds]
    ci = max(max(float(co[k][kn]["pen_ci_hi"]) / P(k, kn) - 1, 1 - float(co[k][kn]["pen_ci_lo"]) / P(k, kn))
             for k in g for kn in KNOBS)
    stab = [float(co[k]["cap"]["term_stability"]) for k in g]
    print(f"caption: seed-bootstrap intervals within +-{100 * ci:.1f}%; term set recurs in "
          f"{100 * min(stab):.0f}-{100 * max(stab):.0f}% of replicates")
    for k in g:
        r = co[k]
        pens = {kn: P(k, kn) for kn in KNOBS}
        top = max(pens, key=pens.get)
        best = min(cells[("ga-gen", "vec4") + next(c[2:] for c in cells if c[:2] == ("ga-gen", "vec4") and c[4] == k[2])].items(),
                   key=lambda kv: kv[1])[0]
        print(f"  D={l2(k[2]):5} configs {r['cap']['n_configs']:>3} launches {r['cap']['n_launches']:>3} "
              f"R2 {float(r['cap']['r2']):.2f} MAPE {100 * float(r['cap']['cv_mape']):.1f}% | pen "
              + " ".join(f"{kn}={pens[kn]:.2f}{'*' if kn == top else ''}" for kn in KNOBS)
              + " | E " + " ".join(f"{kn}={r[kn]['elasticity']} [{r[kn]['ci_lo']},{r[kn]['ci_hi']}]" for kn in KNOBS)
              + f" | best (s,tx,ty)={best}")
    print("cells where s has the largest penalty:", sum(max(KNOBS, key=lambda kn: P(k, kn)) == "cap" for k in g), "of", len(g))
    print("min over cells of the largest penalty:", f"{min(max(P(k, kn) for kn in KNOBS) for k in g):.2f}")
    print("ty penalty, D <= 2^15: max", f"{max(P(k, 'ty') for k in g if k[2] <= 2 ** 15):.2f}")
    print("cells where the tx and ty penalties coincide (within 1%):",
          [l2(k[2]) for k in g if abs(P(k, "tx") / P(k, "ty") - 1) < 0.01])
    print("de-gen scalar s penalty at the smallest D:", f"{P(('de-gen', 'scalar', Ds[0]), 'cap'):.2f}")
    fit = [(l2(k[2]), float(co[k]["cap"]["r2"]), float(co[k]["cap"]["cv_mape"])) for k in g]
    print("R2 / CV-MAPE per cell:", ", ".join(f"{d}: {r2:.2f}/{100 * m:.1f}%" for d, r2, m in fit))

    # ---------- heatmap paragraph ----------
    head("Heatmap paragraph (all operators)")
    allk = list(co)
    top = {k: max(KNOBS, key=lambda kn: P(k, kn)) for k in allk}
    print(f"s is the most costly parameter in {sum(t == 'cap' for t in top.values())} of {len(allk)} cells")
    bigD = Ds[-1]
    print(f"at the largest D ({l2(bigD)}): tx/ty penalties at most "
          f"{max(P(k, kn) for k in allk if k[2] == bigD for kn in ('tx', 'ty')):.2f}, s penalty "
          + rng([P(k, "cap") for k in allk if k[2] == bigD]))
    comp = [(op, D, kn) for op in OPS for D in Ds for kn in KNOBS
            if (op, "scalar", D) in co and (op, "vec4", D) in co]
    lower = sum(P((op, "vec4", D), kn) < P((op, "scalar", D), kn) for op, D, kn in comp)
    print(f"vec4 penalty lower than scalar in {lower} of {len(comp)} comparable (op, D, knob) cells")
    for op in OPS:
        mx = {f: max(P((op, f, D), kn) for D in Ds if (op, f, D) in co for kn in KNOBS) for f in FAMS}
        print(f"  {op}: largest penalty scalar {mx['scalar']:.2f} -> vec4 {mx['vec4']:.2f}; "
              f"s penalty at the largest D: scalar {P((op, 'scalar', bigD), 'cap'):.2f}, vec4 {P((op, 'vec4', bigD), 'cap'):.2f}")
    print(f"ty penalty <= 1.10 in {sum(P(k, 'ty') <= 1.10 for k in allk)} of {len(allk)} cells")

    # ---------- heuristic derivation and Table 3 ----------
    head("Heuristic derivation")
    for r in csv.DictReader(open(os.path.join(A, "launch_heuristic_derivation.csv"))):
        print("  ", dict((k, v) for k, v in r.items() if v))
    keys = sorted(cells, key=lambda k: (k[0], k[1], k[4]))
    one, big = [], []
    for key, t in cells.items():
        op, fam, I, N, D = key
        bestby = {}
        for c, v in t.items():
            bestby[c[0]] = min(v, bestby.get(c[0], math.inf))
        worst = max(bestby, key=bestby.get)
        if bestby[worst] / min(t.values()) < 1.05:
            continue
        x = I * (N // NO[op]) / a.sm
        (one if worst == 1 else big).append((round(x, 2), op, fam, I, N))
    print("s=1 worst (couples/SM, cell):", sorted(one))
    print("large s worst, smallest couples/SM:", min(big))
    for beta in (0.17, 0.25, 0.34, 0.35, 0.5, 0.68, 0.7, 1.0, 1.36, 1.4, 2, 4, 8):
        m, p, w = lh.summary(lh.regrets(cells, keys, a.sm, beta, 16))
        print(f"  beta={beta:<5} median {m:.3f} p90 {p:.2f} worst {w:.2f}")

    head("Table 3 (regret median / worst)")
    sweep = list(csv.DictReader(open(os.path.join(A, "launch_heuristic_sweep.csv"))))
    opt = min(sweep, key=lambda r: (float(r["median"]), float(r["worst"])))
    fixed = lh.fixed_regrets(cells, keys)
    rows = {"Heuristic": lh.regrets(cells, keys, a.sm, 1.0, 16),
            f"Regret-optimal (beta={opt['beta']}, u*={opt['ustar']})":
                lh.regrets(cells, keys, a.sm, float(opt["beta"]), float(opt["ustar"])),
            "Fixed": [fixed[k] for k in keys]}
    for name, r in rows.items():
        parts = [f"{st.median(r):.2f} / {max(r):.2f}"]
        for op in OPS:
            x = [v for k, v in zip(keys, r) if k[0] == op]
            parts.append(f"{st.median(x):.2f} / {max(x):.2f}")
        print(f"  {name:36} " + " & ".join(parts) + f"   ({len(r)} cells)")

    # ---------- L2 paragraph (roofline profiles) ----------
    head("L2 paragraph: I=1, N=256, D=2^22, s=1 vs s=8 (tx=512, ty=1; ncu)")
    for op, fam in (("de-gen", "scalar"), ("ga-gen", "scalar"), ("de-gen", "vec4"), ("ga-gen", "vec4")):
        p = {int(r["cap"]): r for r in roof if r["op"] == op and r["family"] == fam
             and r["I"] == "1" and r["N"] == "256" and "traj" in r["tags"]}
        if 1 in p and 8 in p:
            f = lambda r: (f"reads {float(r['dram_read']) / 1e9:.1f} GB, writes {float(r['dram_write']) / 1e9:.1f} GB, "  # noqa: E731
                           f"L2 hit {float(r['l2_hit_pct']):.0f}%, time {float(r['time_ns']) / 1e6:.1f} ms")
            print(f"  {op} {fam}: s=1 {f(p[1])} | s=8 {f(p[8])}")

    # ---------- EvoX section ----------
    head("EvoX: baselines")
    for op in OPS:
        x = [float(r["evox_panmictic_us"]) / float(r["evox_island_us"]) for k, r in ev.items()
             if k[0] == op and k[1] == "scalar" and r["evox_panmictic_us"] != "nan" and int(r["I"]) > 1]
        print(f"  {op}: panmictic / island-local EvoX time, I>1: {rng(x)}")
    head("EvoX: speedups over island-local EvoX")
    for op in OPS:
        for fam in FAMS:
            rs = [r for k, r in ev.items() if k[:2] == (op, fam)]
            t = [float(r["speedup_tuned"]) for r in rs]
            h = [float(r["speedup_heuristic"]) for r in rs]
            print(f"  {op:7}{fam:7} tuned {rng(t)} | heuristic {rng(h)}")
    allh = [(float(r["speedup_heuristic"]), k) for k, r in ev.items()]
    print("  minimum heuristic speedup:", min(allh))
    de1024 = [float(r["speedup_tuned"]) for k, r in ev.items() if k[0] == "de-gen" and r["I"] == "1024"]
    print("  de-gen tuned speedup at I=1024 (both variants):", rng(de1024))
    for fam in FAMS:
        x = [float(r["speedup_tuned_panmictic"]) for k, r in ev.items() if k[:2] == ("de-gen", fam)]
        print(f"  de-gen {fam} tuned speedup vs panmictic EvoX: median {st.median(x):.2f}")
    for op in ("ga-gen", "ux-bf"):
        x = [abs(float(r["speedup_tuned_panmictic"]) / float(r["speedup_tuned"]) - 1) for k, r in ev.items()
             if k[0] == op]
        print(f"  {op}: baseline choice changes the tuned speedup by at most {100 * max(x):.1f}%")
    print("  per cell, heuristic vs tuned speedup:")
    for k, r in sorted(ev.items(), key=lambda kv: (OPS.index(kv[0][0]), kv[0][1], kv[0][2])):
        print(f"    {k[0]:7}{k[1]:7} D={l2(k[2]):5} I={r['I']:>4} N={r['N']:>4}: "
              f"heur {float(r['speedup_heuristic']):.2f} tuned {float(r['speedup_tuned']):.2f}")

    head("EvoX: vectorization (tuned scalar time / tuned vec4 time)")
    for op in OPS:
        x = {k[2]: float(ev[(op, 'scalar', k[2])]['tuned_us']) / float(r['tuned_us'])
             for k, r in ev.items() if k[:2] == (op, "vec4") and (op, "scalar", k[2]) in ev}
        below = {l2(d): round(v, 2) for d, v in x.items() if v < 1}
        print(f"  {op}: {rng(list(x.values()))}; vec4 slower at {below}")

    head("EvoX: launch shapes of the best configurations")
    unsplit = split1 = hybrid = 0
    rest_unsplit, rest_ty1 = [], []
    for key, t in cells.items():
        op, fam, I, N, D = key
        U = sp.units_per_row(op, fam, D)
        geff = lambda c: min(c[0], math.ceil(U / c[1]))  # noqa: E731
        b = min(t, key=t.get)
        if geff(b) == 1:
            unsplit += 1
        elif b[2] == 1:
            split1 += 1
        else:
            hybrid += 1
        tb = t[b]
        rest_unsplit.append(min(v for c, v in t.items() if geff(c) == 1) / tb)
        rest_ty1.append(min(v for c, v in t.items() if c[2] == 1) / tb)
    print(f"  of {len(cells)} best configs: {unsplit} unsplit, {split1} split with ty=1, {hybrid} hybrids (split, ty>1)")
    print(f"  unsplit only: up to {max(rest_unsplit):.2f}x (median {st.median(rest_unsplit):.2f}x); "
          f"ty=1 only: up to {max(rest_ty1):.2f}x")

    head("Heuristic regret per cell (largest first)")
    for r in sorted(heur, key=lambda r: -float(r["regret"]))[:10]:
        print(f"  {r['op']:7}{r['family']:7} I={r['I']:>4} N={r['N']:>4} D={l2(int(r['D'])):5} "
              f"heur {r['cap']},{r['tx']},{r['ty']} best {r['best_cap']},{r['best_tx']},{r['best_ty']} "
              f"regret {float(r['regret']):.2f}")


if __name__ == "__main__":
    main()
