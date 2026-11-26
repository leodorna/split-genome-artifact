#!/usr/bin/env python3
"""Leave-one-cell-out validation of the launch-configuration choices.

The paper evaluates three launch choices on the same ablation cells they were chosen
on: the heuristic with its derived constants (beta = 1, u* = 16), the heuristic with
the regret-optimal constants of a (beta, u*) grid, and one minimax-regret fixed
configuration per (op, family). The last two are selected on the data, so their
in-sample regret is optimistic. Here each (I, N, D) cell is held out in
turn (all ops and families of that cell, D in 32-bit words): the regret-optimal constants and the fixed
configurations are re-selected on the other eight cells and scored on the held-out
one. The derived heuristic has nothing fitted to the regret, so its held-out regret
equals its in-sample regret; it is listed for comparison.

Regret = t(choice) / t(best config of the cell), median over seeds of per-seed medians,
an infeasible choice replaced by a smaller tx (launch_heuristic.feasible).
Writes DIR/heuristic_loo.csv and prints the summary.

Usage: heuristic_loo.py --final results/rerun_v3/phase3_unified_final_v3.csv [--dir OUT]
(reproduce_figures.sh runs it after launch_heuristic.py)
"""
import argparse
import csv
import itertools
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_heuristic as lh  # noqa: E402
import sensitivity_pmnf as sp  # noqa: E402


def regret_of(cells, k, cfg):
    t = cells[k]
    c = lh.feasible(t, cfg)
    return t[c] / min(t.values())


def heur(cells, k, sm, beta, ustar):
    return regret_of(cells, k, lh.launch_config(*k, sm, beta, ustar))


def pick_constants(cells, keys, sm, crit):
    """(beta, u*) of the grid minimizing crit (median or max) of the regret over keys."""
    best = None
    for b, u in itertools.product(lh.BETAS, lh.USTARS):
        r = [heur(cells, k, sm, b, u) for k in keys]
        score = (crit(r), st.median(r), max(r))
        if best is None or score < best[0]:
            best = (score, (b, u))
    return best[1]


def pick_fixed(cells, keys, op, fam):
    ks = [k for k in keys if k[:2] == (op, fam)]
    common = set.intersection(*[set(cells[k]) for k in ks])
    return min(common, key=lambda c: (max(cells[k][c] / min(cells[k].values()) for k in ks),
                                      st.median(cells[k][c] / min(cells[k].values()) for k in ks)))


def q(r):
    s = sorted(r)
    return f"median {st.median(s):.3f}  p90 {s[min(len(s) - 1, int(0.9 * len(s)))]:.2f}  worst {s[-1]:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--sm", type=int, default=188)
    a = ap.parse_args()

    cells = defaultdict(dict)
    for k, v in sp.load(a.final).items():
        cells[k[:5]][k[5:8]] = st.median(v.values())
    keys = sorted(cells)
    # a cell is (I, N, D) with D in 32-bit words: ux-bf's D counts bits (32x)
    cell = lambda k: (k[2], k[3], k[4] // 32 if k[0] == "ux-bf" else k[4])  # noqa: E731
    shapes = sorted({cell(k) for k in keys})
    opfam = sorted({k[:2] for k in keys})

    ins = {"derived": [heur(cells, k, a.sm, 1.0, 16) for k in keys]}
    for name, crit in (("opt-median", st.median), ("opt-worst", max)):
        b, u = pick_constants(cells, keys, a.sm, crit)
        ins[name] = [heur(cells, k, a.sm, b, u) for k in keys]
        print(f"in-sample {name} constants: beta={b}, u*={u}")
    fx = {of: pick_fixed(cells, keys, *of) for of in opfam}
    ins["fixed"] = [regret_of(cells, k, fx[k[:2]]) for k in keys]

    out = {n: {} for n in ins}
    chosen = []
    for shape in shapes:
        train = [k for k in keys if cell(k) != shape]
        test = [k for k in keys if cell(k) == shape]
        for k in test:
            out["derived"][k] = heur(cells, k, a.sm, 1.0, 16)
        for name, crit in (("opt-median", st.median), ("opt-worst", max)):
            b, u = pick_constants(cells, train, a.sm, crit)
            chosen.append((shape, name, b, u))
            for k in test:
                out[name][k] = heur(cells, k, a.sm, b, u)
        for of in opfam:
            cfg = pick_fixed(cells, train, *of)
            for k in [k for k in test if k[:2] == of]:
                out["fixed"][k] = regret_of(cells, k, cfg)
                chosen.append((shape, f"fixed {of[0]} {of[1]}", *cfg))

    print(f"\n== regret over {len(keys)} (op, family, cell) points: in-sample | held-out cell")
    label = {"derived": "heuristic, derived constants (1, 16)",
             "opt-median": "heuristic, constants re-selected (median)",
             "opt-worst": "heuristic, constants re-selected (worst)",
             "fixed": "fixed config per op/family (minimax)"}
    for n in ins:
        print(f"  {label[n]:42} {q(ins[n])} | {q([out[n][k] for k in keys])}")
    print("\n== constants re-selected per held-out cell (I, N, D): beta, u*")
    for shape in shapes:
        cs = [f"{c[1]}: {c[2]}, {c[3]}" for c in chosen if c[0] == shape and not c[1].startswith("fixed")]
        print(f"  {shape}: " + "; ".join(cs))

    with open(os.path.join(a.dir, "heuristic_loo.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "family", "I", "N", "D"] + [f"{n}_in" for n in ins] + [f"{n}_loo" for n in ins])
        for i, k in enumerate(keys):
            w.writerow(list(k) + [f"{ins[n][i]:.4f}" for n in ins] + [f"{out[n][k]:.4f}" for n in ins])
    print("\nwrote", os.path.join(a.dir, "heuristic_loo.csv"))


if __name__ == "__main__":
    main()
