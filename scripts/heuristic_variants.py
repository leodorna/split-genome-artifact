#!/usr/bin/env python3
"""Variants of the launch heuristic's t_x rule, scored on every data set we have.

The paper's rule t_x = clamp(pow2floor(U / 16), 32, 512) has two parts that only
matter off the ablation grid, whose U are powers of two and whose cells always have
many rows or long genomes:
  u*   the sensitivity analysis gives u* = 9.0 units per thread (geometric mean of
       the t_x flips); 16 picks the same t_x for every power-of-two U, but not for
       U = 1000 (pow2floor(62.5) = 32, pow2floor(111) = 64);
  occ  with s = 1 and few rows, t_x from the genome alone launches too few threads.
       The split rule already aims at n_SM blocks; aiming at one block of 512 threads
       per SM gives the floor t_x >= pow2ceil(n_SM * 512 / (I * C * s)).
Neither constant is fitted to the data scored here. Variants: u* in {16, 9} x
occ in {off, on}. Regret = t(choice) / t(best of the cell), ty = 1 for the transfer
data; infeasible choices fall back to a smaller t_x (launch_heuristic.feasible).

Validation cells (configs/sweep_validation.json, scripts/heuristic_validation.sh) were
chosen, and the revised rule's choices written (configs/validation_predictions.csv),
before they were measured; they are reported separately.

Usage: heuristic_variants.py [--ref results/rerun_v3] [--transfer results/transfer]
"""
import argparse
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_heuristic as lh  # noqa: E402
import sensitivity_pmnf as sp  # noqa: E402
from analyze_transfer import cells_ty1, fill_dup_caps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def config(op, fam, I, N, D, sm, ustar, occ, tmax=512):
    return lh.launch_config(op, fam, I, N, D, sm, 1.0, ustar, occ, tmax)


def score(cells, sm, ustar, occ):
    out = {}
    for k, t in cells.items():
        c = lh.feasible(t, config(*k, sm, ustar, occ))
        if c is not None:
            out[k] = t[c] / min(t.values())
    return out


def q(r):
    if not r:
        return "-"
    s = sorted(r)
    return f"{st.median(s):.2f}/{s[min(len(s) - 1, int(0.9 * len(s)))]:.2f}/{s[-1]:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(HERE, "..", "results", "rerun_v3"))
    ap.add_argument("--transfer", default=os.path.join(HERE, "..", "results", "transfer"))
    ap.add_argument("--sm", type=int, default=188)
    a = ap.parse_args()

    sets = {}
    allty = defaultdict(dict)
    for k, v in sp.load(os.path.join(a.ref, "phase3_unified_final_v3.csv")).items():
        allty[k[:5]][k[5:8]] = st.median(v.values())
    sets["Sphere/OneMax ablation (all ty)"] = allty
    for f in ("rastrigin", "rosenbrock"):
        p = os.path.join(a.ref, "fitness", f, f"phase3_{f}.csv")
        if os.path.exists(p):
            sets[f"{f} (grid)"] = cells_ty1(p)
    for f in ("cec-f1", "cec-f2", "cec-f3", "cec-f12", "knapsack"):
        p = os.path.join(a.transfer, f, f"phase3_{f}.csv")
        if not os.path.exists(p):
            continue
        c = fill_dup_caps(cells_ty1(p))
        grid = {k: v for k, v in c.items() if k[2] * k[3] * k[4] in (1 << 30, 1 << 35)}
        off = {k: v for k, v in c.items() if k not in grid}
        sets[f"{f} grid"] = grid
        if off:
            sets[f"{f} D=1000"] = off

    for f in ("cec-f2", "cec-f12", "knapsack"):
        p = os.path.join(a.transfer, "validation", f, f"phase3_{f}.csv")
        if os.path.exists(p):
            sets[f"VALIDATION {f}"] = fill_dup_caps(cells_ty1(p))

    variants = [("paper: u*=16", 16, False), ("u*=9", 9, False), ("u*=16 + occ", 16, True), ("u*=9 + occ", 9, True)]
    print(f"regret median/p90/worst  (n_SM = {a.sm}; occ: t_x >= pow2ceil(n_SM*512/(I*C*s)))")
    print(f"{'data':34}" + "".join(f"{v[0]:>18}" for v in variants))
    for name, cells in sets.items():
        print(f"{name:34}" + "".join(f"{q(list(score(cells, a.sm, u, o).values())):>18}" for _, u, o in variants)
              + f"   (n={len(cells)})")
    pooled = {}
    for _, u, o in variants:
        pooled[(u, o)] = [x for n, c in sets.items() if not n.startswith(("Sphere", "VALIDATION"))
                          for x in score(c, a.sm, u, o).values()]
    print(f"{'all transfer data pooled':34}" + "".join(f"{q(pooled[(u, o)]):>18}" for _, u, o in variants))
    val = {(u, o): [x for n, c in sets.items() if n.startswith("VALIDATION") for x in score(c, a.sm, u, o).values()]
           for _, u, o in variants}
    if any(val.values()):
        print(f"{'validation cells pooled':34}" + "".join(f"{q(val[(u, o)]):>18}" for _, u, o in variants))


if __name__ == "__main__":
    main()
