#!/usr/bin/env python3
"""Launch-configuration heuristic for the unified HighDim kernel: derivation and evaluation.

launch_config(op, family, I, N, D, sm_count) -> (cap, tx, ty) uses only the problem
shape (I, N, D) and the SM count, plus two fixed properties of the kernel variant:
the units per row U (D, D/4 float4, D/32 bit-packed, D/128 uint4) and the couples
per island C (N for DE, N/2 for GA/UX):

    tx  = clamp(pow2floor(U / u*), 32, 512)
    ty  = 1
    cap = pow2ceil(beta * sm_count / (I * C))     (= 1 when I*C >= beta * sm_count)

(cap above ceil(U/tx) needs no clip: the kernel launches min(cap, ceil(U/tx)) blocks.)

The constants are *derived* from the sensitivity analysis (sensitivity_pmnf.py), not
fitted to the regret:
  ty   penalty of ty with tx and cap re-tuned is small -> fix ty = 1.
  u*   E_tx changes sign (narrower -> wider blocks pay off) between two D values; tx
       should reach its maximum 512 there, so u* = geometric mean over op/family of
       U_flip / 512, with U_flip the geometric midpoint of U at the sign change.
  beta for every cell where cap matters (penalty >= 1.05), the worst cap setting is
       either cap = 1 (rows too few: splitting needed) or a large cap (splitting
       hurts); the switch lies between the largest rows-per-SM value where cap = 1 is
       still always the worst and the smallest value where a large cap always is.
       beta = 1 (one row per SM) lies in that interval.

Evaluation against the augmented ablation grid (median of per-seed medians):
regret = t(heuristic) / t(best config of the cell), with an infeasible choice (register
limit) replaced by a smaller tx. Compared with (a) the regret-optimal constants and
(b) one fixed minimax-regret config per op/family. Also reports the speedup over
island-local compiled EvoX with the heuristic's config instead of per-cell tuning.

Outputs results/rerun_v3/analysis/launch_heuristic_{cells,sweep,derivation}.csv
"""
import argparse
import csv
import itertools
import math
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sensitivity_pmnf as sp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NO = {"de-gen": 1, "ga-gen": 2, "ux-bf": 2}
OPS = ["de-gen", "ga-gen", "ux-bf"]
FAMS = ["scalar", "vec4"]
LAYOUT = {"scalar": "highdim", "vec4": "highdim_vec4"}
BETAS = [0.25, 0.5, 1, 2, 4, 8]
USTARS = [8, 16, 32]


def p2f(x):
    return 1 << max(0, int(math.floor(math.log2(max(x, 1)))))


def p2c(x):
    return 1 << max(0, int(math.ceil(math.log2(max(x, 1)))))


def launch_config(op, family, I, N, D, sm_count, beta=1.0, ustar=16, occ=False, tmax=512):
    """occ=False, ustar=16: the first rule, kept for the sensitivity sweep. Equation 14 of
    the paper: ustar=9 (the derived geometric mean) and occ=True,
    a floor of one block of tmax threads per SM, t_x >= pow2ceil(n_SM tmax / (I C s)),
    and at most pow2floor(U) threads (no thread without a unit)."""
    U = sp.units_per_row(op, family, D)
    C = N // NO[op]
    cap = min(p2c(beta * sm_count / (I * C)), 2048)
    tx = p2f(U / ustar)
    if occ:   # never wider than the genome: at most pow2floor(U) threads
        tx = min(max(tx, p2c(sm_count * tmax / (I * C * cap))), p2f(U))
    return cap, min(512, max(32, tx)), 1


def feasible(t, cfg):
    cap, tx, ty = cfg
    for tyy in [ty] + [y for y in (8, 4, 2, 1) if y < ty]:
        for txx in [tx] + [x for x in (256, 128, 64, 32) if x < tx]:
            if (cap, txx, tyy) in t:
                return cap, txx, tyy
    return None


def regrets(cells, keys, sm, beta, ustar, occ=False):
    return [cells[k][feasible(cells[k], launch_config(*k, sm, beta, ustar, occ))] / min(cells[k].values())
            for k in keys]


def summary(r):
    s = sorted(r)
    return st.median(r), s[min(len(s) - 1, int(0.9 * len(s)))], max(r)


def fixed_regrets(cells, keys):
    """Per cell regret of one minimax-regret fixed config per (op, family)."""
    out = {}
    for op, fam in itertools.product(OPS, FAMS):
        ks = [k for k in keys if k[:2] == (op, fam)]
        common = set.intersection(*[set(cells[k]) for k in ks])
        cfg = min(common, key=lambda c: max(cells[k][c] / min(cells[k].values()) for k in ks))
        for k in ks:
            out[k] = cells[k][cfg] / min(cells[k].values())
    return out


def derive(cells, sens_csv, sm):
    """Constants of the heuristic from the sensitivity analysis. Returns (u*, beta_lo, beta_hi, rows)."""
    E, pen = defaultdict(dict), defaultdict(dict)
    for r in csv.DictReader(open(sens_csv)):
        k = (r["op"], r["family"])
        pen[k].setdefault(int(r["D"]), {})[r["knob"]] = float(r["penalty"])
        if r["knob"] == "tx":
            E[k][int(r["D"])] = float(r["elasticity"])
    out = []
    typ = [p["ty"] for d in pen.values() for p in d.values()]
    print(f"ty: penalty <= 1.10 in {sum(p <= 1.10 for p in typ)}/{len(typ)} cells -> ty = 1")

    ratios = []
    for (op, fam), d in sorted(E.items()):
        Ds = sorted(d)
        i = next(i for i in range(len(Ds) - 1) if d[Ds[i]] > 0 and d[Ds[i + 1]] <= 0)
        u = math.sqrt(sp.units_per_row(op, fam, Ds[i]) * sp.units_per_row(op, fam, Ds[i + 1])) / 512
        ratios.append(u)
        out.append(["tx_flip", op, fam, Ds[i], Ds[i + 1], f"{u:.2f}"])
        print(f"tx: {op:7}{fam:7} E_tx turns <= 0 between D=2^{Ds[i].bit_length()-1} and "
              f"2^{Ds[i+1].bit_length()-1}: U/512 = {u:.1f}")
    ustar = math.exp(st.mean(math.log(u) for u in ratios))
    print(f"tx: u* = geometric mean = {ustar:.1f} units per thread")

    one, big = [], []
    for key, t in cells.items():
        op, fam, I, N, D = key
        bestby = {}
        for c, v in t.items():
            bestby[c[0]] = min(v, bestby.get(c[0], math.inf))
        worst = max(bestby, key=bestby.get)
        if bestby[worst] / min(t.values()) < 1.05:
            continue
        (one if worst == 1 else big).append(I * (N // NO[op]) / sm)
    lo = max(x for x in one if x < min(big))     # largest rows/SM where cap=1 is the only worst kind
    # smallest rows/SM where only large caps are worst; the one cap=1 case far above
    # (a DE island cell whose donors fit in L2, 174 rows/SM) is reported, not used
    hi = min(x for x in big if x > max(x for x in one if x <= 10))
    print(f"cap: cap=1 is the worst setting at rows/SM in {sorted(round(x, 2) for x in one)}")
    print(f"cap: a large cap is the worst setting from rows/SM = {min(big):.2f} upward")
    print(f"cap: switch between {lo:.2f} and {hi:.2f} rows per SM -> beta = 1 (one row per SM)")
    out += [["ustar", "", "", "", "", f"{ustar:.2f}"], ["beta_interval", "", "", "", f"{lo:.3f}", f"{hi:.3f}"]]
    return ustar, lo, hi, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=os.path.join(HERE, "..", "results", "rerun_v3", "phase3_unified_final_v3.csv"))
    ap.add_argument("--sens", default=os.path.join(HERE, "..", "results", "rerun_v3", "analysis", "sensitivity_coeffs_pmnf.csv"))
    ap.add_argument("--evox", default=os.path.join(HERE, "..", "results", "rerun_v3", "evox_autotune"))
    ap.add_argument("--dir", default=os.path.join(HERE, "..", "results", "rerun_v3", "analysis"))
    ap.add_argument("--sm", type=int, default=188)
    a = ap.parse_args()

    data = sp.load(a.final)
    cells = defaultdict(dict)
    for k, v in data.items():
        cells[k[:5]][k[5:8]] = st.median(v.values())
    keys = sorted(cells, key=lambda k: (k[0], k[1], k[4]))

    print("== derivation from the sensitivity analysis ==")
    ustar_d, lo, hi, drows = derive(cells, a.sens, a.sm)
    # Equation 14: u* = 9 (the derived geometric mean) and the occupancy term b. On the nine
    # cells (power-of-two U, many rows for short genomes) it picks the same configurations
    # as u* = 16 without b, the first rule that launch_heuristic_sweep.csv varies.
    beta, ustar, occ = 1.0, 9, True
    assert all(p2f(u / ustar_d) == p2f(u / ustar) for u in (2 ** k for k in range(5, 25)))
    assert all(launch_config(*k, a.sm, beta, ustar, occ) == launch_config(*k, a.sm, beta, 16, False)
               for k in keys)
    with open(os.path.join(a.dir, "launch_heuristic_derivation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "op", "family", "D_lo", "D_hi_or_lo", "value"])
        w.writerows(drows)

    print(f"\n== evaluation of Equation 14 (beta={beta}, u*={ustar}, occupancy term b, SMs={a.sm}) ==")
    fixed = fixed_regrets(cells, keys)
    per = regrets(cells, keys, a.sm, beta, ustar, occ)
    for name, r in (("heuristic (derived)", per),
                    ("regret-optimal constants (beta=2)", regrets(cells, keys, a.sm, 2, 16)),
                    ("fixed config per op/family", [fixed[k] for k in keys])):
        m, p, w = summary(r)
        byop = "  ".join(f"{op} {st.median([x for k, x in zip(keys, r) if k[0] == op]):.3f}/"
                         f"{max(x for k, x in zip(keys, r) if k[0] == op):.2f}" for op in OPS)
        print(f"  {name:34} median {m:.3f}  p90 {p:.2f}  worst {w:.2f} | {byop}")
    for b in (lo, math.sqrt(lo * hi), 1.0, hi):
        m, p, w = summary(regrets(cells, keys, a.sm, b, ustar, occ))
        print(f"  beta={b:.2f} (inside the switch interval): median {m:.3f}  p90 {p:.2f}  worst {w:.2f}")

    rows = []
    for (b, u) in itertools.product(BETAS, USTARS):
        m, p, w = summary(regrets(cells, keys, a.sm, b, u))
        rows.append([b, u, f"{m:.4f}", f"{p:.4f}", f"{w:.4f}"])
    with open(os.path.join(a.dir, "launch_heuristic_sweep.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["beta", "ustar", "median", "p90", "worst"]] + rows)

    # speedup over island-local compiled EvoX, heuristic vs tuned
    from plot_evox_comparison import load_evox_seeds  # noqa: E402
    sp_h, sp_t = defaultdict(list), defaultdict(list)
    with open(os.path.join(a.dir, "launch_heuristic_cells.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "family", "I", "N", "D", "cap", "tx", "ty", "best_cap", "best_tx", "best_ty",
                    "heur_us", "best_us", "regret", "fixed_regret", "evox_island_us"])
        ev = {op: load_evox_seeds(a.evox, op) for op in OPS}
        for key in keys:
            op, fam, I, N, D = key
            t = cells[key]
            c = feasible(t, launch_config(*key, a.sm, beta, ustar, occ))
            b = min(t, key=t.get)
            e = ev[op].get((I * N, D, I))
            if e:
                sp_h[(op, fam)].append(e[0] / t[c]); sp_t[(op, fam)].append(e[0] / t[b])
            w.writerow(list(key) + list(c) + list(b) + [f"{t[c]:.1f}", f"{t[b]:.1f}", f"{t[c] / t[b]:.4f}",
                                                        f"{fixed[key]:.4f}", f"{e[0]:.1f}" if e else ""])
    print("\n== speedup over island-local compiled EvoX: heuristic (tuned) ==")
    for k in sorted(sp_h):
        h, tt = sp_h[k], sp_t[k]
        print(f"  {k[0]:7}{k[1]:7} median {st.median(h):.2f}x ({st.median(tt):.2f}x)  "
              f"range {min(h):.2f}-{max(h):.2f}x ({min(tt):.2f}-{max(tt):.2f}x)")


if __name__ == "__main__":
    main()
