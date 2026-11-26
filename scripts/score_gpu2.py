#!/usr/bin/env python3
"""Score the launch heuristic on a second GPU, in the t_y = 1 space of the paper.

For every case (op, family, I, N, D) measured on GPU 2 (results/gpu2/phase3_gpu2_all.csv):
  regret        t(heuristic with GPU 2's n_SM) / t(best config on GPU 2)
  regret_refsm  same, but the heuristic fed the first GPU's n_SM (188): what ignoring the
                hardware input would cost; differs from `regret` only where s changes
  transfer      t(GPU 1's best config, run on GPU 2) / t(best on GPU 2): cost of reusing
                the first GPU's tuning instead of the rule
and the same regret on GPU 1 (rerun_v3, t_y = 1) for the same cases. Times are the median
over seeds of the per-seed median over reps, as in launch_heuristic.py / analyze_fitness.py.

Writes OUT/score_gpu2_cases.csv and OUT/score_gpu2_summary.txt (OUT defaults to results/gpu2).
"""
import argparse
import csv
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_heuristic as lh  # noqa: E402
import sensitivity_pmnf as sp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def cells_ty1(paths):
    cells = defaultdict(dict)
    seeds = defaultdict(set)
    for p in paths:
        for (op, fam, I, N, D, cap, tx, ty), per_seed in sp.load(p).items():
            if ty == 1:
                cells[(op, fam, I, N, D)][(cap, tx, ty)] = st.median(per_seed.values())
                seeds[(op, fam, I, N, D)].update(per_seed)
    return cells, seeds


def regret(t, key, sm):
    cfg = lh.feasible(t, lh.launch_config(*key, sm))
    return (t[cfg] / min(t.values()), cfg) if cfg else (float("nan"), None)


def summ(xs):
    xs = [x for x in xs if x == x]
    return f"median {st.median(xs):.3f}  worst {max(xs):.3f}  (n={len(xs)})" if xs else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu2", nargs="+", default=[os.path.join(RES, "gpu2", "phase3_gpu2_all.csv")])
    ap.add_argument("--ref", default=os.path.join(RES, "rerun_v3", "phase3_unified_final_v3.csv"))
    ap.add_argument("--sm", type=int, default=84)
    ap.add_argument("--ref-sm", type=int, default=188)
    ap.add_argument("--out", default=os.path.join(RES, "gpu2"))
    a = ap.parse_args()

    g2, seeds2 = cells_ty1(a.gpu2)
    g1, _ = cells_ty1([a.ref])
    rows = []
    for key in sorted(g2, key=lambda k: (k[0], k[1], k[2], k[3])):
        t2, t1 = g2[key], g1.get(key, {})
        r2, c2 = regret(t2, key, a.sm)
        r2ref, _ = regret(t2, key, a.ref_sm)
        r1, c1 = regret(t1, key, a.ref_sm) if t1 else (float("nan"), None)
        b2 = min(t2, key=t2.get)
        b1 = min(t1, key=t1.get) if t1 else None
        tr = t2[b1] / t2[b2] if b1 in t2 else float("nan")
        rows.append(dict(op=key[0], family=key[1], I=key[2], N=key[3], D=key[4],
                         couples_per_sm2=key[2] * (key[3] // lh.NO[key[0]]) / a.sm,
                         s_changes=int(lh.launch_config(*key, a.sm)[0] != lh.launch_config(*key, a.ref_sm)[0]),
                         heur_cfg2=c2, best_cfg2=b2, regret2=r2, regret2_refsm=r2ref,
                         heur_cfg1=c1, best_cfg1=b1, regret1=r1, transfer=tr,
                         n_cfg2=len(t2), seeds2=len(seeds2[key]), best_us2=t2[b2]))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "score_gpu2_cases.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = list(rows[0]) if rows else []
        w.writerow(cols)
        for r in rows:
            w.writerow([f"{v:.4f}" if isinstance(v, float) else
                        ("/".join(map(str, v)) if isinstance(v, tuple) else v) for v in r.values()])

    L = [f"launch heuristic, t_y = 1 space; GPU 2 n_SM = {a.sm}, GPU 1 n_SM = {a.ref_sm}",
         f"GPU 2 data: {', '.join(os.path.relpath(p, RES) for p in a.gpu2)}",
         f"seeds per case on GPU 2: {sorted(set(r['seeds2'] for r in rows))}", ""]
    groups = [("de-gen + ga-gen (paper's 1.10 / 1.57 set)", lambda r: r["op"] in ("de-gen", "ga-gen")),
              ("all three ops", lambda r: True),
              ("cases where s changes with n_SM", lambda r: r["s_changes"]),
              ("cases where s is the same", lambda r: not r["s_changes"])]
    for op in lh.OPS:
        groups.append((op, lambda r, op=op: r["op"] == op))
    for name, sel in groups:
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        L.append(f"== {name}")
        L.append(f"  GPU 2, heuristic (n_SM={a.sm}):   {summ([r['regret2'] for r in rs])}")
        L.append(f"  GPU 2, heuristic (n_SM={a.ref_sm}):  {summ([r['regret2_refsm'] for r in rs])}")
        L.append(f"  GPU 2, GPU 1's tuned config:   {summ([r['transfer'] for r in rs])}")
        L.append(f"  GPU 1, heuristic (same cases): {summ([r['regret1'] for r in rs])}")
    L.append("\n== per case (regret2 / with n_SM=188 / GPU 1 regret / transfer), heuristic and best (s, t_x) on GPU 2")
    for r in rows:
        h, b = r["heur_cfg2"], r["best_cfg2"]
        L.append(f"  {r['op']:7}{r['family']:7} I={r['I']:<5}N={r['N']:<5}D={r['D']:<9} "
                 f"cpSM={r['couples_per_sm2']:8.2f}{' *' if r['s_changes'] else '  '} "
                 f"{r['regret2']:.3f} / {r['regret2_refsm']:.3f} / {r['regret1']:.3f} / {r['transfer']:.3f}   "
                 f"heur s={h[0] if h else '-'} tx={h[1] if h else '-'}  best s={b[0]} tx={b[1]}  "
                 f"({r['n_cfg2']} cfgs)")
    L.append("  * = s chosen by the heuristic differs between the two GPUs")
    txt = "\n".join(L)
    open(os.path.join(a.out, "score_gpu2_summary.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
