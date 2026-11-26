#!/usr/bin/env python3
"""Speedup over island-local compiled EvoX on both GPUs, side by side.

Both GPUs are compared in the t_y = 1 space (the only one swept on GPU 2): "tuned" is the best
(s, t_x) at t_y = 1, "heuristic" the launch rule with each GPU's n_SM. The first GPU's paper
numbers (best over all t_y, results/rerun_v3/analysis/augmented_vs_evox.csv) are printed too.
Also the EvoX and kernel times of GPU 2 relative to GPU 1, against the ratio of the two
memory bandwidths.

Writes OUT/evox_gpu2_vs_gpu1.csv and OUT/evox_gpu2_vs_gpu1.txt (OUT: results/gpu2/analysis).
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
from plot_evox_comparison import load_evox_seeds  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def cells_ty1(path):
    cells = defaultdict(dict)
    for (op, fam, I, N, D, cap, tx, ty), per_seed in sp.load(path).items():
        if ty == 1:
            cells[(op, fam, I, N, D)][(cap, tx, ty)] = st.median(per_seed.values())
    return cells


def speedups(cells, evdir, sm):
    ev = {op: load_evox_seeds(evdir, op) for op in lh.OPS}
    out = {}
    for key, t in cells.items():
        op, fam, I, N, D = key
        e = ev[op].get((I * N, D, I))
        cfg = lh.feasible(t, lh.launch_config(*key, sm))
        if e is None or cfg is None:
            continue
        best = min(t.values())
        out[key] = dict(evox_us=e[0], heur_us=t[cfg], tuned_us=best,
                        x_heur=e[0] / t[cfg], x_tuned=e[0] / best)
    return out


def rng(xs):
    return f"{st.median(xs):5.2f} ({min(xs):.2f}-{max(xs):.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu2", default=os.path.join(RES, "gpu2", "phase3_gpu2_all.csv"))
    ap.add_argument("--evox2", default=os.path.join(RES, "gpu2", "evox"))
    ap.add_argument("--gpu1", default=os.path.join(RES, "rerun_v3", "phase3_unified_final_v3.csv"))
    ap.add_argument("--evox1", default=os.path.join(RES, "rerun_v3", "evox"))
    ap.add_argument("--paper", default=os.path.join(RES, "rerun_v3", "analysis", "augmented_vs_evox.csv"))
    ap.add_argument("--sm2", type=int, default=84)
    ap.add_argument("--sm1", type=int, default=188)
    ap.add_argument("--bw-ratio", type=float, default=None,
                    help="GPU 1 / GPU 2 peak DRAM bandwidth, for context only")
    ap.add_argument("--out", default=os.path.join(RES, "gpu2", "analysis"))
    a = ap.parse_args()

    s2 = speedups(cells_ty1(a.gpu2), a.evox2, a.sm2)
    s1 = speedups(cells_ty1(a.gpu1), a.evox1, a.sm1)
    paper = {}
    for r in csv.DictReader(open(a.paper)):
        paper[(r["op"], r["family"], int(r["I"]), int(r["N"]), int(r["D"]))] = (
            float(r["speedup_heuristic"]), float(r["speedup_tuned"]))
    keys = sorted(set(s2) & set(s1), key=lambda k: (k[0], k[1], k[2], k[3]))

    L = [f"speedup over island-local compiled EvoX, median (range); t_y = 1 on both GPUs; "
         f"n_SM {a.sm2} (GPU 2) and {a.sm1} (GPU 1)", ""]
    L.append(f"{'':15}{'GPU 2 heuristic':>22}{'GPU 2 tuned':>22}{'GPU 1 heuristic':>22}{'GPU 1 tuned':>22}"
             f"{'paper heur/tuned (all t_y)':>30}")
    for op in lh.OPS:
        for fam in lh.FAMS:
            ks = [k for k in keys if k[:2] == (op, fam)]
            pk = [paper[k] for k in ks if k in paper]
            L.append(f"{op:8}{fam:7}{rng([s2[k]['x_heur'] for k in ks]):>22}{rng([s2[k]['x_tuned'] for k in ks]):>22}"
                     f"{rng([s1[k]['x_heur'] for k in ks]):>22}{rng([s1[k]['x_tuned'] for k in ks]):>22}"
                     f"{st.median(p[0] for p in pk):>15.2f} / {st.median(p[1] for p in pk):.2f}")
    slower = [k for k in keys if s2[k]["x_heur"] < 1]
    L.append(f"\nGPU 2 cases where the heuristic is slower than EvoX: {len(slower)}"
             + "".join(f"\n  {k[0]} {k[1]} I={k[2]} N={k[3]} D={k[4]}: {s2[k]['x_heur']:.2f}x "
                       f"(tuned {s2[k]['x_tuned']:.2f}x)" for k in slower))
    L.append(f"GPU 1 cases where the heuristic is slower than EvoX: {sum(s1[k]['x_heur'] < 1 for k in keys)}")

    L.append("\ntime on GPU 2 / time on GPU 1, median (range) over cells"
             + (f"; peak DRAM bandwidth ratio {a.bw_ratio:.2f}" if a.bw_ratio else ""))
    for op in lh.OPS:
        ks = [k for k in keys if k[0] == op]
        ev = {(k[2], k[3], k[4]): s2[k]["evox_us"] / s1[k]["evox_us"] for k in ks}
        L.append(f"  {op:8} EvoX {rng(list(ev.values()))}   ours tuned {rng([s2[k]['tuned_us'] / s1[k]['tuned_us'] for k in ks])}"
                 f"   ours heuristic {rng([s2[k]['heur_us'] / s1[k]['heur_us'] for k in ks])}")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "evox_gpu2_vs_gpu1.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "family", "I", "N", "D", "evox2_us", "heur2_us", "tuned2_us", "x_heur2", "x_tuned2",
                    "evox1_us", "heur1_us", "tuned1_us", "x_heur1", "x_tuned1"])
        for k in keys:
            w.writerow(list(k) + [f"{s2[k][c]:.4g}" for c in ("evox_us", "heur_us", "tuned_us", "x_heur", "x_tuned")]
                       + [f"{s1[k][c]:.4g}" for c in ("evox_us", "heur_us", "tuned_us", "x_heur", "x_tuned")])
    txt = "\n".join(L)
    open(os.path.join(a.out, "evox_gpu2_vs_gpu1.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
