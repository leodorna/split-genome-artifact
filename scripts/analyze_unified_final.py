#!/usr/bin/env python3
"""Phase 3: unified-kernel final experiment on the 9 realistic triples.

Inputs
  --final   results/rerun_v3/phase3_unified_final_v3.csv   (highdim/highdim_vec4,
            swept over cap,tx,ty; 10 seeds x 10 gens)
  --evox    results/rerun_v3/evox_autotune/evox_gen_<op>.csv  (EvoX compiled/eager band,

Aggregation: per (config, seed) median over reps, then median over seeds.

Outputs (into --outdir)
  final_best_by_cell.csv   op,family,I,N,D,total,best cap/tx/ty, us, runner-up
  final_crossover.csv      per op/family/D: best ty, best cap, best tx
  final_vs_evox.csv        per op/family/triple: unified best vs EvoX band
"""
import argparse
import csv
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def load_final(path):
    per = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            try:
                k = (r["op"], r["layout"], int(r["I"]), int(r["N"]), int(r["D"]),
                     int(r["cap"]), int(r["tx"]), int(r["ty"]), r["seed"])
                per[k].append(float(r["time_us"]))
            except (ValueError, KeyError):
                continue
    per_seed = defaultdict(list)
    for k, v in per.items():
        per_seed[k[:8]].append(st.median(v))
    return {k: st.median(v) for k, v in per_seed.items() if v}


def load_evox(evoxdir, op):
    out = {}
    path = os.path.join(evoxdir, f"evox_gen_{op}.csv")
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(int(r["N"]), int(r["D"]), r["mode"])] = float(r["per_gen_mean_us"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=os.path.join(RES, "rerun_v3", "phase3_unified_final_v3.csv"))
    ap.add_argument("--evox", default=os.path.join(RES, "rerun_v3", "evox_autotune"))
    ap.add_argument("--outdir", default=os.path.join(RES, "rerun_v3", "analysis"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    med = load_final(a.final)
    if not med:
        print(f"no ok rows in {a.final}")
        return

    fam = {"highdim": "scalar", "highdim_vec4": "vec4"}
    triples = sorted({(k[2], k[3], k[4]) for k in med}, key=lambda t: t[2])

    best_rows, cross_rows, evox_rows = [], [], []
    print(f"{'op':8}{'fam':7}{'D':>10}{'best cfg (cap,tx,ty)':>24}{'us':>10}{'2nd':>8}{'evox':>10}{'vs evox':>9}")
    for op in sorted({k[0] for k in med}):
        ev = load_evox(a.evox, op)
        for lay, family in fam.items():
            for (I, N, D) in triples:
                cand = sorted((t, k) for k, t in med.items()
                              if k[0] == op and k[1] == lay and k[2] == I and k[3] == N and k[4] == D)
                if not cand:
                    continue
                m, k = cand[0]
                second = cand[1][0] if len(cand) > 1 else m
                best_rows.append([op, family, I, N, D, I * N, k[5], k[6], k[7],
                                  f"{m:.1f}", f"{second:.1f}",
                                  f"{second / m:.3f}" if m else "", len(cand)])
                # crossover: the joint-best config for this cell
                cross_rows.append([op, family, D, I, N, k[6], k[5], k[7], f"{m:.1f}"])
                em = ev.get((I * N, D, "compiled"))
                evox_rows.append([op, family, I, N, D, I * N, f"{m:.1f}",
                                  f"{em:.1f}" if em else "", f"{em / m:.3f}" if em else ""])
                print(f"{op:8}{family:7}{D:>10}{'('+str(k[5])+','+str(k[6])+','+str(k[7])+')':>24}"
                      f"{m:>10.0f}{second/m:>8.3f}{(em or 0):>10.0f}{(em/m if em else 0):>9.3f}")

    with open(os.path.join(a.outdir, "final_best_by_cell.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "family", "I", "N", "D", "total", "cap", "tx", "ty",
                    "us", "second_us", "second_over_best", "n_configs"])
        w.writerows(best_rows)
    with open(os.path.join(a.outdir, "final_crossover.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "family", "D", "I", "N", "best_tx", "best_cap", "best_ty", "best_us"])
        w.writerows(cross_rows)
    with open(os.path.join(a.outdir, "final_vs_evox.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "family", "I", "N", "D", "total", "unified_us", "evox_compiled_us", "evox_over_unified"])
        w.writerows(evox_rows)
    print("\nwrote " + os.path.join(a.outdir, "final_best_by_cell.csv"))
    print("wrote " + os.path.join(a.outdir, "final_crossover.csv"))
    print("wrote " + os.path.join(a.outdir, "final_vs_evox.csv"))


if __name__ == "__main__":
    main()
