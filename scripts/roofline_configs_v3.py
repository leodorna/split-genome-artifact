#!/usr/bin/env python3
"""Launch configurations for the roofline profiles (run_roofline_v3.sh).

Prints one line per unique config, "tags op layout I N D tx ty cap", where tags is
a '+'-joined subset of
  heur  the launch heuristic (launch_heuristic.launch_config) for every op, family
        and sweep cell;
  traj  split factor s = 1, 2, 4, ... up to fully split (s >= ceil(U/tx)) at the
        heuristic's tx and ty, for the cells in --traj;
  best  the fastest config of each cell of --best (an ablation CSV, median of
        per-seed medians), if given.
D is in genes, so in bits for ux-bf (32x the sweep tuple, as in the ablation). A
config that the ablation would skip (threads * regs > 65536, tx > U) gets the next
smaller tx, as in launch_heuristic.feasible.
"""
import argparse
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_heuristic as lh  # noqa: E402
import sensitivity_pmnf as sp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
REG_TOKEN = {"de-gen": "DE1bin", "ga-gen": "SBXCrossover", "ux-bf": "UX"}


def load_regs(path):
    regs = {}
    for line in open(path):
        f = line.strip().split(",")
        if len(f) >= 4 and f[3].isdigit():
            regs[(f[0], f[1], f[2])] = int(f[3])
    return regs


def regs_for(regs, op, fam):
    return regs.get((REG_TOKEN[op], "highdim", "1" if fam == "vec4" else "0"), 255)


def fit(tx, U, R):
    """Largest power-of-two tx <= the requested one that the ablation would run."""
    while tx > 32 and (tx > U or tx * R > 65536):
        tx //= 2
    return tx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=os.path.join(HERE, "..", "configs", "sweep_realistic.json"))
    ap.add_argument("--regs", default=os.path.join(RES, "kernel_regs_v2.csv"))
    ap.add_argument("--sm", type=int, default=188)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--ustar", type=float, default=16)
    ap.add_argument("--traj", default="1,64 1,256", help='cells "I,N ..." for the split-factor paths')
    ap.add_argument("--max-cap", type=int, default=2048)
    ap.add_argument("--best", default="", help="ablation CSV; adds the best config of each cell")
    a = ap.parse_args()

    tuples = json.load(open(a.sweep))["tuples"]
    traj = {tuple(int(v) for v in c.split(",")) for c in a.traj.split()}
    regs = load_regs(a.regs)
    tags = defaultdict(set)

    for op in lh.OPS:
        for fam in lh.FAMS:
            lay = lh.LAYOUT[fam]
            R = regs_for(regs, op, fam)
            for I, N, Dt in tuples:
                D = Dt * 32 if op == "ux-bf" else Dt
                U = sp.units_per_row(op, fam, D)
                cap, tx, ty = lh.launch_config(op, fam, I, N, D, a.sm, a.beta, a.ustar)
                tx = fit(tx, U, R)
                tags[(op, lay, I, N, D, tx, ty, cap)].add("heur")
                if (I, N) in traj:
                    full = math.ceil(U / tx)
                    s = 1
                    while True:
                        tags[(op, lay, I, N, D, tx, ty, s)].add("traj")
                        if s >= full or s >= a.max_cap:
                            break
                        s *= 2

    if a.best:
        seeds = sp.load(a.best)              # (op, fam, I, N, D, cap, tx, ty) -> {seed: median}
        best = {}
        for (op, fam, I, N, D, cap, tx, ty), per_seed in seeds.items():
            t = st.median(per_seed.values())
            k = (op, fam, I, N, D)
            if k not in best or t < best[k][0]:
                best[k] = (t, cap, tx, ty)
        for (op, fam, I, N, D), (_, cap, tx, ty) in best.items():
            tags[(op, lh.LAYOUT[fam], I, N, D, tx, ty, cap)].add("best")

    order = {"heur": 0, "traj": 1, "best": 2}
    for k in sorted(tags, key=lambda k: (lh.OPS.index(k[0]), k[1], k[2], k[3], k[7], k[5])):
        op, lay, I, N, D, tx, ty, cap = k
        t = "+".join(sorted(tags[k], key=order.get))
        print(t, op, lay, I, N, D, tx, ty, cap)


if __name__ == "__main__":
    main()
