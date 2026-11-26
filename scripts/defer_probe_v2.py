#!/usr/bin/env python3
"""Probe: does the in-kernel parent restore explain why split rows hurt DE?

For the low-row DE cells (I in {1, 32}) and, as a control, the I=1 ga-gen cells,
sweeps cap at the heuristic's block shape (tx=512, ty=1) plus the cell's tuned best
config, with the default restore (last-arriving block copies a losing parent row
back; deferred only when rows <= 64) and with --defer-always (separate parallel
restore kernel). Same seeds for both.

Output: results/phase3_defer_probe_v2.csv
  op,layout,I,N,D,cap,tx,ty,defer,seed,rep,time_us
"""
import argparse
import csv
import os
import statistics as st
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BIN = os.path.join(ROOT, "bin", "bench_generation_v2")
RES = os.path.join(HERE, "..", "results")
CELLS = [(1, 64, 1 << 24), (1, 256, 1 << 22), (1, 1024, 1 << 20),
         (32, 64, 1 << 19), (32, 256, 1 << 17), (32, 1024, 1 << 15)]
CAPS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def tuned_best(path):
    """(op, layout, I, N, D) -> best (cap, tx, ty) from the augmented sweep."""
    per = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["status"] == "ok":
            per[(r["op"], r["layout"], int(r["I"]), int(r["N"]), int(r["D"]))][
                (int(r["cap"]), int(r["tx"]), int(r["ty"]))].append(float(r["time_us"]))
    return {k: min(v, key=lambda c: st.median(v[c])) for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=os.path.join(RES, "rerun_v3", "phase3_unified_final_v3.csv"))
    ap.add_argument("--out", default=os.path.join(RES, "rerun_v3", "defer_probe_v3.csv"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    best = tuned_best(a.final)
    jobs = []
    for op, cells in (("de-gen", CELLS), ("ga-gen", CELLS[:3])):
        for layout in ("highdim", "highdim_vec4"):
            for (I, N, D) in cells:
                cfgs = [(c, 512, 1) for c in CAPS]
                b = best.get((op, layout, I, N, D))
                if b and b not in cfgs:
                    cfgs.append(b)
                jobs += [(op, layout, I, N, D, c) for c in cfgs]

    tmp = tempfile.NamedTemporaryFile(delete=False).name
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "layout", "I", "N", "D", "cap", "tx", "ty", "defer", "seed", "rep", "time_us"])
        for n, (op, layout, I, N, D, (cap, tx, ty)) in enumerate(jobs):
            for defer in (0, 1):
                for sd in range(a.seeds):
                    seed = 42 + sd * 1000003
                    cmd = [BIN, "--mode", "generation", "--algo", op, "--layout", layout,
                           "--islands", str(I), "--n", str(N), "--d", str(D),
                           "--xthreads", str(tx), "--yblock", str(ty), "--hd-cap", str(cap),
                           "--reps", str(a.reps), "--warmup", str(a.warmup), "--seed", str(seed),
                           "--dump-times", tmp] + (["--defer-always"] if defer else [])
                    open(tmp, "w").close()
                    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        timeout=300).returncode
                    times = [l.strip() for l in open(tmp) if l.strip()] if rc == 0 else []
                    for rep, us in enumerate(times):
                        w.writerow([op, layout, I, N, D, cap, tx, ty, defer, sd, rep, us])
            f.flush()
            print(f"[{n + 1}/{len(jobs)}] {op} {layout} I={I} N={N} D={D} cfg=({cap},{tx},{ty})",
                  flush=True, file=sys.stderr)
    os.unlink(tmp)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
