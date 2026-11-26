#!/usr/bin/env python3
"""Costs of the EvoX baseline's implementation choices, per cell (paper, "Same algorithm"
and "Same measurement"):
  tournament_selection   ga-gen sweep with --ga-select evox / default sweep (panmictic)
  evox.utils.clamp       bench_evox_variants.py: evox clamp / torch.clamp (ga-gen)
  StdWorkflow            bench_evox_variants.py: StdWorkflow.step / algorithm.step
All times are medians over seeds of the per-seed median.

Usage: analyze_evox_variants.py [--dir results/rerun_v3/evox]
"""
import argparse
import csv
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def sweep(path):
    per = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["mode"] == "compiled" and r["islands"] == "1":
            per[(int(r["N"]), int(r["D"]))].append(float(r["per_gen_median_us"]))
    return {k: st.median(v) for k, v in per.items()}


def show(name, ratios):
    v = list(ratios.values())
    print(f"{name}: {min(v):.3f}-{max(v):.3f} (median {st.median(v):.3f})")
    for (n, d), x in sorted(ratios.items(), key=lambda kv: kv[0][1]):
        print(f"    N={n:>8} D=2^{d.bit_length() - 1:<3} {x:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "..", "results", "rerun_v3", "evox"))
    a = ap.parse_args()
    fast = sweep(os.path.join(a.dir, "evox_gen_ga-gen.csv"))
    evsel = sweep(os.path.join(a.dir, "tournament_selection", "evox_gen_ga-gen.csv"))
    show("ga-gen tournament_selection / two-draw tournament", {k: evsel[k] / fast[k] for k in evsel if k in fast})

    per = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(a.dir, "evox_variants.csv"))):
        if r["per_gen_median_us"] == "oom":
            continue
        per[(r["test"], r["op"], int(r["N"]), int(r["D"]), r["variant"])].append(float(r["per_gen_median_us"]))
    t = {k: st.median(v) for k, v in per.items()}
    show("ga-gen evox.utils.clamp / torch.clamp",
         {(k[2], k[3]): t[k] / t[k[:4] + ("torch.clamp",)] for k in t if k[0] == "clamp" and k[4] == "evox.utils.clamp"})
    for op in ("de-gen", "ga-gen"):
        show(f"{op} StdWorkflow.step / algorithm.step",
             {(k[2], k[3]): t[k] / t[k[:4] + ("algorithm.step",)]
              for k in t if k[0] == "workflow" and k[1] == op and k[4] == "StdWorkflow.step"})


if __name__ == "__main__":
    main()
