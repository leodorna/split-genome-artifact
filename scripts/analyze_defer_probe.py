#!/usr/bin/env python3
"""Summarize defer_probe_v2.py output: per cell and split factor s (tx=512, ty=1),
the time relative to the cell's fastest probed config, default restore / deferred
restore (median over seeds of per-seed medians), plus the cell's tuned config.

Usage: analyze_defer_probe.py results/rerun_v3/defer_probe_v3.csv
"""
import csv
import statistics as st
import sys
from collections import defaultdict

per = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(sys.argv[1])):
    k = (r["op"], r["layout"], int(r["I"]), int(r["N"]), int(r["D"]),
         int(r["cap"]), int(r["tx"]), int(r["ty"]), int(r["defer"]))
    per[k][r["seed"]].append(float(r["time_us"]))
t = {k: st.median([st.median(v) for v in s.values()]) for k, s in per.items()}
cells = sorted({k[:5] for k in t}, key=lambda c: (c[0], c[1], c[2], -c[4]))
for c in cells:
    best = min(v for k, v in t.items() if k[:5] == c)
    line = []
    for cap in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        a, b = t.get(c + (cap, 512, 1, 0)), t.get(c + (cap, 512, 1, 1))
        if a and b:
            line.append(f"{cap}:{a / best:.2f}/{b / best:.2f}")
    tuned = [f"tuned{k[5:8]}:{t[k] / best:.2f}/{t[k[:8] + (1,)] / best:.2f}"
             for k in t if k[:5] == c and k[8] == 0 and not (k[6] == 512 and k[7] == 1)]
    print(f"{c[0]} {c[1]:13} I={c[2]:<3} N={c[3]:<5} D=2^{c[4].bit_length() - 1}: "
          + " ".join(line) + "  " + " ".join(tuned))
