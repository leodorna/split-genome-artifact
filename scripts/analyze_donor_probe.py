#!/usr/bin/env python3
"""Summarize scripts/donor_probe.sh: time per generation of unsplit de-gen and ga-gen
launches as built, with constant bounds, and (de-gen) with one donor row instead of three.
Median over seeds of the per-seed medians.

Usage: analyze_donor_probe.py [results/donor_probe/donor_probe.csv]
Writes donor_probe_summary.txt next to the CSV.
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "donor_probe", "donor_probe.csv")
per = defaultdict(lambda: defaultdict(list))
fails = 0
for r in csv.DictReader(open(path)):
    if r["rep"] == "fail":
        fails += 1
        continue
    k = (r["op"], r["layout"], int(r["I"]), int(r["N"]), int(r["D"]), int(r["tx"]), r["variant"])
    per[k][r["seed"]].append(float(r["time_us"]))
t = {k: st.median(st.median(v) for v in s.values()) / 1e3 for k, s in per.items()}   # ms
spread = [(max(m) - min(m)) / st.median(m) for s in per.values() for m in [[st.median(v) for v in s.values()]]]
out = []
P = out.append
P(f"{len(t)} timings, {fails} failed runs, seeds per timing {sorted({len(s) for s in per.values()})}, "
  f"seed spread median {st.median(spread):.3f} max {max(spread):.3f}")
cells = sorted({k[2:5] for k in t}, key=lambda c: (c[0], -c[2]))
txs = sorted({k[5] for k in t})
P("\n== ms per generation, unsplit (s = 1, t_y = 1): as built / constant bounds / one donor row")
for op in ("de-gen", "ga-gen"):
    for lay in ("highdim", "highdim_vec4"):
        for c in cells:
            row = []
            for tx in txs:
                v = [t.get((op, lay) + c + (tx, x)) for x in ("base", "constb", "onedonor")]
                row.append(f"tx={tx}: " + " / ".join("-" if x is None else f"{x:6.2f}" for x in v))
            P(f"  {op} {lay:13} I={c[0]:<5} N={c[1]:<5} D=2^{c[2].bit_length() - 1:<3} " + "   ".join(row))


def stats(name, sel):
    """ratios against the kernel as built, and the spread between block widths."""
    cb, od, sp_b, sp_o = [], [], [], []
    for k in t:
        if k[6] != "base" or not sel(k):
            continue
        b = t[k]
        if k[:6] + ("constb",) in t:
            cb.append(t[k[:6] + ("constb",)] / b)
        if k[:6] + ("onedonor",) in t:
            od.append(t[k[:6] + ("onedonor",)] / b)
    for key in {k[:5] for k in t if sel(k)}:
        for var, dst in (("base", sp_b), ("onedonor", sp_o)):
            v = [t[key + (tx, var)] for tx in txs if key + (tx, var) in t]
            if len(v) > 1:
                dst.append(max(v) / min(v))
    line = f"  {name:34} constant bounds x{min(cb):.2f}-{max(cb):.2f} (median {st.median(cb):.2f})"
    if od:
        line += (f" | one donor x{min(od):.2f}-{max(od):.2f} (median {st.median(od):.2f})"
                 f" | slower/faster t_x: as built median {st.median(sp_b):.2f} max {max(sp_b):.2f},"
                 f" one donor median {st.median(sp_o):.2f} max {max(sp_o):.2f}")
    P(line)


P("\n== time relative to the kernel as built")
for op in ("de-gen", "ga-gen"):
    for nm, sel in (("all cells", lambda k: True), ("I <= 32", lambda k: k[2] <= 32), ("I = 1024", lambda k: k[2] == 1024)):
        stats(f"{op} {nm}", lambda k, op=op, sel=sel: k[0] == op and sel(k))
    for lay in ("highdim", "highdim_vec4"):
        stats(f"{op} {lay}, I <= 32", lambda k, op=op, lay=lay: k[0] == op and k[1] == lay and k[2] <= 32)

P("\n== de-gen against ga-gen, I <= 32, widest block: ms (min-max over cells)")
tx = txs[-1]
for lay in ("highdim", "highdim_vec4"):
    def rng(op, var):
        v = [t[(op, lay) + c + (tx, var)] for c in cells if c[0] <= 32 and (op, lay) + c + (tx, var) in t]
        return f"{min(v):.1f}-{max(v):.1f}"
    P(f"  {lay:13} de-gen as built {rng('de-gen', 'base')} | de-gen one donor {rng('de-gen', 'onedonor')} | ga-gen as built {rng('ga-gen', 'base')}")
text = "\n".join(out)
print(text)
open(os.path.join(os.path.dirname(path), "donor_probe_summary.txt"), "w").write(text + "\n")
