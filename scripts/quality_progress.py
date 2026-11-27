#!/usr/bin/env python3
"""Search progress in the fitness traces, offset-free: the share of the improvement of the
median best fitness since generation 0 (median over seeds, per generation) that the last
100 and the last 500 generations contribute, and the share reached by generations 50, 100
and 200. Raw relative changes depend on the function's offset (Ackley near 21, the
knapsack near -1.2e6), this share does not. Section 4.2 of the paper quotes the ranges
over the panels of Figure 6 (configurations evox and s1) and the large-D cell.

Usage: quality_progress.py [--root results]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_quality_paper import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    a = ap.parse_args()
    dirs = {"Sphere/OneMax": "rerun_v3/quality"}
    for f in ("cec-f1", "cec-f2", "cec-f3", "cec-f12", "knapsack"):
        dirs[f] = f"transfer/{f}/quality"
    dirs["large D (I=1, N=64, D=2^20)"] = "largeD/quality"
    fig6 = {"last100": [], "last500": []}
    for name, d in dirs.items():
        T = load(os.path.join(a.root, d))
        for op in sorted({k[0] for k in T}):
            for cfg in ("evox", "s1", "heur"):
                Y = np.array([v for (o, c, _), v in sorted(T.items()) if o == op and c == cfg])
                if not len(Y):
                    continue
                m = np.median(Y, 0)
                tot = m[0] - m[-1]
                l100, l500 = (m[-101] - m[-1]) / tot, (m[-501] - m[-1]) / tot
                by = {g: (m[0] - m[g]) / tot for g in (50, 100, 200)}
                if not name.startswith("large") and cfg != "heur":
                    fig6["last100"].append(l100)
                    fig6["last500"].append(l500)
                print(f"{name:28s} {op:7s} {cfg:5s} last 100: {100 * l100:5.2f}%  last 500: {100 * l500:5.2f}%  "
                      + "  ".join(f"by gen {g}: {100 * v:5.1f}%" for g, v in by.items()))
    print(f"Figure 7 panels: the last 100 generations contribute {100 * min(fig6['last100']):.2f}-"
          f"{100 * max(fig6['last100']):.2f}% of the improvement, the last 500 {100 * min(fig6['last500']):.2f}-"
          f"{100 * max(fig6['last500']):.2f}%")


if __name__ == "__main__":
    main()
