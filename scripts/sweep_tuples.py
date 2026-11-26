"""Print the ablation sweep tuples as "I N D" lines.

Reads a sweep JSON (default configs/sweep_realistic.json) and optionally divides D by
--divide, so the same script covers the full (I*N*D=2^32) and half/quarter-memory
grids (e.g. --divide 2 -> max D = 2^29).
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--divide", type=int, default=1)
a = ap.parse_args()

path = a.config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "sweep_realistic.json")
data = json.load(open(path))
for I, N, D in data["tuples"]:
    print(I, N, max(1, D // a.divide))
