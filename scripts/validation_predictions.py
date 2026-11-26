#!/usr/bin/env python3
"""Configurations the first and the revised launch heuristic choose on the validation
cells (configs/sweep_validation.json), written before those cells are measured."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_heuristic as lh  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SM = 188
tuples = json.load(open(os.path.join(HERE, "..", "configs", "sweep_validation.json")))["tuples"]
print("op,family,I,N,D,first_cap,first_tx,revised_cap,revised_tx")
for op in ("de-gen", "ga-gen", "ux-bf"):
    for fam in ("scalar", "vec4"):
        for I, N, D in tuples:
            D = D * 32 if op == "ux-bf" else D
            a = lh.launch_config(op, fam, I, N, D, SM)
            b = lh.launch_config(op, fam, I, N, D, SM, 1.0, 9, True)
            print(f"{op},{fam},{I},{N},{D},{a[0]},{a[1]},{b[0]},{b[1]}")
