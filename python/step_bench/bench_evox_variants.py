"""Cost of the EvoX baseline's implementation choices (paper, Section "Comparison with EvoX").

On every realistic cell (panmictic, compiled, median of per-seed medians):
  clamp      ga-gen with ``evox.utils.clamp`` instead of ``torch.clamp``
  workflow   de-gen and ga-gen run through ``StdWorkflow.step`` instead of the
             algorithm's ``step`` (what the timing runs measure)
``tournament_selection`` is measured by the sweep itself (``--ga-select evox``).

Usage (from python/, PYTHONPATH=.:<evox>/src):
  python -m step_bench.bench_evox_variants --sweep ../configs/sweep_realistic.json --out x.csv
Rows: test,op,N,D,seed,variant,per_gen_median_us
"""

import argparse
import csv
import gc
import json
import statistics as st

import torch

from evox.core import compile
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp
from evox.workflows import StdWorkflow

from . import bench_generation as bg
from .common import time_op


class GAEvoxClamp(bg.GA):
    """bg.GA with the ``evox.utils.clamp`` of a literal composition of EvoX utilities."""

    def step(self):
        mating_pool = self.select()
        crossovered = simulated_binary(self.pop[mating_pool])
        offspring = polynomial_mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        compare = off_fit < self.fit
        self.pop = torch.where(compare[:, None], offspring, self.pop)
        self.fit = torch.where(compare, off_fit, self.fit)


def make(op, n, d, dev, cls=bg.GA):
    if op == "de-gen":
        from evox.algorithms import DE
        return DE(pop_size=n, lb=torch.zeros(d, device=dev), ub=torch.ones(d, device=dev),
                  base_vector="rand", num_difference_vectors=1, differential_weight=0.5,
                  cross_probability=0.9, device=dev)
    return cls(n, torch.zeros(d, device=dev), torch.ones(d, device=dev), 1, "fast", dev)


def timed(build, gens, warmup):
    """Median generation time of ``build()``'s step; frees the population and the
    compiled graph afterwards (they would otherwise pile up across cells)."""
    try:
        obj = build()
        return f"{st.median(time_op(compile(obj.step), reps=gens, warmup=warmup)):.1f}"
    except torch.OutOfMemoryError:
        return "oom"
    finally:
        obj = None
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()


def algorithm(op, n, d, dev, seed, cls=bg.GA):
    torch.manual_seed(seed)
    algo = make(op, n, d, dev, cls)
    algo.evaluate = bg.Sphere().evaluate
    algo.init_step()
    return algo


def workflow(op, n, d, dev, seed):
    torch.manual_seed(seed)
    wf = StdWorkflow(make(op, n, d, dev), bg.Sphere())
    wf.init_step()
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()
    dev = "cuda"
    torch.set_default_device(dev)
    tuples = json.load(open(a.sweep))["tuples"]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "op", "N", "D", "seed", "variant", "per_gen_median_us"])
        for I, N, D in tuples:
            n = I * N
            for s in range(a.seeds):
                for variant, cls in (("torch.clamp", bg.GA), ("evox.utils.clamp", GAEvoxClamp)):
                    t = timed(lambda: algorithm("ga-gen", n, D, dev, s, cls), a.generations, a.warmup)
                    w.writerow(["clamp", "ga-gen", n, D, s, variant, t])
                for op in ("de-gen", "ga-gen"):
                    t = timed(lambda: algorithm(op, n, D, dev, s), a.generations, a.warmup)
                    w.writerow(["workflow", op, n, D, s, "algorithm.step", t])
                    t = timed(lambda: workflow(op, n, D, dev, s), a.generations, a.warmup)
                    w.writerow(["workflow", op, n, D, s, "StdWorkflow.step", t])
                f.flush()
            print(f"I={I} N={N} D={D} done", flush=True)


if __name__ == "__main__":
    main()
