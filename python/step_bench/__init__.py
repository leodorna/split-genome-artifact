"""step_bench: the EvoX-based baselines of the split-genome paper (bench_generation).

Public helpers:
    * :func:`time_op` / :func:`run_benchmark` -- the timing/CSV harness.
    * :func:`uniform_crossover` / :func:`bitflip_mutation` -- bit-packed binary
      operators (not shipped by EvoX).
"""

from .common import (
    CSV_HEADER,
    bitflip_mutation,
    random_int32,
    run_benchmark,
    time_op,
    uniform_crossover,
)

__all__ = [
    "CSV_HEADER",
    "time_op",
    "run_benchmark",
    "uniform_crossover",
    "bitflip_mutation",
    "random_int32",
]
