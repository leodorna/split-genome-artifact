# Split-genome GPU kernels for evolutionary operators

CUDA C++ kernels for one fused generation of an island-based evolutionary algorithm
(selection, crossover, mutation, fitness and replacement in one launch), with the
*split-genome* layout of the paper *Split-Genome Layouts for Evolutionary
Algorithms on GPUs: Bottlenecks and Trade-offs at Million-Variable Scale*.

Each mating couple is processed by `t_x × t_y` threads and its genome is split over up to
`s` thread blocks, so the mapping of a genome to blocks is a launch parameter instead of a
design decision. Three operators are implemented, each in a scalar and a 128-bit
vectorized (`vec4`) variant:

| `--algo` | Algorithm | Fitness |
|---|---|---|
| `de-gen` | DE/rand/1/bin, 1:1 replacement | Sphere, Rastrigin, Rosenbrock, CEC 2013 f1–f3, f12 |
| `ga-gen` | binary tournament + SBX + polynomial mutation, 1:1 replacement | Sphere, Rastrigin, Rosenbrock, CEC 2013 f1–f3, f12 |
| `ux-bf`  | binary tournament + uniform crossover + bit-flip mutation on packed 32-bit words | OneMax, 0/1 knapsack |

The paper calls `de-gen` and `ga-gen` `de` and `sbx-pm`. The evaluation artifact of the
paper (raw data, figure scripts, EvoX baselines and sweep drivers) is published separately.

## Requirements

- An NVIDIA GPU and the CUDA Toolkit (the paper used CUDA 13.2 for `sm_120` and 12.9 for `sm_86`).
- NVIDIA MathDx 25.12, for its cuRANDDx header (counter-based Philox generator). It is not
  redistributed here: download it from NVIDIA and unpack it, for example into
  `external/mathdx/25.12/`, or point `MATHDX_DIR` to its `include/` directory.

## Build and test

    make CUDA_ARCH=120 MATHDX_DIR=/path/to/mathdx/25.12/include      # bin/bench_generation_v2
    make CUDA_ARCH=120 MATHDX_DIR=... test                             # RNG streams + fused vs decomposed
    make CUDA_ARCH=120 MATHDX_DIR=... roofline                         # bin/roofline_ceilings_v3

`CUDA_ARCH` is the SM architecture of the GPU (120 for the RTX PRO 6000 Blackwell, 86 for
the RTX A6000).

## Run

    ./bin/bench_generation_v2 --mode generation --algo ga-gen --layout highdim_vec4 \
        --islands 32 --n 256 --d 131072 --xthreads 512 --yblock 1 --hd-cap 1 \
        --reps 10 --warmup 5 --seed 42

prints a CSV line whose last field is the median time per generation in microseconds
(about 7140 on an RTX PRO 6000 Blackwell).

| Option | Meaning |
|---|---|
| `--islands I --n N --d D` | islands, individuals per island, genes per individual (`ux-bf`: bits, a multiple of 32) |
| `--layout` | `highdim` (scalar) or `highdim_vec4` (`D` a multiple of 4; 128 for `ux-bf`); `individual*` is the block-per-individual baseline |
| `--hd-cap s` | split factor; the kernel launches `min(s, ceil(U / t_x))` blocks per genome |
| `--xthreads t_x`, `--yblock t_y` | threads per couple, couples per block |
| `--fitness` | `sphere` (default), `rastrigin`, `rosenbrock`, `cec-f1`, `cec-f2`, `cec-f3`, `cec-f12`; `--lb/--ub` set the bounds |
| `--fitness knapsack`, `--kp-seed`, `--kp-type` | `ux-bf` only: 0/1 knapsack with D items (weights 1–1000, profits uncorrelated or `strong`ly correlated, capacity half the total weight), fitness −profit + linear penalty |
| `--shift-seed`, `--shift-file` | shift of `cec-f1..f3` and `cec-f12` (CEC 2013 LSGO f1, f2, f3 with their element-wise transformations, unrotated as in the suite, and f12, the shifted Rosenbrock): generated from a seed (default 1), or read as D float32 values or from the suite's `.txt` xopt file |
| `--check` | after every generation, recompute each fitness on the host and check that the best fitness never gets worse (elitism) |
| `--trace FILE` | write the best and mean fitness of every generation |

### Launch heuristic

Without tuning, the paper's heuristic sets, with `U` units per genome (`D`, `D/4`, `D/32`
or `D/128`), `C` couples per island (`N` for DE, `N/2` for the GAs) and `n_SM` SMs:

    s   = max(1, 2^ceil(log2(n_SM / (I * C))))
    t_x = min(512, 2^floor(log2(U)), max(32, a, b))
    a   = 2^floor(log2(U / 9))
    b   = 2^ceil(log2(512 * n_SM / (I * C * s)))
    t_y = 1

(Equation 14 of the paper.) On the paper's grid its median regret against exhaustive
tuning is 1.04 for `ga-gen` and `ux-bf`; DE's best launch depends on its memory traffic and
needs tuning on each device (see the paper).

## Layout

| Path | Content |
|---|---|
| `include/common.ipp` | counter-based random streams (Philox4x32 via cuRANDDx) |
| `include/operators.hpp`, `operators_aux.hpp` | selection, crossover, mutation, fitness and replacement policies |
| `include/kernels/fused_generation_v2.hpp` | the fused generation kernel and the deferred restore |
| `include/kernels/fused_v2.hpp`, `decomposed_v2.hpp` | fused variation, and the decomposed pipeline used as a test oracle |
| `include/generation_buffer.hpp` | device buffers |
| `apps/bench_generation_v2.cpp` | benchmark and correctness driver |
| `apps/roofline_ceilings_v3.cu` | measured DRAM and issue-rate roofs |
| `tests/` | RNG stream independence, fused vs decomposed pipeline |

## License

MIT (see `LICENSE`). MathDx is distributed by NVIDIA under its own license.
