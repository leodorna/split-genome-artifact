"""Benchmark whole EvoX generations against the fused kernels (``bench_generation``).

Three generations are available, one per invocation of an EvoX step:

* ``de-gen`` -- ``evox.algorithms.DE`` (base_vector="rand", K=1, F=0.5, CR=0.9)
  with a Sphere problem: rand/1/1 donors (with replacement) + binomial
  crossover + greedy survival.
* ``ga-gen`` -- ``GA``, a single-objective GA written as an EvoX ``Algorithm``
  from EvoX operators only (``tournament_selection`` + ``simulated_binary`` +
  ``polynomial_mutation``) with the local "better of (parent, child)" survival
  used by the fused kernel.
* ``ux-bf`` -- a binary GA (UX + bitflip + OneMax) built from the bit-packed
  helpers (EvoX ships no binary operators), same local survival.

EvoX has no island decomposition.  With ``--islands 1`` (default) ``N =
pop_size`` is one panmictic population, as before.  With ``--islands I`` the
population holds ``I * N`` rows and every random parent/donor index is drawn
island-locally (a local index in ``[0, N)`` mapped to its island's rows), which
is the selection the fused kernel performs.  Everything else is the unchanged
EvoX code applied to the flat ``(I*N, D)`` tensor.  For SBX/UX, which pair row
``j`` with row ``j + n/2``, rows are stored "paired": island ``i`` owns rows
``[i*N/2, (i+1)*N/2)`` and ``[n/2 + i*N/2, n/2 + (i+1)*N/2)``, so every couple
stays inside one island.  DE stores island ``i`` contiguously at ``[i*N, (i+1)*N)``.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from .common import _popcount, bitflip_mutation, random_int32, time_op

from evox.core import Algorithm, Mutable, Problem, compile
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection
from evox.utils import clamp


class Sphere(Problem):
    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        return (pop**2).sum(-1)


class RosenbrockPad(Problem):
    """Rosenbrock with x_{j+1} read through a padded shift instead of EvoX's slices.

    Same values as EvoX's ``Rosenbrock``; under ``torch.compile`` it lets the
    replacement fuse into the evaluation (the offspring are still materialized).
    A sensitivity check of the baseline, not used for the main comparison.
    """

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        xn = torch.nn.functional.pad(x[..., 1:], (0, 1), value=1.0)
        t = 100 * (xn - x * x) ** 2 + (x - 1) ** 2
        keep = torch.arange(x.shape[-1], device=x.device) < x.shape[-1] - 1
        return torch.where(keep, t, 0.0).sum(-1)


# Gene bounds per problem, the same as the fused kernel's --fitness defaults.
BOUNDS = {"sphere": (0.0, 1.0), "rastrigin": (-5.12, 5.12), "rosenbrock": (-2.048, 2.048),
          "rosenbrock-pad": (-2.048, 2.048), "cec-f1": (-100.0, 100.0), "cec-f2": (-5.0, 5.0),
          "cec-f3": (-32.0, 32.0), "cec-f12": (-100.0, 100.0),
          "knapsack": (0.0, 1.0)}   # knapsack: ux-bf only (binary); the bounds are unused

# Knapsack instance of ux-bf (bench_generation_v2 --kp-seed / --kp-type).
KP_SEED = 1
KP_TYPE = "strong"

# Shift of the CEC functions: generated from SHIFT_SEED (bench_generation_v2 --shift-seed) unless
# SHIFT_FILE names a shift file (the suite's .txt xopt file or raw float32; --shift-file).
SHIFT_SEED = 1
SHIFT_FILE = None


def make_problem(name: str, d: int = 0, device=None) -> Problem:
    """Sphere as above; Rastrigin and Rosenbrock are EvoX's own implementations; CEC 2013
    LSGO f1-f3 and f12 are in cec.py."""
    if name == "sphere":
        return Sphere()
    if name.startswith("cec-"):
        from .cec import CEC, load_shift, make_shift
        cls, lo, hi = CEC[name]
        o = load_shift(SHIFT_FILE, d) if SHIFT_FILE else make_shift(d, lo, hi, SHIFT_SEED)
        return cls(d, torch.from_numpy(o).to(device))
    if name == "rosenbrock-pad":
        return RosenbrockPad()
    from evox.problems.numerical import Rastrigin, Rosenbrock
    return {"rastrigin": Rastrigin, "rosenbrock": Rosenbrock}[name]()


def _oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


# ---------------------------------------------------------------------------
# Island-local index maps (see module docstring).  ``None`` = panmictic.
# ---------------------------------------------------------------------------

def island_base_contiguous(n: int, islands: int, device):
    """Per-slot first row of the slot's island, contiguous layout (DE)."""
    if islands == 1:
        return None
    n_isl = n // islands
    return (torch.arange(n, device=device) // n_isl) * n_isl


def island_base_paired(n: int, islands: int, device):
    """Per-slot island offset ``island * N/2`` for the paired layout (SBX/UX)."""
    if islands == 1:
        return None
    half, h = n // 2, (n // islands) // 2
    return ((torch.arange(n, device=device) % half) // h) * h


def draw_parents(n: int, n_isl: int, base, paired: bool, device):
    """``n`` random row indices, island-local when ``base`` is given."""
    if base is None:
        return torch.randint(0, n, (n,), device=device)
    k = torch.randint(0, n_isl, (n,), device=device)
    if not paired:
        return base + k
    h = n_isl // 2
    return torch.where(k < h, base + k, base + k + (n // 2 - h))


# ---------------------------------------------------------------------------
# DE generation
# ---------------------------------------------------------------------------

def build_de(n: int, d: int, device: str, problem: str = "sphere"):
    from evox.algorithms import DE

    lo, hi = BOUNDS[problem]
    algo = DE(
        pop_size=n,
        lb=torch.full((d,), lo, device=device),
        ub=torch.full((d,), hi, device=device),
        base_vector="rand",
        num_difference_vectors=1,
        differential_weight=0.5,
        cross_probability=0.9,
        device=device,
    )
    algo.evaluate = make_problem(problem, d, device).evaluate
    algo.init_step()
    return algo.step


def build_de_islands(n: int, d: int, islands: int, device: str, problem: str = "sphere"):
    """``evox.algorithms.DE.step`` (rand/1/bin) with island-local donors.

    A line-by-line copy of EvoX's ``DE.step`` for ``base_vector="rand"``,
    ``num_difference_vectors=1``; only the ``torch.randint(0, pop_size)`` donor
    draws are replaced by island-local ones.
    """
    n_isl = n // islands
    base = island_base_contiguous(n, islands, device)
    lo, hi = BOUNDS[problem]
    lb = torch.full((1, d), lo, device=device)
    ub = torch.full((1, d), hi, device=device)
    F, CR = 0.5, 0.9
    prob = make_problem(problem, d, device)
    pop = torch.rand(n, d, device=device) * (ub - lb) + lb
    fit = prob.evaluate(pop)

    def step():
        nonlocal pop, fit
        rc = [draw_parents(n, n_isl, base, False, device) for _ in range(3)]
        base_vector = pop[rc[0]]
        difference_vector = torch.stack([pop[rc[1]] - pop[rc[2]]]).sum(dim=0)
        new_pop = base_vector + F * difference_vector
        cross_prob = torch.rand(n, d, device=device)
        random_dim = torch.randint(0, d, (n, 1), device=device)
        mask = cross_prob < CR
        mask = mask.scatter(dim=1, index=random_dim, value=1)
        new_pop = torch.where(mask, new_pop, pop)
        new_pop = clamp(new_pop, lb, ub)
        new_fitness = prob.evaluate(new_pop)
        compare = new_fitness < fit
        pop = torch.where(compare[:, None], new_pop, pop)
        fit = torch.where(compare, new_fitness, fit)

    step.fitness = lambda: fit
    return step


# ---------------------------------------------------------------------------
# GA generation: an EvoX ``Algorithm`` built from EvoX operators only
# ---------------------------------------------------------------------------

class GA(Algorithm):
    """Single-objective GA written as an EvoX ``Algorithm`` (EvoX ships none).

    ``step`` uses EvoX operators only -- ``tournament_selection`` ->
    ``simulated_binary`` -> ``polynomial_mutation`` -- then ``torch.clamp`` and ends with
    the better-of (parent, child) survival of the fused kernel, the same two
    ``torch.where`` lines that end EvoX's ``DE.step``.  With ``islands > 1`` the
    two tournament entrants are drawn island-locally (paired layout, see the
    module docstring) and the winner is picked as in ``tournament_selection``.

    ``select="fast"`` (default, panmictic only) draws the two entrants as two
    index vectors and keeps the better with ``torch.where`` -- the same
    distribution as ``tournament_selection``, but torch.compile fuses the parent
    gather into SBX; with ``tournament_selection`` (``select="evox"``) the
    compiled generation is ~25% slower at N=2^20, D=2^10.

    ``torch.clamp`` instead of ``evox.utils.clamp`` (a relu-based TorchScript
    workaround whose docstring recommends ``torch.clamp``): identical result on
    [0, 1] bounds, and ~15% faster here at N=64, D=2^24 (no effect on DE).
    """

    def __init__(self, pop_size: int, lb: torch.Tensor, ub: torch.Tensor, islands: int = 1,
                 select: str = "fast", device: torch.device | None = None):
        super().__init__()
        assert select in ("fast", "evox")
        self.evox_select = select == "evox"
        device = torch.get_default_device() if device is None else device
        self.pop_size = pop_size
        self.n_isl = pop_size // islands
        self.base = island_base_paired(pop_size, islands, device)
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)
        pop = torch.rand(pop_size, lb.shape[0], device=device) * (self.ub - self.lb) + self.lb
        self.pop = Mutable(pop)
        self.fit = Mutable(torch.empty(pop_size, device=device).fill_(float("inf")))

    def init_step(self):
        self.fit = self.evaluate(self.pop)

    def select(self) -> torch.Tensor:
        n, dev = self.pop_size, self.pop.device
        if self.base is None:
            if self.evox_select:
                return tournament_selection(n, self.fit)
            a = torch.randint(0, n, (n,), device=dev)
            b = torch.randint(0, n, (n,), device=dev)
            return torch.where(self.fit[a] <= self.fit[b], a, b)
        parents = torch.stack([draw_parents(n, self.n_isl, self.base, True, dev) for _ in range(2)], dim=1)
        winner = torch.argmin(self.fit[parents], dim=1)
        return torch.gather(parents, 1, winner.unsqueeze(1)).squeeze()

    def step(self):
        mating_pool = self.select()
        crossovered = simulated_binary(self.pop[mating_pool])
        offspring = polynomial_mutation(crossovered, self.lb, self.ub)
        offspring = torch.clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        compare = off_fit < self.fit
        self.pop = torch.where(compare[:, None], offspring, self.pop)
        self.fit = torch.where(compare, off_fit, self.fit)


def build_ga(n: int, d: int, device: str, islands: int = 1, select: str = "fast", problem: str = "sphere"):
    lo, hi = BOUNDS[problem]
    algo = GA(n, torch.full((d,), lo, device=device), torch.full((d,), hi, device=device), islands, select, device)
    algo.evaluate = make_problem(problem, d, device).evaluate
    algo.init_step()
    return algo.step


# ---------------------------------------------------------------------------
# Binary GA generation (UX + bitflip + OneMax) -- "ux-bf"
# ---------------------------------------------------------------------------

def onemax_value(pop):
    return _popcount(pop).sum(-1).to(torch.float32)


def uxbf_ga_step(pop, fit, pc, pm, n_isl=None, base=None, value=onemax_value):
    """One binary GA generation: tournament + UX + bitflip + local survival (value maximized)."""
    n = pop.shape[0]
    half = n // 2
    a = draw_parents(n, n_isl, base, True, pop.device)
    b = draw_parents(n, n_isl, base, True, pop.device)
    parents = pop[torch.where(fit[a] >= fit[b], a, b)]   # maximize the value

    p1, p2 = parents[:half], parents[half:]
    active = torch.rand(half, device=pop.device) < pc
    mask = random_int32(half, pop.shape[1], device=pop.device)
    cA = (~mask & p1) | (mask & p2)
    cB = (mask & p1) | (~mask & p2)
    cA = torch.where(active[:, None], cA, p1)
    cB = torch.where(active[:, None], cB, p2)

    children = torch.empty_like(parents)
    children[:half] = cA
    children[half:] = cB
    children = bitflip_mutation(children, pm)
    child_fit = value(children)

    better = child_fit > fit
    return torch.where(better[:, None], children, pop), torch.where(better, child_fit, fit)


def build_uxbf(n: int, d: int, device: str, pc: float, pm: float, islands: int = 1, problem: str = "sphere"):
    n_words = d // 32
    if problem == "knapsack":
        from .knapsack import KnapsackValue
        value = KnapsackValue(d, device, KP_SEED, KP_TYPE == "strong")
    else:
        value = onemax_value
    pop = random_int32(n, n_words, device=device)
    fit = value(pop)
    n_isl, base = n // islands, island_base_paired(n, islands, device)

    def step():
        nonlocal pop, fit
        pop, fit = uxbf_ga_step(pop, fit, pc, pm, n_isl, base, value)

    step.fitness = lambda: fit
    return step


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--op", default="de-gen",
                        help="de-gen | ga-gen | ux-bf  (aliases: uxbf, uxbf-gen, ux-bf)")
    parser.add_argument("--n", type=int, default=8192,
                        help="total population rows (islands * per-island N)")
    parser.add_argument("--islands", type=int, default=1,
                        help="island-local parent/donor draws; --n must be divisible, N/islands even")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ga-select", choices=["fast", "evox"], default="fast",
                        help="ga-gen panmictic tournament: two-draw (fast) or evox tournament_selection")
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--problem", choices=list(BOUNDS), default="sphere",
                        help="fitness of de-gen and ga-gen (Rastrigin and Rosenbrock are EvoX's; cec-f1..f3 and cec-f12 are CEC 2013 LSGO functions of cec.py; "
                             "rosenbrock-pad is Rosenbrock written as a padded shift)")
    parser.add_argument("--shift-seed", type=int, default=1, help="cec-f*: seed of the generated shift")
    parser.add_argument("--shift-file", type=str, default=None,
                        help="cec-f*: shift file (the suite's .txt xopt file or raw float32)")
    parser.add_argument("--kp-seed", type=int, default=1, help="ux-bf knapsack: instance seed")
    parser.add_argument("--kp-type", choices=["strong", "uncorrelated"], default="strong",
                        help="ux-bf knapsack: strongly correlated or uncorrelated profits")
    parser.add_argument("--pc", type=float, default=1.0, help="ux-bf UX couple gate")
    parser.add_argument("--pm", type=float, default=-1.0, help="ux-bf per-bit mutation; <0 uses max(1/D, 0.01)")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--compiled-only", action="store_true", help="skip the eager timing")
    parser.add_argument("--compile-mode", default=None,
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="torch.compile mode (default: torch.compile's default, as evox.core.compile); "
                             "the timing rows are labelled compiled-<mode>")
    parser.add_argument("--dump-times", type=str, default=None,
                        help="write the time of every timed generation (us), one per line, of the last mode timed")
    parser.add_argument("--trace", type=str, default=None,
                        help="instead of timing, run --generations compiled generations and write "
                             "gen,best,mean of the fitness (minimized; ux-bf as D - ones); gen 0 = initial")
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    torch.manual_seed(args.seed)
    I = args.islands
    if I < 1 or args.n % I or (args.n // I) % 2:
        parser.error("--n must be a multiple of --islands with an even per-island N")

    if args.op in ("ux-bf", "uxbf", "uxbf-gen", "ux-bf"):
        op = "ux-bf"
    elif args.op == "de-gen":
        op = "de-gen"
    elif args.op == "ga-gen":
        op = "ga-gen"
    else:
        parser.error(f"unknown op: {args.op}")

    global SHIFT_SEED, SHIFT_FILE, KP_SEED, KP_TYPE
    SHIFT_SEED, SHIFT_FILE = args.shift_seed, args.shift_file
    KP_SEED, KP_TYPE = args.kp_seed, args.kp_type
    if (op == "ux-bf" and args.problem not in ("sphere", "knapsack")) or (op != "ux-bf" and args.problem == "knapsack"):
        parser.error("ux-bf evaluates OneMax (--problem sphere, the default) or knapsack; knapsack is ux-bf only")
    if op == "de-gen":
        step = (build_de(args.n, args.d, device, args.problem) if I == 1
                else build_de_islands(args.n, args.d, I, device, args.problem))
    elif op == "ga-gen":
        step = build_ga(args.n, args.d, device, I, args.ga_select, args.problem)
    else:
        pm = max(1.0 / args.d, 0.01) if args.pm < 0 else args.pm
        step = build_uxbf(args.n, args.d, device, args.pc, pm, I, args.problem)

    if args.trace:
        # closures expose .fitness(); EvoX Algorithms keep it in .fit
        fitness = step.fitness if hasattr(step, "fitness") else (lambda: step.__self__.fit)
        to_min = ((lambda f: -f) if args.problem == "knapsack" else (lambda f: args.d - f)) if op == "ux-bf" \
            else (lambda f: f)
        compiled = compile(step)
        with open(args.trace, "w") as tf:
            tf.write("gen,best,mean\n")
            for g in range(args.generations + 1):
                if g:
                    compiled()
                f = to_min(fitness().double())
                tf.write(f"{g},{f.min().item():.9g},{f.mean().item():.9g}\n")
        return

    # correctness/behaviour sanity, then time eager and compiled.  Both are made
    # OOM-resilient: the largest sweep points (small D, huge total) can exceed the
    # 97 GB card for the float generations, and we want an "oom" row, not a crash.
    sink = open(args.out, "w") if args.out else None

    def emit(line):
        (sink.write(line + "\n") if sink is not None else print(line, flush=True))

    emit("op,N,D,mode,per_gen_mean_us,per_gen_min_us,islands,seed,per_gen_median_us")
    tag = f"{I},{args.seed}"

    def timed(fn, mode):
        try:
            for _ in range(3):
                fn()
            times = time_op(fn, reps=args.generations, warmup=args.warmup, device=device)
        except RuntimeError as e:
            if _oom(e):
                emit(f"{op},{args.n},{args.d},{mode},oom,oom,{tag},oom")
                torch.cuda.empty_cache()
                return
            raise
        med = statistics.median(times)
        if args.dump_times:
            with open(args.dump_times, "w") as tf:
                tf.write("".join(f"{t:.3f}\n" for t in times))
        emit(f"{op},{args.n},{args.d},{mode},{sum(times)/len(times):.3f},{min(times):.3f},{tag},{med:.3f}")

    if not args.compiled_only:
        timed(step, "eager")
    if not args.eager:
        cmode = args.compile_mode if args.compile_mode not in (None, "default") else None
        label = "compiled" if cmode is None else f"compiled-{cmode}"
        try:
            compiled = compile(step) if cmode is None else compile(step, mode=cmode)
            if cmode in ("reduce-overhead", "max-autotune"):
                # CUDA graphs: the state a step reassigns is an output of the graph that the
                # next replay overwrites; clone the user-visible tensors that persist across
                # generations and mark each step (as in the MOEA/D comparison's driver)
                torch._inductor.config.triton.cudagraph_trees_generation_cloning = "user_visible"
                graphed = compiled

                def compiled(graphed=graphed):
                    torch.compiler.cudagraph_mark_step_begin()
                    return graphed()
        except RuntimeError as e:
            if _oom(e):
                emit(f"{op},{args.n},{args.d},{label},oom,oom,{tag},oom")
                compiled = None
            else:
                raise
        if compiled is not None:
            timed(compiled, label)

    if sink is not None:
        sink.close()


if __name__ == "__main__":
    main()
