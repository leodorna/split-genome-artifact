"""0/1 knapsack for the binary GA baseline (ux-bf), the same instance as bench_generation_v2.

Item j: weight w_j = 1 + (h mod R), profit p_j = 1 + ((h >> 32) mod R) (uncorrelated) or
w_j + R/10 (strongly correlated), h = splitmix64(seed * 2^32 + j), R = 1000; capacity
c = floor(sum w / 2); penalty factor rho = max_j p_j / w_j.
Value (maximized by the baseline, whose OneMax convention is "higher is better"):
    P - rho max(0, W - c),  P = sum p_j x_j,  W = sum w_j x_j,
the negative of the kernel's minimized fitness. Bit b of int32 word u selects item 32 u + b.
"""
import numpy as np
import torch

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
R = 1000


def make_knapsack(n: int, seed: int = 1, strong: bool = True, chunk: int = 1 << 24):
    """(weights, profits, capacity, rho) as in the kernel's make_knapsack.

    Weights and profits are int32 (at most R + R/10); generated in chunks so that the
    largest instance (2^29 items) needs about 4 GB of host memory, not several times that.
    """
    w = np.empty(n, dtype=np.int32)
    p = np.empty(n, dtype=np.int32)
    total, rho = 0, 0.0
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        with np.errstate(over="ignore"):
            z = (np.uint64(seed) << np.uint64(32)) + np.arange(lo, hi, dtype=np.uint64) + _GAMMA
            z = (z ^ (z >> np.uint64(30))) * _M1
            z = (z ^ (z >> np.uint64(27))) * _M2
            z = z ^ (z >> np.uint64(31))
        wc = (np.uint64(1) + z % np.uint64(R)).astype(np.int32)
        if strong:
            pc = wc + np.int32(R // 10)
        else:
            pc = (np.uint64(1) + (z >> np.uint64(32)) % np.uint64(R)).astype(np.int32)
        w[lo:hi], p[lo:hi] = wc, pc
        total += int(wc.sum(dtype=np.int64))
        rho = max(rho, float(np.max(pc / wc)))
    capacity = float(total // 2)
    return w, p, capacity, rho


class KnapsackValue:
    """Value of packed int32 genomes (rows) on one instance, in float32 like the kernel."""

    def __init__(self, n_items: int, device, seed: int = 1, strong: bool = True):
        w, p, self.capacity, self.rho = make_knapsack(n_items, seed, strong)
        words = n_items // 32
        self.w = torch.as_tensor(w, dtype=torch.float32, device=device).view(words, 32)
        self.p = torch.as_tensor(p, dtype=torch.float32, device=device).view(words, 32)
        self.shifts = torch.arange(32, dtype=torch.int32, device=device)

    def _value(self, pop: torch.Tensor) -> torch.Tensor:
        bits = ((pop.unsqueeze(-1) >> self.shifts) & 1).to(torch.float32)   # (rows, words, 32)
        P = (bits * self.p).sum((-2, -1))
        W = (bits * self.w).sum((-2, -1))
        return P - self.rho * torch.clamp(W - self.capacity, min=0.0)

    def __call__(self, pop: torch.Tensor) -> torch.Tensor:
        # Compiled (the timed step), Inductor fuses the unpacking into the reductions.
        # Eager (the initial population, the uncompiled reference), the (rows, words, 32)
        # temporary is materialized, so evaluate a few rows at a time (<= 2 GiB each).
        if torch.compiler.is_compiling():
            return self._value(pop)
        per_row = pop.shape[-1] * 32 * 4
        step = max(1, (2 << 30) // per_row)
        return torch.cat([self._value(pop[i:i + step]) for i in range(0, pop.shape[0], step)])
