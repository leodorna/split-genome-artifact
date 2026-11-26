"""CEC 2013 LSGO f1, f2, f3 and f12 for the EvoX-based baselines.

Definitions as in the organizers' C++ code (Benchmarks.cpp of the CEC 2013 LSGO suite), with
z = x - o and t_j = j / (D - 1), j = 0..D-1:
  T_osz(z)      = sign(z) exp(h + 0.049 (sin(c1 h) + sin(c2 h))), h = log|z| (0 if z = 0),
                  (c1, c2) = (10, 7.9) if z > 0 else (5.5, 3.1)
  T_asy^0.2(z)  = z^(1 + 0.2 t sqrt(z)) if z > 0 else z
  Lambda^10(z)  = 10^(t / 2) z
  f1 = sum 10^(6 t) T_osz(z)^2                                                  [-100, 100]^D
  f2 = sum (y^2 - 10 cos(2 pi y) + 10),  y = Lambda^10 T_asy^0.2 T_osz(z)        [-5, 5]^D
  f3 = -20 exp(-0.2 sqrt(mean y^2)) - exp(mean cos(2 pi y)) + 20 + e, y as f2   [-32, 32]^D
  f12 = sum_{j<D-1} 100 (z_j^2 - z_{j+1})^2 + (z_j - 1)^2   (no transformation)    [-100, 100]^D
The suite defines them for D = 1000 with fixed shift files (F1-xopt.txt, ...); for other D the
shift o is generated exactly as in bench_generation_v2 (make_shift):
o_j = lo + (hi - lo) (0.1 + 0.8 u_j), u_j = top 24 bits of splitmix64(seed * 2^32 + j) / 2^24.

    python -m step_bench.cec --d 1024 --lo -100 --hi 100 --seed 1 --out shift.f32
writes a generated shift as D float32 values (the kernel's --shift-file format).
"""
import argparse
import math

import numpy as np
import torch
from evox.core import Problem

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def make_shift(d: int, lo: float, hi: float, seed: int = 1) -> np.ndarray:
    """The shift vector of the fused kernel's make_shift, as float32."""
    with np.errstate(over="ignore"):
        z = (np.uint64(seed) << np.uint64(32)) + np.arange(d, dtype=np.uint64) + _GAMMA
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(40)).astype(np.float64) / 16777216.0
    return (lo + (hi - lo) * (0.1 + 0.8 * u)).astype(np.float32)


def load_shift(path: str, d: int) -> np.ndarray:
    """A shift file: the suite's .txt xopt file (comma/newline separated) or raw float32."""
    if path.endswith(".txt"):
        v = np.array([float(t) for t in open(path).read().replace(",", " ").split()], dtype=np.float64)
    else:
        v = np.fromfile(path, dtype=np.float32)
    if v.size < d:
        raise ValueError(f"{path}: {v.size} values, need {d}")
    return v[:d].astype(np.float32)


def _t(d: int, device) -> torch.Tensor:
    return torch.arange(d, dtype=torch.float32, device=device) / max(d - 1, 1)


def t_osz(z: torch.Tensor) -> torch.Tensor:
    h = torch.where(z == 0, torch.zeros_like(z), torch.log(torch.abs(z)))
    pos = z > 0
    c1 = torch.where(pos, 10.0, 5.5)
    c2 = torch.where(pos, 7.9, 3.1)
    return torch.sign(z) * torch.exp(h + 0.049 * (torch.sin(c1 * h) + torch.sin(c2 * h)))


def t_asy_lambda(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Lambda^10 T_asy^0.2 (y)."""
    y = torch.where(y > 0, torch.pow(torch.clamp(y, min=0.0), 1.0 + 0.2 * t * torch.sqrt(torch.clamp(y, min=0.0))), y)
    return y * torch.pow(10.0, 0.5 * t)


class CecF1(Problem):
    def __init__(self, d: int, shift: torch.Tensor):
        super().__init__()
        self.shift = shift
        self.weight = torch.pow(10.0, 6.0 * _t(d, shift.device))

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        y = t_osz(x - self.shift)
        return (self.weight * y * y).sum(-1)


class CecF2(Problem):
    def __init__(self, d: int, shift: torch.Tensor):
        super().__init__()
        self.shift = shift
        self.t = _t(d, shift.device)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        y = t_asy_lambda(t_osz(x - self.shift), self.t)
        return (y * y - 10.0 * torch.cos(2.0 * math.pi * y) + 10.0).sum(-1)


class CecF3(Problem):
    def __init__(self, d: int, shift: torch.Tensor):
        super().__init__()
        self.shift = shift
        self.t = _t(d, shift.device)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        y = t_asy_lambda(t_osz(x - self.shift), self.t)
        s1 = (y * y).mean(-1)
        s2 = torch.cos(2.0 * math.pi * y).mean(-1)
        return -20.0 * torch.exp(-0.2 * torch.sqrt(s1)) - torch.exp(s2) + 20.0 + math.e


class CecF12(Problem):
    """f12, shifted Rosenbrock: z = x - o, sum_{j<D-1} 100 (z_j^2 - z_{j+1})^2 + (z_j - 1)^2.
    Written with slices, as EvoX's own Rosenbrock."""

    def __init__(self, d: int, shift: torch.Tensor):
        super().__init__()
        self.shift = shift

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        z = x - self.shift
        a = z[:, :-1]
        return (100.0 * (a * a - z[:, 1:]) ** 2 + (a - 1.0) ** 2).sum(-1)


CEC = {"cec-f1": (CecF1, -100.0, 100.0), "cec-f2": (CecF2, -5.0, 5.0), "cec-f3": (CecF3, -32.0, 32.0),
       "cec-f12": (CecF12, -100.0, 100.0)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--lo", type=float, required=True)
    p.add_argument("--hi", type=float, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    make_shift(a.d, a.lo, a.hi, a.seed).tofile(a.out)


if __name__ == "__main__":
    main()
