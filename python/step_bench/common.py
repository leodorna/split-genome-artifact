"""Shared timing harness and bit-packed binary operators.

The bit-packed helpers (``uniform_crossover`` / ``bitflip_mutation``) reproduce
the semantics of ``evoopt/benchmarks/divergence/kernels/{highdim,individual}``
uniform crossover and bitflip mutation. Neither operator is shipped by EvoX, so
they are provided here for the Python side of the benchmark.
"""

from __future__ import annotations

import time

import torch

CSV_HEADER = "operator,N,islands,D,pv,time_us"


def _raise_recompile_limit(limit: int = 256) -> None:
    """Every sweep tuple is a distinct shape; keep dynamo from falling back to eager."""
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, limit)


def random_int32(*shape, device):
    """Uniform random int32 tensor (full 32-bit range, interpreted as bits)."""
    return torch.randint(-(2**31), 2**31, shape, device=device, dtype=torch.int64).to(torch.int32)


def _popcount(x):
    """Integer popcount per int32 word (SWAR doubling, exact; no int64)."""
    x = (x & 0x55555555) + ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x & 0x0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F)
    return (x + (x >> 8) + (x >> 16) + (x >> 24)) & 0xFF


def uniform_crossover(pop, islands, pc=0.5):
    """Bit-packed uniform crossover (evoopt ``UX``), two children per couple.

    ``pop`` is an int32 tensor ``(total, n_words)`` holding ``D = 32 * n_words``
    bits per individual. Parents are paired within each island
    (``p2 = base + (i + N // 2) % N``), matching ``ux_rand_select``.
    """
    total, n_words = pop.shape
    n = total // islands
    device = pop.device

    idx = torch.arange(total, device=device)
    within = idx % n
    island = idx // n
    p1 = idx
    p2 = island * n + (within + n // 2) % n

    v1 = pop[p1]
    v2 = pop[p2]

    active = torch.rand(total, device=device) < pc
    mask = random_int32(total, n_words, device=device)

    c1 = (~mask & v1) | (mask & v2)
    c2 = (mask & v1) | (~mask & v2)

    c1 = torch.where(active.unsqueeze(1), c1, v1)
    c2 = torch.where(active.unsqueeze(1), c2, v2)

    return torch.stack((c1, c2), dim=1).reshape(2 * total, n_words)


def bitflip_mutation(pop, pm, _budget_bytes=4 << 30):
    """Bit-packed bitflip mutation (evoopt ``BF``).

    Each of the 32 bits of every word flips independently with probability
    ``pm``. Vectorized as ``sum(cond << bit)`` (equivalent to ``mask |= cond <<
    bit`` because the bits are disjoint); rows are chunked so the transient
    ``(rows, n_words, 32)`` int32 tensor stays under ``_budget_bytes``.

    The vectorized form matters: a 32-iteration ``torch.rand`` loop makes
    inductor recompute the whole bitflip inside a downstream reduction.
    """
    total, n_words = pop.shape
    device = pop.device
    shifts = torch.arange(32, device=device, dtype=torch.int32)

    def mask_for(rows):
        bits = (torch.rand(rows, n_words, 32, device=device) < pm).to(torch.int32)
        return (bits << shifts).sum(dim=-1, dtype=torch.int32)

    chunk = max(1, min(total, _budget_bytes // max(1, n_words * 32 * 4)))
    if chunk >= total:
        return pop ^ mask_for(total)

    out = torch.empty_like(pop)
    for i0 in range(0, total, chunk):
        i1 = min(i0 + chunk, total)
        out[i0:i1] = pop[i0:i1] ^ mask_for(i1 - i0)
    return out


def time_op(step, reps=20, warmup=10, device="cuda"):
    """Time ``step()`` and return per-rep times in microseconds.

    Uses CUDA events on GPU (GPU-only time, like the evoopt benchmark) and
    ``time.perf_counter`` on CPU.
    """
    for _ in range(warmup):
        step()

    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(reps):
            start.record()
            step()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1e3)  # ms -> us
        return times

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        step()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return times


def run_benchmark(op_specs, tuples, reps=20, warmup=10, out=None, device=None, compile_ops=None, limit=None, start=0):
    """Run every ``(name, builder)`` spec over the sweep tuples and emit CSV.

    ``builder(I, N, D, pv, device, compile_ops)`` returns ``(step, report_pv)``.
    OOM configs are recorded with ``time_us == oom`` and skipped.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compile_ops is None:
        compile_ops = device == "cuda"
    if compile_ops:
        _raise_recompile_limit()

    tuples = tuples[start:]
    if limit is not None:
        tuples = tuples[:limit]

    torch.set_default_device(device)

    sink = open(out, "w") if out else None

    def emit(line):
        if sink is not None:
            sink.write(line + "\n")
        else:
            print(line, flush=True)

    emit(CSV_HEADER)

    for name, builder in op_specs:
        for I, N, D, pv in tuples:
            step = None
            try:
                step, report_pv = builder(I, N, D, pv, device, compile_ops)
                times = time_op(step, reps=reps, warmup=warmup, device=device)
            except MemoryError:
                emit(f"{name},{N},{I},{D},{pv},oom")
                continue
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    emit(f"{name},{N},{I},{D},{pv},oom")
                    continue
                raise
            finally:
                if step is not None:
                    del step
                if device == "cuda":
                    torch.cuda.empty_cache()

            for t in times:
                emit(f"{name},{N},{I},{D},{report_pv},{t:.3f}")

    if sink is not None:
        sink.close()
