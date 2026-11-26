"""Bit-packed operator semantics (no GPU required)."""

import torch

from step_bench.common import bitflip_mutation


def _reference_pack(bits):
    ref = torch.zeros(bits.shape[:-1], dtype=torch.int32)
    for bit in range(32):
        ref |= bits[..., bit].to(torch.int32) << bit
    return ref


def test_bitflip_packing_matches_loop():
    torch.manual_seed(0)
    rows, nw = 8, 5
    bits = torch.rand(rows, nw, 32) < 0.3
    shifts = torch.arange(32, dtype=torch.int32)
    packed = (bits.to(torch.int32) << shifts).sum(dim=-1, dtype=torch.int32)
    assert torch.equal(packed, _reference_pack(bits))


def test_bitflip_flip_rate():
    torch.manual_seed(0)
    rows, nw = 64, 16
    pop = torch.randint(-(2**31), 2**31, (rows, nw), dtype=torch.int32)
    out = bitflip_mutation(pop, 0.01)
    flips = pop ^ out
    ones = sum(int(flips[i, j]).bit_count() for i in range(rows) for j in range(nw))
    rate = ones / (rows * nw * 32)
    assert abs(rate - 0.01) < 0.005
