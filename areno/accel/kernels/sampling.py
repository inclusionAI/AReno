"""Sorting-free pivot truncation. Inputs are sanitized FP32 probabilities.

One program per row; positive FP32 bit patterns provide an exact ordered
search space. No approximation of the cutoff and no per-row host decisions.
This inference-only kernel has no backward.
"""

import triton
import triton.language as tl


@triton.jit
def _truncate(P, O, V: tl.constexpr, K: tl.constexpr, TOP_P: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ids = tl.arange(0, BLOCK)
    p = tl.load(P + row * V + ids, ids < V, 0)
    bits = p.to(tl.int32, bitcast=True)
    kl = tl.full((), 0, tl.int32)
    kh = tl.full((), 1065353217, tl.int32)  # next bit pattern after 1.0
    pl = tl.full((), 0, tl.int32)
    ph = tl.full((), 1065353217, tl.int32)
    # Bisection on ordered positive FP32 bits terminates in at most 30 steps.
    for _ in range(30):
        if K < V:
            km = (kl + kh) // 2
            count = tl.sum(((bits >= km) & (ids < V)).to(tl.int32), 0)
            ok = count >= K
            kl = tl.where(ok, km, kl)
            kh = tl.where(ok, kh, km)
        if TOP_P < 1.0:
            pm = (pl + ph) // 2
            mass = tl.sum(tl.where(bits >= pm, p, 0).to(tl.float64), 0)
            ok = mass > TOP_P
            pl = tl.where(ok, pm, pl)
            ph = tl.where(ok, ph, pm)
    cutoff = tl.maximum(kl, pl)
    q = cutoff.to(tl.float32, bitcast=True)
    greater = (bits > cutoff) & (ids < V)
    equal = (bits == cutoff) & (ids < V)
    count_gt = tl.sum(greater.to(tl.int32), 0)
    mass_gt = tl.sum(tl.where(greater, p, 0).to(tl.float64), 0)
    tie_rank = tl.cumsum(equal.to(tl.int32), 0) - 1
    keep = greater | (equal & (count_gt + tie_rank < K))
    if TOP_P < 1.0:
        # Strict exclusive-prefix > p matches the original sampler. Include
        # one further token when the preceding mass equals top_p exactly.
        keep = keep & (greater | (mass_gt + tie_rank.to(tl.float64) * q <= TOP_P))
    out = tl.where(keep, p, 0)
    total = tl.sum(out, 0)
    tl.store(O + row * V + ids, out / total, ids < V)


def truncate_probs(probs, top_k, top_p):
    import torch

    batch, vocab = probs.shape
    if vocab > 4096 and 0 < top_k <= 256:
        # The global top-k is contained in the union of tile-local top-k.
        # Compact in token-id order, retaining original (not renormalized)
        # probability mass for the subsequent nucleus cutoff.
        tiles = triton.cdiv(vocab, 4096)
        compact = torch.zeros((batch, tiles * top_k), device=probs.device, dtype=probs.dtype)
        ids = torch.zeros((batch, tiles * top_k), device=probs.device, dtype=torch.long)
        _compact_topk[(batch, tiles)](probs, compact, ids, vocab, top_k, tiles, 4096, num_warps=4)
        selected = truncate_probs(compact, top_k, top_p)
        # Padding candidates have zero mass; scatter_add avoids a padded id=0
        # overwriting a real token. Nonzero destinations are unique.
        return torch.zeros_like(probs).scatter_add_(1, ids, selected)
    out = torch.empty_like(probs)
    _truncate[(batch,)](
        probs, out, vocab, min(top_k, vocab) if top_k > 0 else vocab, top_p, triton.next_power_of_2(vocab), num_warps=32
    )
    return out


@triton.jit
def _compact_topk(P, C, I, V: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    ids = tile * BLOCK + offsets
    p = tl.load(P + row * V + ids, ids < V, 0)
    bits = p.to(tl.int32, bitcast=True)
    lo = tl.full((), 0, tl.int32)
    hi = tl.full((), 1065353217, tl.int32)
    for _ in range(30):
        mid = (lo + hi) // 2
        count = tl.sum(((bits >= mid) & (ids < V)).to(tl.int32), 0)
        ok = count >= K
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid)
    greater = (bits > lo) & (ids < V)
    equal = (bits == lo) & (ids < V)
    tie_rank = tl.cumsum(equal.to(tl.int32), 0)
    count_gt = tl.sum(greater.to(tl.int32), 0)
    keep = greater | (equal & (count_gt + tie_rank <= K))
    position = tl.cumsum(keep.to(tl.int32), 0) - 1
    dest = row * TILES * K + tile * K + position
    tl.store(C + dest, p, keep)
    tl.store(I + dest, ids, keep)
