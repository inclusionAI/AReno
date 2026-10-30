"""KIVI KV-cache correctness against an independent PyTorch reference.

The reference below is written from the KIVI definition (per-channel key
groups, per-token value groups, fp16 scale/zero, residual window) and the
documented packing layout; it does not reuse areno's kernels or helpers.
"""

from __future__ import annotations

import pytest
import torch

from areno.engine.layers.attention_backend.kivi import KiviInferBackend, KiviKVCache, KiviSpec
from areno.engine.runtime.metadata import InferMeta

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="KIVI kernels require CUDA")

BLOCK_SIZE = 256
HEAD_DIM = 128
FP16_MAX = 65504.0


# ---------------------------------------------------------------- reference


def ref_quantized_length(tokens: int, residual: int, group: int) -> int:
    past = tokens - 1 - residual
    return 0 if past < 0 else (past // group) * group


def ref_quantize(x: torch.Tensor, bits: int, dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min/max asymmetric quantization reducing over ``dim`` (fp32 math, fp16 metadata)."""

    qmax = (1 << bits) - 1
    x = x.float()
    lo = x.amin(dim=dim, keepdim=True)
    hi = x.amax(dim=dim, keepdim=True)
    scale = ((hi - lo) / qmax).clamp(max=FP16_MAX).half().float()
    zero = lo.clamp(-FP16_MAX, FP16_MAX).half().float()
    inv = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    codes = torch.floor((x - zero) * inv + 0.5).clamp(0, qmax)
    return codes.to(torch.int32), scale.squeeze(dim).half(), zero.squeeze(dim).half()


def ref_pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Byte j holds channel ``j * (8 // bits) + i`` at bit offset ``i * bits``."""

    pack = 8 // bits
    out = torch.zeros(*codes.shape[:-1], codes.shape[-1] // pack, dtype=torch.int32, device=codes.device)
    for i in range(pack):
        out |= codes[..., i::pack] << (i * bits)
    return out.to(torch.uint8)


def ref_unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    pack = 8 // bits
    mask = (1 << bits) - 1
    data = packed.to(torch.int32)
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * pack, dtype=torch.int32, device=packed.device)
    for i in range(pack):
        out[..., i::pack] = (data >> (i * bits)) & mask
    return out


def stored_rows(cache: KiviKVCache, block_row: torch.Tensor, positions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Gather the stored codes/metadata of ``positions`` of one sequence (token-major)."""

    group = cache.spec.group_size
    blocks = block_row[positions // cache.block_size].long()
    offsets = (positions % cache.block_size).long()
    return {
        "k_codes": cache.k_codes[blocks, :, offsets],
        "k_scale": cache.k_scale[blocks, :, offsets // group],
        "k_zero": cache.k_zero[blocks, :, offsets // group],
        "v_codes": cache.v_codes[blocks, :, offsets],
        "v_scale": cache.v_scale[blocks, :, offsets],
        "v_zero": cache.v_zero[blocks, :, offsets],
    }


def ref_dequant_stored(cache: KiviKVCache, block_row: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize positions [0, length) from the stored tensors with the reference unpacker."""

    if length == 0:
        empty = torch.empty(0, cache.num_kv_heads, cache.head_dim, device=block_row.device)
        return empty, empty
    group = cache.spec.group_size
    rows = stored_rows(cache, block_row, torch.arange(length, device=block_row.device))
    k = ref_unpack(rows["k_codes"], cache.spec.bits).float() * rows["k_scale"].float() + rows["k_zero"].float()
    v_scale = rows["v_scale"].float().repeat_interleave(group, dim=-1)
    v_zero = rows["v_zero"].float().repeat_interleave(group, dim=-1)
    v = ref_unpack(rows["v_codes"], cache.spec.bits).float() * v_scale + v_zero
    return k, v


def ref_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Full attention of ``(queries, q_heads, dim)`` against all ``(keys, kv_heads, dim)``."""

    repeat = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(repeat, dim=1)
    v = v.float().repeat_interleave(repeat, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * scale
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v)


def ref_tail_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_start: int, scale: float):
    """Causal attention for queries at positions query_start + i over keys [0, len(k))."""

    outs = [
        ref_attention(q[i : i + 1], k[: query_start + i + 1], v[: query_start + i + 1], scale) for i in range(len(q))
    ]
    return torch.cat(outs)


# ------------------------------------------------------------------ harness


class Sequence:
    """One sequence's full-precision history plus its cache coordinates."""

    def __init__(self, num_kv_heads: int, slot: int, block_row: torch.Tensor):
        self.k = torch.empty(0, num_kv_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        self.v = torch.empty_like(self.k)
        self.slot = slot
        self.block_row = block_row

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k = torch.cat([self.k, k])
        self.v = torch.cat([self.v, v])


def make_cache(spec: KiviSpec, *, num_blocks: int, num_slots: int, num_kv_heads: int) -> KiviKVCache:
    return KiviKVCache.allocate(
        spec,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_slots=num_slots,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )


def make_block_table(rows: int, blocks_per_row: int, generator: torch.Generator) -> torch.Tensor:
    # Shuffled physical blocks exercise block-table indirection.
    order = torch.randperm(rows * blocks_per_row, generator=generator)
    return order.view(rows, blocks_per_row).to(device="cuda", dtype=torch.int32)


def random_kv(tokens: int, heads: int, generator: torch.Generator) -> torch.Tensor:
    x = torch.randn(tokens, heads, HEAD_DIM, generator=generator)
    # Key-like outlier channels make per-channel scales matter.
    x[..., 3] *= 12.0
    x[..., 77] += 6.0
    return x.to(device="cuda", dtype=torch.bfloat16)


def prefill(backend, cache, seqs, q_tokens, k_tokens, v_tokens, *, chunked: bool):
    lengths = [len(k) for k in k_tokens]
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32, device="cuda")
    cached = [len(seq.k) for seq in seqs]
    meta = InferMeta(
        mode="prefill",
        cu_seqlens=cu,
        max_seqlen=max(lengths),
        cache_seqlens=torch.tensor(cached, dtype=torch.int32, device="cuda") if chunked else None,
        block_table=torch.stack([seq.block_row for seq in seqs]),
        recurrent_slots=torch.tensor([seq.slot for seq in seqs], dtype=torch.long, device="cuda"),
    )
    q = torch.cat(q_tokens).unsqueeze(0)
    k = torch.cat(k_tokens).unsqueeze(0)
    v = torch.cat(v_tokens).unsqueeze(0)
    out = backend(q, k, v, cache, meta).squeeze(0)
    for seq, k_row, v_row in zip(seqs, k_tokens, v_tokens, strict=True):
        seq.append(k_row, v_row)
    return out


def decode(backend, cache, seqs, q, k, v):
    meta = InferMeta(
        mode="decode",
        cache_seqlens=torch.tensor([len(seq.k) for seq in seqs], dtype=torch.int32, device="cuda"),
        block_table=torch.stack([seq.block_row for seq in seqs]),
        recurrent_slots=torch.tensor([seq.slot for seq in seqs], dtype=torch.long, device="cuda"),
    )
    out = backend(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), cache, meta).squeeze(1)
    for row, seq in enumerate(seqs):
        seq.append(k[row : row + 1], v[row : row + 1])
    return out


def assert_state_matches_reference(cache: KiviKVCache, seq: Sequence) -> None:
    spec = cache.spec
    n = len(seq.k)
    quantized = ref_quantized_length(n, spec.residual_length, spec.group_size)
    group = spec.group_size
    if quantized:
        positions = torch.arange(quantized, device="cuda")
        stored = stored_rows(cache, seq.block_row, positions)
        k_groups = seq.k[:quantized].float().view(-1, group, *seq.k.shape[1:])
        k_codes, k_scale, k_zero = ref_quantize(k_groups, spec.bits, dim=1)
        v_groups = seq.v[:quantized].float().view(quantized, seq.v.shape[1], -1, group)
        v_codes, v_scale, v_zero = ref_quantize(v_groups, spec.bits, dim=3)
        # Metadata is computed with identical fp32 ops; allow one fp16 ulp for
        # division lowering differences between Triton and PyTorch.
        torch.testing.assert_close(
            stored["k_zero"].float(), k_zero.repeat_interleave(group, dim=0).float(), atol=0, rtol=0
        )
        torch.testing.assert_close(
            stored["k_scale"].float(), k_scale.repeat_interleave(group, dim=0).float(), atol=0, rtol=1e-3
        )
        torch.testing.assert_close(stored["v_zero"].float(), v_zero.float(), atol=0, rtol=0)
        torch.testing.assert_close(stored["v_scale"].float(), v_scale.float(), atol=0, rtol=1e-3)
        got_k = ref_unpack(stored["k_codes"], spec.bits)
        got_v = ref_unpack(stored["v_codes"], spec.bits)
        want_k = k_codes.flatten(0, 1)
        want_v = v_codes.flatten(2, 3)
        # Codes may differ by one step only at exact rounding ties.
        assert (got_k - want_k).abs().max() <= 1
        assert (got_v - want_v).abs().max() <= 1
        assert (got_k != want_k).float().mean() < 1e-3
        assert (got_v != want_v).float().mean() < 1e-3
        # Packed bytes of the reference codes decode identically (layout check).
        assert torch.equal(ref_unpack(ref_pack(want_k, spec.bits), spec.bits), want_k)
    ring_positions = torch.arange(quantized, n, device="cuda") % spec.ring_size
    assert torch.equal(cache.k_ring[seq.slot][:, ring_positions].transpose(0, 1), seq.k[quantized:])
    assert torch.equal(cache.v_ring[seq.slot][:, ring_positions].transpose(0, 1), seq.v[quantized:])


def reference_decode(cache: KiviKVCache, seq: Sequence, q: torch.Tensor, scale: float) -> torch.Tensor:
    """Reference for the last decode step of ``seq`` using the stored quantized tensors."""

    n = len(seq.k)
    quantized = ref_quantized_length(n, cache.spec.residual_length, cache.spec.group_size)
    k_q, v_q = ref_dequant_stored(cache, seq.block_row, quantized)
    keys = torch.cat([k_q, seq.k[quantized:].float()])
    values = torch.cat([v_q, seq.v[quantized:].float()])
    return ref_attention(q.unsqueeze(0), keys, values, scale).squeeze(0)


# -------------------------------------------------------------------- tests


@pytest.mark.parametrize("bits", [2, 4])
def test_triton_pack_and_unpack_match_reference_layout(bits):
    import triton
    import triton.language as tl

    from areno.accel.kernels.kivi import _pack, _unpack

    @triton.jit
    def roundtrip(CODES, PACKED, UNPACKED, ROWS: tl.constexpr, DIM: tl.constexpr, BITS: tl.constexpr):
        rows = tl.arange(0, ROWS)[:, None]
        byte = tl.arange(0, DIM * BITS // 8)[None, :]
        dim = tl.arange(0, DIM)[None, :]
        tl.store(PACKED + rows * (DIM * BITS // 8) + byte, _pack(tl.load(CODES + rows * DIM + dim), ROWS, DIM, BITS))
        packed = tl.load(PACKED + rows * (DIM * BITS // 8) + byte)
        tl.store(UNPACKED + rows * DIM + dim, _unpack(packed, ROWS, DIM, BITS))

    generator = torch.Generator().manual_seed(bits)
    codes = torch.randint(0, 1 << bits, (64, HEAD_DIM), generator=generator, dtype=torch.int32).cuda()
    packed = torch.empty(64, HEAD_DIM * bits // 8, dtype=torch.uint8, device="cuda")
    unpacked = torch.empty(64, HEAD_DIM, dtype=torch.float32, device="cuda")
    roundtrip[(1,)](codes, packed, unpacked, 64, HEAD_DIM, bits)
    assert torch.equal(packed, ref_pack(codes, bits))
    assert torch.equal(unpacked, codes.float())


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@torch.inference_mode()
def test_prefill_store_matches_reference_quantizer(bits, group_size):
    generator = torch.Generator().manual_seed(bits * 100 + group_size)
    spec = KiviSpec(bits=bits, group_size=group_size, residual_length=128)
    lengths = [1, 127, 128, 129, 159, 160, 161, 1024, 1500]
    blocks_per_row = 8
    cache = make_cache(spec, num_blocks=len(lengths) * blocks_per_row, num_slots=len(lengths), num_kv_heads=2)
    table = make_block_table(len(lengths), blocks_per_row, generator)
    seqs = [Sequence(2, slot, table[slot]) for slot in range(len(lengths))]
    k_rows = [random_kv(n, 2, generator) for n in lengths]
    v_rows = [random_kv(n, 2, generator) for n in lengths]
    # A constant group must round-trip exactly through a zero scale.
    k_rows[-1][:group_size] = 0.5
    q_rows = [random_kv(n, 4, generator) for n in lengths]
    prefill(KiviInferBackend("flash"), cache, seqs, q_rows, k_rows, v_rows, chunked=False)
    for seq in seqs:
        assert_state_matches_reference(cache, seq)


@pytest.mark.parametrize(
    "bits,total,group_size",
    [(bits, total, 32) for bits in (2, 4) for total in (1024, 4096, 8192)] + [(2, 1024, 64), (4, 1024, 128)],
)
@torch.inference_mode()
def test_decode_crosses_residual_boundaries_and_matches_reference(bits, total, group_size):
    generator = torch.Generator().manual_seed(total + bits + group_size)
    spec = KiviSpec(bits=bits, group_size=group_size, residual_length=128)
    steps = 2 * spec.group_size + 7  # crosses at least two flush boundaries
    rows = 3
    blocks_per_row = -(-(total + steps) // BLOCK_SIZE)
    cache = make_cache(spec, num_blocks=rows * blocks_per_row, num_slots=rows + 1, num_kv_heads=8)
    table = make_block_table(rows, blocks_per_row, generator)
    # Different prompt lengths put each row at a different phase of the ring.
    prompt_lengths = [total - steps, total - steps - 13, spec.residual_length - 5]
    seqs = [Sequence(8, slot=row + 1, block_row=table[row]) for row in range(rows)]
    backend = KiviInferBackend("flash")
    prefill(
        backend,
        cache,
        seqs,
        [random_kv(n, 16, generator) for n in prompt_lengths],
        [random_kv(n, 8, generator) for n in prompt_lengths],
        [random_kv(n, 8, generator) for n in prompt_lengths],
        chunked=False,
    )
    scale = HEAD_DIM**-0.5
    for _ in range(steps):
        q = random_kv(rows, 16, generator)
        out = decode(backend, cache, seqs, q, random_kv(rows, 8, generator), random_kv(rows, 8, generator))
        for row, seq in enumerate(seqs):
            want = reference_decode(cache, seq, q[row], scale)
            torch.testing.assert_close(out[row].float(), want, atol=1e-2, rtol=1e-2)
    for seq in seqs:
        assert_state_matches_reference(cache, seq)


def dense_tail_attention(backend: str, q, k, v, query_start: int, scale: float) -> torch.Tensor:
    """The unmodified dense kernel the prefill path delegates to, on caller-built K/V."""

    if backend == "native":
        from areno.accel.attention import areno_causal_attention

        repeat = q.shape[1] // k.shape[1]
        out = areno_causal_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0),
            v.repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0),
            query_start=query_start,
            softmax_scale=scale,
            force_native=True,
        )
        return out.squeeze(0).transpose(0, 1)
    from flash_attn import flash_attn_varlen_func

    cu_q = torch.tensor([0, len(q)], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, len(k)], dtype=torch.int32, device="cuda")
    return flash_attn_varlen_func(q, k, v, cu_q, cu_k, len(q), len(k), causal=True, softmax_scale=scale)


@pytest.mark.parametrize("backend", ["flash", "native"])
@pytest.mark.parametrize("bits", [2, 4])
@torch.inference_mode()
def test_chunked_prefill_reads_quantized_prefix(backend, bits):
    generator = torch.Generator().manual_seed(7 + bits)
    spec = KiviSpec(bits=bits, group_size=32, residual_length=128)
    chunks = [700, 333, 1]  # second chunk straddles ring residue and new groups
    cache = make_cache(spec, num_blocks=8, num_slots=2, num_kv_heads=8)
    seq = Sequence(8, slot=1, block_row=make_block_table(1, 8, generator)[0])
    kivi = KiviInferBackend(backend)
    scale = HEAD_DIM**-0.5
    for index, length in enumerate(chunks):
        q = random_kv(length, 16, generator)
        k = random_kv(length, 8, generator)
        v = random_kv(length, 8, generator)
        old = len(seq.k)
        quantized = ref_quantized_length(old, spec.residual_length, spec.group_size)
        k_prefix, v_prefix = ref_dequant_stored(cache, seq.block_row, quantized)
        # Semantics: the chunk sees the dequantized prefix, the full-precision
        # ring residue and its own fresh tokens, materialized in the model dtype.
        keys = torch.cat([k_prefix.to(k.dtype), seq.k[quantized:], k])
        values = torch.cat([v_prefix.to(v.dtype), seq.v[quantized:], v])
        out = prefill(kivi, cache, [seq], [q], [k], [v], chunked=index > 0)
        # Same dense kernel on independently built inputs: bitwise equal.
        assert torch.equal(out, dense_tail_attention(backend, q, keys, values, old, scale))
        if backend == "flash":
            # The dense kernel's own bf16 error versus fp32 is the baseline's.
            want = ref_tail_attention(q, keys.float(), values.float(), old, scale)
            error = (out.float() - want).abs().max() / want.abs().max()
            assert error < 1e-2
        assert_state_matches_reference(cache, seq)


@torch.inference_mode()
def test_slot_and_block_reuse_does_not_leak_previous_sequence():
    generator = torch.Generator().manual_seed(11)
    spec = KiviSpec(bits=2, group_size=32, residual_length=128)
    backend = KiviInferBackend("flash")
    shared = make_cache(spec, num_blocks=4, num_slots=2, num_kv_heads=8)
    table = make_block_table(1, 4, generator)
    # Sequence A fills slot 1 and the blocks, then B reuses both from scratch.
    a = Sequence(8, slot=1, block_row=table[0])
    prefill(
        backend,
        shared,
        [a],
        [random_kv(600, 16, generator)],
        [random_kv(600, 8, generator)],
        [random_kv(600, 8, generator)],
        chunked=False,
    )
    b_inputs = [random_kv(300, 16, generator), random_kv(300, 8, generator), random_kv(300, 8, generator)]
    b_steps = [(random_kv(1, 16, generator), random_kv(1, 8, generator), random_kv(1, 8, generator)) for _ in range(40)]

    def run_b(cache: KiviKVCache) -> list[torch.Tensor]:
        seq = Sequence(8, slot=1, block_row=table[0])
        outs = [prefill(backend, cache, [seq], [b_inputs[0]], [b_inputs[1]], [b_inputs[2]], chunked=False)]
        outs.extend(decode(backend, cache, [seq], *step) for step in b_steps)
        return outs

    fresh = make_cache(spec, num_blocks=4, num_slots=2, num_kv_heads=8)
    for got, want in zip(run_b(shared), run_b(fresh), strict=True):
        assert torch.equal(got, want)


@torch.inference_mode()
def test_decode_cuda_graph_replay_matches_eager():
    generator = torch.Generator().manual_seed(5)
    spec = KiviSpec(bits=2, group_size=32, residual_length=128)
    backend = KiviInferBackend("flash")
    rows, blocks_per_row = 4, 4
    table = make_block_table(rows, blocks_per_row, generator)
    lengths = [150, 159, 160, 700]  # includes a row that flushes on its next step
    k_rows = [random_kv(n, 8, generator) for n in lengths]
    v_rows = [random_kv(n, 8, generator) for n in lengths]
    q_rows = [random_kv(n, 16, generator) for n in lengths]
    steps = [
        (random_kv(rows, 16, generator), random_kv(rows, 8, generator), random_kv(rows, 8, generator)) for _ in range(3)
    ]

    def setup():
        cache = make_cache(spec, num_blocks=rows * blocks_per_row, num_slots=rows, num_kv_heads=8)
        seqs = [Sequence(8, slot=row, block_row=table[row]) for row in range(rows)]
        prefill(backend, cache, seqs, q_rows, k_rows, v_rows, chunked=False)
        return cache, seqs

    eager_cache, eager_seqs = setup()
    eager = [decode(backend, eager_cache, eager_seqs, *step) for step in steps]

    graph_cache, graph_seqs = setup()
    static_q, static_k, static_v = (t.clone() for t in steps[0])
    seqlens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    meta = InferMeta(
        mode="decode",
        cache_seqlens=seqlens,
        block_table=table,
        recurrent_slots=torch.arange(rows, device="cuda"),
    )

    def step():
        return backend(static_q.unsqueeze(1), static_k.unsqueeze(1), static_v.unsqueeze(1), graph_cache, meta)

    # Warm up on a throwaway cache so kernels compile outside capture.
    warm_cache, _ = setup()
    warm_out = backend(static_q.unsqueeze(1), static_k.unsqueeze(1), static_v.unsqueeze(1), warm_cache, meta)
    del warm_out, warm_cache
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = step()
    for index, (q, k, v) in enumerate(steps):
        static_q.copy_(q)
        static_k.copy_(k)
        static_v.copy_(v)
        seqlens.copy_(torch.tensor([n + index for n in lengths], dtype=torch.int32))
        graph.replay()
        torch.testing.assert_close(static_out.squeeze(1), eager[index], atol=0, rtol=0)


def test_clear_releases_kivi_storage_attached_to_attention_layers():
    import gc
    from types import SimpleNamespace

    from areno.engine.config import ModelConfig
    from areno.engine.layers.attention import CausalSelfAttention
    from areno.engine.layers.attention_backend.kivi import attach_kivi_kv_caches

    config = ModelConfig(hidden_size=512, num_attention_heads=4, num_key_value_heads=2, head_dim=128)
    model = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=CausalSelfAttention(config, index).cuda()) for index in range(3)]
    )
    spec = KiviSpec(bits=2, group_size=32, residual_length=128)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    caches = attach_kivi_kv_caches(
        model,
        spec,
        num_blocks=16,
        block_size=BLOCK_SIZE,
        num_slots=5,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    attached = torch.cuda.memory_allocated() - baseline
    expected = sum(cache.nbytes() for cache in caches)
    # The caching allocator rounds each tensor up to 512 bytes.
    assert expected <= attached <= expected + 512 * 10 * len(caches)
    assert all(layer.self_attn.kivi_cache is cache for layer, cache in zip(model.layers, caches, strict=True))
    del caches
    for layer in model.layers:
        layer.self_attn.clear_kv_cache()
    gc.collect()
    assert torch.cuda.memory_allocated() == baseline


@pytest.mark.parametrize("bits,max_ratio", [(2, 0.25), (4, 0.375)])
def test_long_context_cache_bytes_below_bf16(bits, max_ratio):
    spec = KiviSpec(bits=bits, group_size=32, residual_length=128)
    tokens, kv_heads = 8192, 8
    blocks = tokens // BLOCK_SIZE
    cache = make_cache(spec, num_blocks=blocks, num_slots=1, num_kv_heads=kv_heads)
    bf16_bytes = 2 * tokens * kv_heads * HEAD_DIM * 2
    # Includes scales, zeros and the full-precision residual ring.
    assert cache.nbytes() < max_ratio * bf16_bytes
