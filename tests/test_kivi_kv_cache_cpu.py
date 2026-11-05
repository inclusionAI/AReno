"""CPU checks for KIVI KV-cache configuration, residual bookkeeping and planning."""

from __future__ import annotations

import warnings

import pytest
import torch

from areno.engine.config import ModelConfig, RuntimeConfig
from areno.engine.layers.attention_backend.kivi import (
    KiviSpec,
    _build_prefill_plan,
    unpack_codes,
    validate_kivi_geometry,
)
from areno.engine.runtime.metadata import InferMeta


def _simulate_decode(spec: KiviSpec, tokens: int) -> list[int]:
    """Replay the decode rule token by token: flush a group when n - R hits a positive multiple of G."""

    quantized, history = 0, []
    for cached in range(tokens):
        past = cached - spec.residual_length
        if past >= spec.group_size and past % spec.group_size == 0:
            assert quantized == past - spec.group_size  # the flushed group is the oldest ring group
            quantized += spec.group_size
        assert cached + 1 - quantized <= spec.ring_size  # the ring never overflows
        history.append(quantized)
    return history


@pytest.mark.parametrize("residual", [0, 1, 31, 32, 128])
@pytest.mark.parametrize("group", [16, 32, 64])
def test_quantized_length_matches_decode_simulation(residual, group):
    spec = KiviSpec(bits=2, group_size=group, residual_length=residual)
    for cached, quantized in enumerate(_simulate_decode(spec, 600)):
        tokens = cached + 1
        assert spec.quantized_length(tokens) == quantized
        # The newest token always sees at least R full-precision predecessors.
        assert tokens - quantized >= min(tokens, residual + 1)


def test_prefill_plan_covers_groups_and_ring_for_mixed_rows():
    spec = KiviSpec(bits=2, group_size=32, residual_length=128)
    block_size = 256
    # Row 0 is fresh; row 1 extends 300 cached tokens by 70 new ones.
    meta = InferMeta(
        mode="prefill",
        cu_seqlens=torch.tensor([0, 200, 270], dtype=torch.int32),
        max_seqlen=200,
        cache_seqlens=torch.tensor([0, 300], dtype=torch.int32),
        block_table=torch.tensor([[5, 6], [7, 8]], dtype=torch.int32),
        recurrent_slots=torch.tensor([3, 4]),
    )
    plan = _build_prefill_plan(meta, spec, block_size, torch.device("cpu"))
    assert plan.has_prefix
    assert plan.rows == [(0, 200, 0, 3), (200, 70, 300, 4)]
    # Row 0: q(200) = 64 -> groups at 0, 32. Row 1: q(300) = 160 -> q(370) = 224.
    assert spec.quantized_length(200) == 64
    assert spec.quantized_length(300) == 160 and spec.quantized_length(370) == 224
    assert plan.group_block.tolist() == [5, 5, 7, 7]
    assert plan.group_offset.tolist() == [0, 32, 160, 192]
    # Staged axis: row 0 occupies [0, 200); row 1 stages ring residue [160, 300) then 70 fresh tokens.
    assert plan.stage_tokens == 200 + (370 - 160)
    assert plan.group_src.tolist() == [0, 32, 200, 232]
    assert plan.stage_ring_slot.tolist() == [4] * 140
    assert plan.stage_ring_pos.tolist() == [p % spec.ring_size for p in range(160, 300)]
    assert plan.stage_fresh_dst.tolist()[:200] == list(range(200))
    assert plan.stage_fresh_dst.tolist()[200:] == list(range(200 + 140, 200 + 210))
    # Ring writes: row 0 keeps [64, 200); row 1 adds fresh [300, 370) (old residue stays in place).
    expected_ring = [(3, p % spec.ring_size, p) for p in range(64, 200)]
    expected_ring += [(4, p % spec.ring_size, 200 + p - 300) for p in range(300, 370)]
    got_ring = list(zip(plan.ring_slot.tolist(), plan.ring_pos.tolist(), plan.ring_src.tolist(), strict=True))
    assert got_ring == expected_ring


def test_prefill_plan_without_prefix_reads_fresh_tokens_directly():
    spec = KiviSpec(bits=4, group_size=32, residual_length=128)
    meta = InferMeta(
        mode="prefill",
        cu_seqlens=torch.tensor([0, 100, 400], dtype=torch.int32),
        max_seqlen=300,
        block_table=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        recurrent_slots=torch.tensor([0, 1]),
    )
    plan = _build_prefill_plan(meta, spec, 256, torch.device("cpu"))
    assert not plan.has_prefix
    assert plan.stage_ring_dst is None
    # Only row 1 (300 tokens, q = 160) has complete groups; sources index the fresh axis.
    assert plan.group_src.tolist() == [100 + p for p in range(0, 160, 32)]


@pytest.mark.parametrize("bits", [2, 4])
def test_unpack_codes_inverts_interleaved_packing(bits):
    generator = torch.Generator().manual_seed(bits)
    codes = torch.randint(0, 1 << bits, (3, 5, 128), generator=generator)
    pack = 8 // bits
    packed = torch.zeros(3, 5, 128 // pack, dtype=torch.int32)
    for i in range(pack):
        packed |= codes[..., i::pack] << (i * bits)
    assert torch.equal(unpack_codes(packed.to(torch.uint8), bits), codes.float())


@pytest.mark.parametrize(
    "spec,block_size,head_dim,match",
    [
        (KiviSpec(bits=3, group_size=32, residual_length=128), 256, 128, "2 or 4 bits"),
        (KiviSpec(bits=2, group_size=24, residual_length=128), 256, 128, "power of two"),
        (KiviSpec(bits=2, group_size=16, residual_length=128), 256, 128, ">= 32"),
        (KiviSpec(bits=2, group_size=32, residual_length=-1), 256, 128, "non-negative"),
        (KiviSpec(bits=2, group_size=32, residual_length=128), 256, 96, "head dim"),
        (KiviSpec(bits=2, group_size=128, residual_length=128), 64, 128, "block size"),
    ],
)
def test_validate_geometry_rejects_unsupported_shapes(spec, block_size, head_dim, match):
    with pytest.raises(ValueError, match=match):
        validate_kivi_geometry(spec, block_size=block_size, head_dim=head_dim)


def test_runtime_config_defaults_to_model_dtype_cache():
    runtime = RuntimeConfig()
    assert runtime.kv_cache_dtype == "auto"
    assert (runtime.kivi_group_size, runtime.kivi_residual_length) == (32, 128)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"kv_cache_dtype": "int2"}, "kv_cache_dtype"),
        ({"kivi_group_size": 48}, "kivi_group_size"),
        ({"kivi_group_size": 8}, "kivi_group_size"),
        ({"kivi_group_size": 16}, "kivi_group_size"),
        ({"kivi_residual_length": -1}, "kivi_residual_length"),
    ],
)
def test_runtime_config_rejects_invalid_kivi_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RuntimeConfig(**kwargs)


def test_resolve_keeps_kivi_for_supported_dense_model():
    runtime = RuntimeConfig(kv_cache_dtype="kivi2")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        runtime.resolve_kv_cache_dtype(model=ModelConfig(model_type="qwen3", head_dim=128))
    assert runtime.kv_cache_dtype == "kivi2"


@pytest.mark.parametrize(
    "runtime_kwargs,model_kwargs,reason",
    [
        ({}, {"model_type": "bailing_moe_v3"}, "no KIVI attention cache"),
        ({"speculative_draft_tokens": 2}, {}, "speculative decoding"),
        ({"device_type": "npu"}, {}, "require CUDA"),
        ({}, {"head_dim": 96}, "head_dim 96"),
    ],
)
def test_resolve_falls_back_to_model_dtype_cache_with_warning(runtime_kwargs, model_kwargs, reason):
    runtime = RuntimeConfig(kv_cache_dtype="kivi4", **runtime_kwargs)
    model = ModelConfig(**{"model_type": "qwen3", "head_dim": 128, **model_kwargs})
    with pytest.warns(RuntimeWarning, match=reason):
        runtime.resolve_kv_cache_dtype(model=model)
    assert runtime.kv_cache_dtype == "auto"
