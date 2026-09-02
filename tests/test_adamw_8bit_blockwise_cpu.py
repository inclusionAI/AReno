from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import patch

import torch

from areno.engine.optim import AdamW8bit, set_optimizer_state_precision
from areno.engine.optim.adamw_8bit import (
    _dequantize_positive,
    _dequantize_symmetric,
    _quantize_positive,
    _quantize_symmetric,
)
from areno.engine.optim.dynamic_quant import (
    SIGNED_DYNAMIC_MAP,
    SIGNED_DYNAMIC_ZERO,
    UNSIGNED_DYNAMIC_MAP,
    UNSIGNED_DYNAMIC_ZERO,
)


def test_adamw8bit_uses_parameter_local_block_scales() -> None:
    first = torch.nn.Parameter(torch.zeros(9))
    second = torch.nn.Parameter(torch.zeros(3))
    optimizer = AdamW8bit(
        [first, second],
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=32,
        quant_block_size=4,
    )
    first.grad = torch.tensor([1000.0, 1.0, -1.0, 0.5, 0.25, -0.5, 0.75, -0.25, 0.125])
    second.grad = torch.tensor([0.01, -0.02, 0.03])

    optimizer.step()

    state = optimizer.state_dict()["state"][0]
    assert state["exp_avg_scale"].shape == (4,)
    assert state["exp_avg_sq_scale"].shape == (4,)
    assert state["exp_avg_scale"][0] > 1000 * state["exp_avg_scale"][-1]
    assert state["exp_avg_sq_scale"][0] > 1000 * state["exp_avg_sq_scale"][-1]
    torch.testing.assert_close(second, torch.tensor([-1.0e-3, 1.0e-3, -1.0e-3]), atol=1.0e-6, rtol=0.0)


def test_adamw8bit_dynamic_codebooks_match_paper_reference_construction() -> None:
    assert len(SIGNED_DYNAMIC_MAP) == len(UNSIGNED_DYNAMIC_MAP) == 256
    assert SIGNED_DYNAMIC_ZERO == 127
    assert UNSIGNED_DYNAMIC_ZERO == 0
    assert SIGNED_DYNAMIC_MAP[0] == -0.99296875
    assert SIGNED_DYNAMIC_MAP[-1] == UNSIGNED_DYNAMIC_MAP[-1] == 1.0
    assert all(left <= right for left, right in zip(SIGNED_DYNAMIC_MAP, SIGNED_DYNAMIC_MAP[1:]))
    assert all(left <= right for left, right in zip(UNSIGNED_DYNAMIC_MAP, UNSIGNED_DYNAMIC_MAP[1:]))


def test_adamw8bit_dynamic_codebook_golden_round_trip() -> None:
    signed = torch.tensor(SIGNED_DYNAMIC_MAP)
    unsigned = torch.tensor(UNSIGNED_DYNAMIC_MAP)

    signed_q, signed_scale = _quantize_symmetric(signed)
    unsigned_q, unsigned_scale = _quantize_positive(unsigned)

    expected_codes = torch.arange(256, dtype=torch.int64).to(torch.uint8)
    assert torch.equal(signed_q, expected_codes)
    assert torch.equal(unsigned_q, expected_codes)
    torch.testing.assert_close(_dequantize_symmetric(signed_q, signed_scale), signed)
    torch.testing.assert_close(_dequantize_positive(unsigned_q, unsigned_scale), unsigned)


def test_adamw8bit_unsigned_dynamic_map_preserves_small_second_moments() -> None:
    values = torch.tensor([0.0, 7.75e-7, 5.5e-5, 5.5e-3, 0.55, 1.0])

    quantized, scale = _quantize_positive(values)
    restored = _dequantize_positive(quantized, scale)

    torch.testing.assert_close(restored, values, rtol=0.15, atol=1.0e-8)


def test_adamw8bit_same_lr_does_not_amplify_constant_gradient_step() -> None:
    parameter = torch.nn.Parameter(torch.ones(4096))
    optimizer = AdamW8bit(
        [parameter],
        lr=1.0e-3,
        betas=(0.0, 0.0),
        weight_decay=0.0,
        quant_block_size=2048,
    )
    parameter.grad = torch.full_like(parameter, 0.25)

    optimizer.step()

    torch.testing.assert_close(parameter, torch.full_like(parameter, 0.999), rtol=0.0, atol=1.0e-6)


def test_adamw8bit_routes_embedding_role_to_fp32_state_without_name_matching() -> None:
    embedding = torch.nn.Parameter(torch.ones(12))
    embedding._areno_optimizer_role = "token_embedding"
    ordinary = torch.nn.Parameter(torch.ones(12))
    optimizer = AdamW8bit(
        [embedding, ordinary],
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=64,
        quant_block_size=4,
    )

    assert [state.precision for state in optimizer._states] == ["fp32", "8bit"]
    embedding.grad = torch.linspace(-1.0, 1.0, 12)
    ordinary.grad = embedding.grad.clone()
    optimizer.step()
    state = optimizer.state_dict()

    assert state["quantizer"] == "dynamic-tree-v1"
    assert [item["precision"] for item in state["precision_policy"]] == ["fp32", "8bit"]
    assert state["precision_policy"][0]["role"] == "token_embedding"
    assert state["state"][0]["exp_avg"] is not None
    assert state["state"][0]["exp_avg_q"] is None
    assert state["state"][1]["exp_avg"] is None
    assert state["state"][1]["exp_avg_q"].dtype == torch.uint8


def test_vocab_parallel_embedding_role_is_safe_for_pretrained_weights_and_dp_sharding() -> None:
    from areno.engine.layers.vocab import VocabParallelEmbedding

    with patch(
        "areno.engine.layers.vocab.get_tp_context",
        return_value=SimpleNamespace(rank=1, world_size=2),
    ):
        embedding = VocabParallelEmbedding(18, 6)
    loaded_weight = torch.linspace(-1.0, 1.0, embedding.weight.numel()).reshape_as(embedding.weight)
    embedding.weight.data.copy_(loaded_weight)
    optimizer = AdamW8bit(
        embedding.parameters(),
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        dp_rank=1,
        dp_size=2,
    )

    assert embedding.weight._areno_optimizer_role == "token_embedding"
    assert optimizer._states[0].precision == "fp32"
    assert optimizer.buckets[0].shard_numel == (embedding.weight.numel() + 1) // 2
    torch.testing.assert_close(embedding.weight, loaded_weight, rtol=0.0, atol=0.0)


def test_adamw8bit_explicit_precision_override_beats_embedding_default_and_deduplicates_ties() -> None:
    tied = torch.nn.Parameter(torch.ones(8))
    tied._areno_optimizer_role = "token_embedding"
    set_optimizer_state_precision(tied, "8bit")
    fp32 = torch.nn.Parameter(torch.ones(4))
    optimizer = AdamW8bit(
        [
            {"params": [tied], "state_precision": "fp32"},
            {"params": [tied, fp32], "state_precision": "fp32"},
        ],
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=64,
    )

    assert optimizer.model_params == [tied, fp32]
    assert [state.precision for state in optimizer._states] == ["8bit", "fp32"]


def test_adamw8bit_mixed_state_checkpoint_round_trip_preserves_next_update() -> None:
    embedding = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 12))
    embedding._areno_optimizer_role = "token_embedding"
    ordinary = torch.nn.Parameter(torch.linspace(0.25, -0.25, 9))
    first = AdamW8bit(
        [embedding, ordinary],
        lr=4.0e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        bucket_numel=64,
        quant_block_size=4,
    )
    embedding.grad = torch.linspace(-0.7, 0.3, embedding.numel())
    ordinary.grad = torch.linspace(0.4, -0.8, ordinary.numel())
    first.step()
    checkpoint = copy.deepcopy(first.state_dict())

    restored_embedding = torch.nn.Parameter(embedding.detach().clone())
    restored_embedding._areno_optimizer_role = "token_embedding"
    restored_ordinary = torch.nn.Parameter(ordinary.detach().clone())
    restored = AdamW8bit(
        [restored_embedding, restored_ordinary],
        lr=4.0e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        bucket_numel=64,
        quant_block_size=4,
    )
    restored.load_state_dict(checkpoint)

    next_embedding_grad = torch.linspace(0.6, -0.2, embedding.numel())
    next_ordinary_grad = torch.linspace(-0.1, 0.9, ordinary.numel())
    embedding.grad = next_embedding_grad.clone()
    restored_embedding.grad = next_embedding_grad.clone()
    ordinary.grad = next_ordinary_grad.clone()
    restored_ordinary.grad = next_ordinary_grad.clone()
    first.step()
    restored.step()

    torch.testing.assert_close(restored_embedding, embedding, rtol=0.0, atol=0.0)
    torch.testing.assert_close(restored_ordinary, ordinary, rtol=0.0, atol=0.0)
    for actual, expected in zip(restored.state_dict()["state"], first.state_dict()["state"], strict=True):
        for key in ("exp_avg_q", "exp_avg_scale", "exp_avg_sq_q", "exp_avg_sq_scale", "exp_avg", "exp_avg_sq"):
            if expected[key] is None:
                assert actual[key] is None
            else:
                torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)


def test_adamw8bit_checkpoint_restores_saved_precision_policy_by_parameter_identity() -> None:
    embedding = torch.nn.Parameter(torch.ones(6))
    embedding._areno_optimizer_role = "token_embedding"
    ordinary = torch.nn.Parameter(torch.ones(6))
    source = AdamW8bit(
        [embedding, ordinary],
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=16,
        quant_block_size=4,
    )
    embedding.grad = torch.ones_like(embedding)
    ordinary.grad = torch.ones_like(ordinary)
    source.step()

    restored_embedding = torch.nn.Parameter(embedding.detach().clone())
    restored_ordinary = torch.nn.Parameter(ordinary.detach().clone())
    restored = AdamW8bit(
        [restored_embedding, restored_ordinary],
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=16,
        quant_block_size=4,
    )
    assert [state.precision for state in restored._states] == ["8bit"]

    restored.load_state_dict(source.state_dict())

    assert [state.precision for state in restored._states] == ["fp32", "8bit"]
    assert restored._parameter_roles[id(restored_embedding)] == "token_embedding"


def test_adamw8bit_reports_mixed_state_storage() -> None:
    embedding = torch.nn.Parameter(torch.ones(12))
    embedding._areno_optimizer_role = "token_embedding"
    ordinary = torch.nn.Parameter(torch.ones(12))
    optimizer = AdamW8bit(
        [embedding, ordinary],
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=64,
        quant_block_size=4,
    )
    embedding.grad = torch.ones_like(embedding)
    ordinary.grad = torch.ones_like(ordinary)
    optimizer.step()

    assert optimizer.state_memory_metrics() == {
        "quantized_state_bytes": 24,
        "fp32_exempt_bytes": 96,
        "block_metadata_bytes": 24,
        "total_bytes": 144,
    }


def test_adamw8bit_nonfinite_gradient_skips_only_affected_block() -> None:
    parameter = torch.nn.Parameter(torch.zeros(8))
    optimizer = AdamW8bit(
        [parameter],
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=16,
        quant_block_size=4,
    )
    gradient = torch.ones_like(parameter)
    gradient[1] = torch.inf
    parameter.grad = gradient
    optimizer.step()

    torch.testing.assert_close(parameter[:4], torch.zeros(4))
    assert torch.all(parameter[4:] < 0)


def test_adamw8bit_disk_offload_supports_mixed_state(tmp_path) -> None:
    embedding = torch.nn.Parameter(torch.ones(8))
    embedding._areno_optimizer_role = "token_embedding"
    ordinary = torch.nn.Parameter(torch.ones(8))
    candidate = AdamW8bit(
        [embedding, ordinary],
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=16,
        quant_block_size=4,
    )
    candidate.configure_state_offload(mode="disk", directory=str(tmp_path), batch_size=2)
    embedding.grad = torch.linspace(-1.0, 1.0, embedding.numel())
    ordinary.grad = torch.linspace(1.0, -1.0, ordinary.numel())
    candidate.step()

    assert all(state.offload_file is not None for state in candidate._states)
    saved = candidate.state_dict()["state"]
    assert saved[0]["exp_avg"] is not None and saved[0]["exp_avg_q"] is None
    assert saved[1]["exp_avg"] is None and saved[1]["exp_avg_q"] is not None
    candidate.onload_state(torch.device("cpu"))
    assert candidate._states[0].exp_avg is not None
    assert candidate._states[1].exp_avg_q is not None
    assert not list(tmp_path.rglob("*.mmap"))
