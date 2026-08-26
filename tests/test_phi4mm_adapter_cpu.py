from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

import areno.models
from areno.engine.config import ModelConfig, OptimizerConfig
from areno.engine.layers import mlp, norm, vocab
from areno.engine.modeling import build_optimizer
from areno.engine.parallel.collectives import is_sequence_parallel_active
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.engine.runtime.decode_graph import _validate_decode_cache_length
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.models import registry
from areno.models.phi4mm import Phi4MMAdapter, Phi4MMForCausalLM
from areno.models.phi4mm.model import (
    Phi4MMLongRoPEScaledRotaryEmbedding,
    _phi4mm_longrope_sequence_length,
)


@pytest.fixture(autouse=True)
def _isolate_tp_context():
    previous_context = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    try:
        yield
    finally:
        set_tp_context(previous_context)


def _phi4mm_config() -> dict:
    return {
        "model_type": "phi4mm",
        "vocab_size": 200064,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 32,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "max_position_embeddings": 131072,
        "original_max_position_embeddings": 4096,
        "partial_rotary_factor": 0.75,
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0] * 48,
            "long_factor": [float(index + 1) for index in range(48)],
        },
        "sliding_window": 262144,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "lm_head_bias": False,
        "tie_word_embeddings": True,
        "pad_token_id": 199999,
        "torch_dtype": "bfloat16",
    }


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        model_type="phi4mm",
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        qkv_bias=False,
        qk_norm=False,
        dtype=torch.float32,
        hidden_act="silu",
        partial_rotary_factor=0.75,
        sequence_parallel=False,
        attn_backend="native",
        hf_text_config={
            "original_max_position_embeddings": 32,
            "rope_scaling": {
                "type": "longrope",
                "short_factor": (1.0, 1.0, 1.0),
                "long_factor": (1.0, 2.0, 3.0),
            },
        },
    )


@pytest.fixture
def cpu_reference_kernels(monkeypatch):
    def embedding(input_ids, weight, vocab_start, vocab_end):
        local_ids = input_ids - vocab_start
        local_mask = (input_ids >= vocab_start) & (input_ids < vocab_end)
        safe_ids = local_ids.masked_fill(~local_mask, 0)
        return F.embedding(safe_ids, weight) * local_mask.unsqueeze(-1)

    def rms_norm(x, weight, eps):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
        return normalized.to(dtype=x.dtype) * weight.to(dtype=x.dtype)

    def silu_and_mul(x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up

    monkeypatch.setattr(vocab, "areno_vocab_embedding", embedding)
    monkeypatch.setattr(norm, "_areno_rmsnorm_no_compile", rms_norm)
    monkeypatch.setattr(mlp, "_areno_silu_and_mul_no_compile", silu_and_mul)


def test_phi4mm_config_translation_matches_official_language_backbone():
    config = Phi4MMAdapter().config_from_hf(_phi4mm_config())

    assert config.model_type == "phi4mm"
    assert config.vocab_size == 200064
    assert config.hidden_size == 3072
    assert config.intermediate_size == 8192
    assert config.num_hidden_layers == 32
    assert config.num_attention_heads == 24
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.partial_rotary_factor == 0.75
    assert config.qk_norm is False
    assert config.qkv_bias is False
    assert config.tie_word_embeddings is True
    assert config.dtype == torch.bfloat16
    assert config.hf_text_config is not None
    assert config.hf_text_config["original_max_position_embeddings"] == 4096
    assert config.hf_text_config["rope_scaling"]["short_factor"] == (1.0,) * 48


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"tie_word_embeddings": False}, "tie_word_embeddings=True"),
        ({"attention_bias": True}, "attention_bias=False"),
        ({"hidden_act": "gelu"}, "hidden_act='silu'"),
        ({"rope_scaling": {"type": "linear", "short_factor": [1.0] * 48, "long_factor": [1.0] * 48}}, "longrope"),
        ({"rope_scaling": {"type": "longrope", "short_factor": [1.0] * 47, "long_factor": [1.0] * 48}}, "48 values"),
    ],
)
def test_phi4mm_config_rejects_unsupported_language_semantics(update, message):
    hf_config = _phi4mm_config()
    hf_config.update(update)

    with pytest.raises(ValueError, match=message):
        Phi4MMAdapter().config_from_hf(hf_config)


def test_phi4mm_registry_resolves_config(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(_phi4mm_config()), encoding="utf-8")
    monkeypatch.setattr(registry, "_PLUGINS_LOADED", False)
    monkeypatch.setattr(areno.models, "_REGISTERED_GROUPS", set())
    monkeypatch.setattr(registry, "_ADAPTERS", {})

    config = registry.config_from_hf(tmp_path)

    assert config.model_type == "phi4mm"
    assert isinstance(registry.adapter_from_hf(tmp_path), Phi4MMAdapter)


def test_phi4mm_tp_validation_rejects_non_divisible_kv_heads():
    config = Phi4MMAdapter().config_from_hf(_phi4mm_config())

    config.validate_tp(1)
    config.validate_tp(2)
    config.validate_tp(4)
    config.validate_tp(8)
    with pytest.raises(ValueError, match="num_key_value_heads must be divisible"):
        config.validate_tp(3)
    with pytest.raises(ValueError, match="num_key_value_heads must be divisible"):
        config.validate_tp(6)


def test_phi4mm_model_construction_has_expected_text_layers():
    config = _tiny_model_config()
    model = Phi4MMAdapter().build(config)

    assert isinstance(model, Phi4MMForCausalLM)
    assert len(model.model.layers) == 2
    assert model.model.embed_tokens.weight.shape == (32, 64)
    assert model.lm_head.weight.shape == (32, 64)
    assert model.model.norm.eps == 1e-5
    for layer in model.model.layers:
        assert layer.input_layernorm.eps == 1e-5
        assert layer.post_attention_layernorm.eps == 1e-5
        assert layer.self_attn.qkv_proj.out_features == (64, 32, 32)
        assert layer.self_attn.qkv_proj.local_out_features == [64, 32, 32]
        assert layer.self_attn.o_proj.weight.shape == (64, 64)
        assert layer.mlp.gate_up_proj.out_features == (128, 128)
        assert layer.mlp.gate_up_proj.weight.shape == (256, 64)
        assert layer.mlp.down_proj.weight.shape == (64, 128)


def test_phi4mm_projection_biases_and_qk_norm_are_disabled():
    model = Phi4MMAdapter().build(_tiny_model_config())

    assert not hasattr(model.lm_head, "bias")
    for layer in model.model.layers:
        assert layer.self_attn.qkv_proj.bias is None
        assert layer.self_attn.o_proj.bias is None
        assert layer.self_attn.q_norm is None
        assert layer.self_attn.k_norm is None
        assert layer.mlp.gate_up_proj.bias is None
        assert layer.mlp.down_proj.bias is None


def test_phi4mm_embedding_and_lm_head_share_one_optimizer_parameter():
    model = Phi4MMAdapter().build(_tiny_model_config())

    assert model.lm_head.weight is model.model.embed_tokens.weight
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    assert len(parameter_ids) == len(set(parameter_ids))

    optimizer = build_optimizer(
        model.parameters(),
        OptimizerConfig(),
        type("Context", (), {"dp_rank": 0, "dp_size": 1, "dp_group": None})(),
    )
    optimizer_parameter_ids = [id(parameter) for parameter in optimizer.model_params]
    assert len(optimizer_parameter_ids) == len(set(optimizer_parameter_ids))
    assert optimizer_parameter_ids.count(id(model.model.embed_tokens.weight)) == 1


def test_phi4mm_text_forward_shapes_and_causal_prefix(cpu_reference_kernels):
    del cpu_reference_kernels
    torch.manual_seed(0)
    model = Phi4MMAdapter().build(_tiny_model_config()).eval()
    input_ids = torch.tensor([[1, 2, 3], [1, 2, 4]])

    output = model(input_ids)

    assert output.hidden_states is not None
    assert output.logits_shard is not None
    assert output.hidden_states.shape == (2, 3, 64)
    assert output.logits_shard.shape == (2, 3, 32)
    assert torch.isfinite(output.hidden_states).all()
    assert torch.isfinite(output.logits_shard).all()
    torch.testing.assert_close(output.logits_shard[0, :2], output.logits_shard[1, :2])


def test_phi4mm_lm_head_runs_inside_sequence_parallel_region(cpu_reference_kernels, monkeypatch):
    del cpu_reference_kernels
    model = Phi4MMAdapter().build(_tiny_model_config()).eval()
    original_forward = model.lm_head.forward
    sequence_parallel_states = []

    def record_sequence_parallel_state(hidden_states):
        sequence_parallel_states.append(is_sequence_parallel_active())
        return original_forward(hidden_states)

    monkeypatch.setattr(model.lm_head, "forward", record_sequence_parallel_state)
    model(torch.tensor([[1, 2, 3]]), train_meta=TrainMeta(sequence_parallel=True))

    assert sequence_parallel_states == [True]


def test_phi4mm_kv_cache_lifecycle():
    model = Phi4MMAdapter().build(_tiny_model_config())
    caches = model.allocate_kv_caches(num_blocks=3, block_size=4, device=torch.device("cpu"))

    assert len(caches) == len(model.layers)
    assert caches[0][0].shape == (3, 4, 4, 8)
    assert caches[0][0].dtype == model.config.dtype

    model.set_kv_caches(caches)
    assert model.layers[0].self_attn.k_cache is caches[0][0]
    assert model.onload_kv_caches(torch.device("cpu"))

    model.offload_kv_caches()
    assert model.layers[0].self_attn.infer_backend is None
    model.clear_kv_caches()
    assert model.layers[0].self_attn.k_cache.numel() == 0
    assert not model.onload_kv_caches(torch.device("cpu"))

    with pytest.raises(ValueError, match="expected 2 layer caches"):
        model.set_kv_caches(caches[:1])


def _official_longrope_reference(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    dim: int,
    base: float,
    factors: tuple[float, ...],
    max_position_embeddings: int,
    original_max_position_embeddings: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ext_factors = torch.tensor(factors, dtype=torch.float32, device=x.device)
    inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=x.device).float() / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    embedding = torch.cat((freqs, freqs), dim=-1)
    scale = max_position_embeddings / original_max_position_embeddings
    scaling_factor = (
        1.0 if scale <= 1.0 else math.sqrt(1.0 + math.log(scale) / math.log(original_max_position_embeddings))
    )
    return (embedding.cos() * scaling_factor).to(x.dtype), (embedding.sin() * scaling_factor).to(x.dtype)


def test_phi4mm_official_config_builds_partial_longrope_without_position_caches():
    config = Phi4MMAdapter().config_from_hf(_phi4mm_config())

    rope = Phi4MMLongRoPEScaledRotaryEmbedding(config)

    assert rope.dim == 96
    assert config.head_dim - rope.dim == 32
    assert rope.short_inv_freq.shape == (48,)
    assert rope.long_inv_freq.shape == (48,)
    assert all("cached" not in name for name, _ in rope.named_buffers())


def test_phi4mm_longrope_keeps_inverse_frequencies_in_fp32_when_model_is_cast():
    rope = Phi4MMLongRoPEScaledRotaryEmbedding(_tiny_model_config()).to(dtype=torch.bfloat16)

    assert rope.short_inv_freq.dtype == torch.float32
    assert rope.long_inv_freq.dtype == torch.float32


@pytest.mark.parametrize(
    ("sequence_length", "positions", "factor_key"),
    [
        (32, [0, 1, 7, 31], "short_factor"),
        (64, [0, 1, 31, 32, 63], "long_factor"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_phi4mm_longrope_cos_sin_matches_official_reference(sequence_length, positions, factor_key, dtype):
    config = _tiny_model_config()
    rope = Phi4MMLongRoPEScaledRotaryEmbedding(config)
    x = torch.zeros(1, len(positions), 1, config.head_dim, dtype=dtype)
    position_ids = torch.tensor([positions])

    actual_cos, actual_sin = rope.cos_sin(x, position_ids, sequence_length)
    expected_cos, expected_sin = _official_longrope_reference(
        x,
        position_ids,
        rope.dim,
        config.rope_theta,
        config.hf_text_config["rope_scaling"][factor_key],
        config.max_position_embeddings,
        config.hf_text_config["original_max_position_embeddings"],
    )

    torch.testing.assert_close(actual_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(actual_sin, expected_sin, rtol=0, atol=0)


def test_phi4mm_longrope_preserves_non_rotary_head_dimensions_and_applies_scale():
    config = _tiny_model_config()
    rope = Phi4MMLongRoPEScaledRotaryEmbedding(config)
    q = torch.randn(1, 3, 2, config.head_dim)
    k = torch.randn(1, 3, 1, config.head_dim)
    position_ids = torch.tensor([[0, 1, 2]])

    rotated_q, rotated_k = rope(q, k, position_ids, sequence_length=3)
    cos, sin = rope.cos_sin(q, torch.tensor([[0]]), sequence_length=3)

    torch.testing.assert_close(rotated_q[..., rope.dim :], q[..., rope.dim :])
    torch.testing.assert_close(rotated_k[..., rope.dim :], k[..., rope.dim :])
    assert cos[0, 0, 0].item() == pytest.approx(rope.scaling_factor)
    assert sin[0, 0, 0].item() == 0.0


def test_phi4mm_longrope_full_long_prefill_selects_long_factors():
    attention = Phi4MMAdapter().build(_tiny_model_config()).model.layers[0].self_attn
    positions = torch.arange(40).unsqueeze(0)
    infer_meta = InferMeta(mode="prefill", cu_seqlens=torch.tensor([0, 40], dtype=torch.int32), max_seqlen=40)
    q = torch.randn(1, 40, attention.local_heads, attention.head_dim)
    k = torch.randn(1, 40, attention.local_kv_heads, attention.head_dim)

    sequence_length = _phi4mm_longrope_sequence_length(positions, None, infer_meta, 32)
    actual_q, actual_k = attention.apply_rotary(q, k, positions, None, infer_meta)
    expected_q, expected_k = attention.rope(q, k, positions, sequence_length=40)

    assert sequence_length == 40
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)


def test_phi4mm_longrope_rejects_chunked_prefill_crossing_boundary():
    positions = torch.arange(28, 40).unsqueeze(0)
    infer_meta = InferMeta(mode="prefill", cu_seqlens=torch.tensor([0, 12], dtype=torch.int32), max_seqlen=12)

    with pytest.raises(ValueError, match="chunked prefill cannot cross"):
        _phi4mm_longrope_sequence_length(positions, None, infer_meta, 32)


def test_phi4mm_longrope_rejects_cached_decode_crossing_boundary():
    below_boundary = InferMeta(mode="decode", cache_seqlens=torch.tensor([31], dtype=torch.int32))
    crossing_boundary = InferMeta(mode="decode", cache_seqlens=torch.tensor([32], dtype=torch.int32))

    assert _phi4mm_longrope_sequence_length(torch.tensor([[31]]), None, below_boundary, 32) == 32
    with pytest.raises(ValueError, match="cached decode cannot cross"):
        _phi4mm_longrope_sequence_length(torch.tensor([[32]]), None, crossing_boundary, 32)


def test_phi4mm_longrope_decode_graph_replay_enforces_same_boundary():
    _validate_decode_cache_length(torch.tensor([31, 99], dtype=torch.int32), actual=1, limit=32)
    with pytest.raises(ValueError, match="rotary-factor boundary"):
        _validate_decode_cache_length(torch.tensor([32], dtype=torch.int32), actual=1, limit=32)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_phi4mm_tp_construction_uses_compatible_local_shards(tp_size):
    old_context = get_tp_context()
    try:
        set_tp_context(TPContext(rank=0, world_size=tp_size, device=torch.device("cpu"), group=None))
        config = _tiny_model_config()
        config.validate_tp(tp_size)
        model = Phi4MMAdapter().build(config)
    finally:
        set_tp_context(old_context)

    attention = model.model.layers[0].self_attn
    assert attention.local_heads == 8 // tp_size
    assert attention.local_kv_heads == 4 // tp_size
    assert attention.qkv_proj.local_out_features == [64 // tp_size, 32 // tp_size, 32 // tp_size]
    assert attention.o_proj.weight.shape == (64, 64 // tp_size)
    assert model.model.layers[0].mlp.gate_up_proj.weight.shape == (256 // tp_size, 64)
    assert model.model.layers[0].mlp.down_proj.weight.shape == (64, 128 // tp_size)
    assert model.model.embed_tokens.weight.shape == (32 // tp_size, 64)
    assert model.lm_head.weight.shape == (32 // tp_size, 64)
    assert model.lm_head.weight is model.model.embed_tokens.weight
