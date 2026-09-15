"""Focused native-LoRA target-contract checks across model adapters."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from areno.adapters import LoraConfig
from areno.adapters.lora import initialize_lora
from areno.engine.config import ModelConfig
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.models.olmo2 import Olmo2ForCausalLM
from areno.models.phi4mm import Phi4MMForCausalLM
from areno.models.gemma4.model import Gemma4MLP, Gemma4MoeExperts
from areno.models.bailing.model import BailingDenseMLP, BailingGroupedExperts, BailingSoftmaxAttention
from areno.models.minicpmv46.model import MiniCPMV46ForCausalLM
from areno.models.qwen3_5.model import Qwen35ForCausalLM


@pytest.fixture(autouse=True)
def _cpu_tp_context():
    previous = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    try:
        yield
    finally:
        set_tp_context(previous)


def _dense_config(model_type: str) -> ModelConfig:
    return ModelConfig(
        model_type=model_type,
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        dtype=torch.float32,
        attn_backend="native",
        sequence_parallel=False,
        tie_word_embeddings=model_type == "phi4mm",
        hf_text_config=(
            {
                "original_max_position_embeddings": 32,
                "rope_scaling": {
                    "type": "longrope",
                    "short_factor": (1.0, 1.0, 1.0, 1.0),
                    "long_factor": (1.0, 2.0, 3.0, 4.0),
                },
            }
            if model_type == "phi4mm"
            else None
        ),
    )


@pytest.mark.parametrize(
    ("model", "targets"),
    (
        (
            lambda: Olmo2ForCausalLM(_dense_config("olmo2")),
            ("layers.0.self_attn.q_proj", "layers.0.mlp.down_proj"),
        ),
        (
            lambda: Phi4MMForCausalLM(_dense_config("phi4mm")),
            ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"),
        ),
    ),
)
def test_dense_model_exact_lora_targets_resolve(model, targets: tuple[str, ...]) -> None:
    policy = model()
    registry = initialize_lora(
        policy,
        LoraConfig(rank=2, alpha=2.0, target_modules=targets),
        seed=7,
    )

    assert tuple(registry.slots) == targets
    assert all(parameter.requires_grad for parameter in registry.parameters())
    trainable_ids = {id(parameter) for parameter in registry.parameters()}
    assert all(parameter.requires_grad == (id(parameter) in trainable_ids) for parameter in policy.parameters())


class _GemmaPolicy(nn.Module):
    def __init__(self, *, moe: bool) -> None:
        super().__init__()
        self.config = _dense_config("gemma4")
        self.config.num_experts = 2
        self.config.moe_intermediate_size = 16
        self.layers = nn.ModuleList([nn.Module()])
        if moe:
            self.layers[0].experts = Gemma4MoeExperts(self.config)
        else:
            self.layers[0].mlp = Gemma4MLP(self.config, use_double_wide_mlp=False)


@pytest.mark.parametrize(
    ("policy_factory", "targets", "logical_targets"),
    (
        (
            lambda: _GemmaPolicy(moe=False),
            ("layers.0.mlp.gate_proj", "layers.0.mlp.down_proj"),
            ("layers.0.mlp.gate_proj", "layers.0.mlp.down_proj"),
        ),
        (
            lambda: _GemmaPolicy(moe=True),
            ("layers.0.experts.gate_proj", "layers.0.experts.down_proj"),
            ("layers.0.experts.{expert}.gate_proj", "layers.0.experts.{expert}.down_proj"),
        ),
    ),
)
def test_gemma4_exact_lora_targets_resolve(policy_factory, targets, logical_targets) -> None:
    policy = policy_factory()
    registry = initialize_lora(policy, LoraConfig(rank=2, alpha=2.0, target_modules=targets), seed=7)

    assert tuple(registry.slots) == logical_targets


def test_qwen35_full_and_linear_attention_exact_lora_targets_resolve() -> None:
    config = _dense_config("qwen3_5")
    config.num_hidden_layers = 2
    config.layer_types = ("linear_attention", "full_attention")
    config.attn_output_gate = True
    config.linear_conv_kernel_dim = 4
    config.linear_key_head_dim = 8
    config.linear_value_head_dim = 8
    config.linear_num_key_heads = 4
    config.linear_num_value_heads = 4
    policy = Qwen35ForCausalLM(config)
    targets = (
        "layers.0.attention.in_proj_q",
        "layers.0.attention.in_proj_z",
        "layers.0.attention.out_proj",
        "layers.1.attention.q_proj",
        "layers.1.attention.k_proj",
        "layers.1.mlp.down_proj",
    )

    registry = initialize_lora(policy, LoraConfig(rank=2, alpha=2.0, target_modules=targets), seed=7)

    assert tuple(registry.slots) == targets


def test_minicpmv46_language_exact_lora_targets_resolve() -> None:
    config = _dense_config("minicpmv46")
    config.num_hidden_layers = 2
    config.layer_types = ("linear_attention", "full_attention")
    config.linear_conv_kernel_dim = 4
    config.linear_key_head_dim = 8
    config.linear_value_head_dim = 8
    config.linear_num_key_heads = 4
    config.linear_num_value_heads = 4
    policy = MiniCPMV46ForCausalLM(config)
    targets = (
        "layers.0.attention.in_proj_q",
        "layers.0.attention.in_proj_a",
        "layers.0.attention.out_proj",
        "layers.1.attention.q_proj",
        "layers.1.attention.q_gate_proj",
        "layers.1.attention.o_proj",
        "layers.1.mlp.gate_proj",
    )

    registry = initialize_lora(policy, LoraConfig(rank=2, alpha=2.0, target_modules=targets), seed=7)

    assert tuple(registry.slots) == targets


class _BailingPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _dense_config("bailing_moe_linear_v2")
        self.config.num_experts = 2
        self.config.moe_intermediate_size = 16
        self.config.kv_lora_rank = 8
        self.config.qk_nope_head_dim = 8
        self.config.qk_rope_head_dim = 8
        self.config.v_head_dim = 8
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].attention = BailingSoftmaxAttention(self.config, 0)
        self.layers[0].mlp = BailingDenseMLP(self.config, 64)
        self.layers[0].experts = BailingGroupedExperts(self.config)


def test_legacy_bailing_exact_lora_targets_resolve() -> None:
    policy = _BailingPolicy()
    targets = (
        "layers.0.attention.q_proj",
        "layers.0.attention.kv_a_proj_with_mqa",
        "layers.0.attention.kv_b_proj",
        "layers.0.attention.dense",
        "layers.0.mlp.gate_proj",
        "layers.0.experts.linear_fc1",
        "layers.0.experts.linear_fc2",
    )

    registry = initialize_lora(policy, LoraConfig(rank=2, alpha=2.0, target_modules=targets), seed=7)

    assert tuple(registry.slots) == (
        *targets[:-2],
        "layers.0.experts.{expert}.linear_fc1",
        "layers.0.experts.{expert}.linear_fc2",
    )
