"""Focused native-LoRA target-contract checks across model adapters."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from areno.adapters import LoraConfig
from areno.adapters.lora import initialize_lora
from areno.engine.config import ModelConfig
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.models.bailing.model import (
    BailingDenseMLP,
    BailingGroupedExperts,
    BailingSoftmaxAttention,
)
from areno.models.gemma4.model import Gemma4MLP, Gemma4MoeExperts, Gemma4MoeMLP
from areno.models.minicpmv46.model import MiniCPMV46ForCausalLM
from areno.models.olmo2 import Olmo2ForCausalLM
from areno.models.phi4mm import Phi4MMForCausalLM
from areno.models.qwen3.model import Qwen3ForCausalLM, Qwen3MoeMLP
from areno.models.qwen3_5.model import Qwen35ForCausalLM, Qwen35MoeMLP


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
        qk_norm=model_type != "phi4mm",
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
            lambda: Qwen3ForCausalLM(_dense_config("llama")),
            ("layers.0.self_attn.q_proj", "layers.0.mlp.down_proj"),
        ),
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


@pytest.mark.parametrize(
    "model_type,mlp_factory", (("qwen3_moe", Qwen3MoeMLP), ("qwen3_5_moe", Qwen35MoeMLP), ("gemma4", Gemma4MoeMLP))
)
def test_expert_infer_merge_preserves_base_and_refreshes(model_type, mlp_factory):
    config = _dense_config(model_type)
    config.num_experts = 2
    config.num_experts_per_tok = 1
    config.moe_intermediate_size = 16
    policy = nn.Module()
    policy.config = config
    policy.mlp = mlp_factory(config, routing_layer_slot=0)
    with torch.no_grad():
        policy.mlp.experts.gate_up_weight.normal_(std=0.1)
        policy.mlp.experts.down_weight.normal_(std=0.1)
    base1 = policy.mlp.experts.gate_up_weight.detach().clone()
    base2 = policy.mlp.experts.down_weight.detach().clone()
    registry = initialize_lora(
        policy, LoraConfig(rank=2, alpha=2.0, target_modules=("mlp.experts.gate_proj", "mlp.experts.down_proj")), seed=7
    )
    with torch.no_grad():
        for slot in registry.slots.values():
            slot.lora_B.fill_(0.1)
    policy.mlp.prepare_infer_weights()
    merged = policy.mlp._infer_w1_weight.clone()
    assert not torch.equal(merged, base1)
    assert policy.mlp._infer_w1_weight.data_ptr() != policy.mlp.experts.gate_up_weight.data_ptr()
    policy.mlp.prepare_infer_weights()
    torch.testing.assert_close(policy.mlp._infer_w1_weight, merged, rtol=0, atol=0)
    torch.testing.assert_close(policy.mlp.experts.gate_up_weight, base1, rtol=0, atol=0)
    torch.testing.assert_close(policy.mlp.experts.down_weight, base2, rtol=0, atol=0)
    with registry.base_only():
        policy.mlp.prepare_infer_weights()
        torch.testing.assert_close(policy.mlp._infer_w1_weight, base1, rtol=0, atol=0)
        torch.testing.assert_close(policy.mlp._infer_w2_weight, base2, rtol=0, atol=0)
    policy.mlp.prepare_infer_weights()
    torch.testing.assert_close(policy.mlp._infer_w1_weight, merged, rtol=0, atol=0)


def test_existing_fft_policy_state_survives_lora_optimizer_sync_and_reload(tmp_path):
    from areno.adapters.peft import export_peft_adapter, load_peft_adapter
    from areno.engine.layers.linear import ColumnParallelLinear
    from areno.engine.policy_sync import build_adapter_policy_plan
    from areno.models.bailing_v3.model import BailingGate

    def make_policy():
        policy = nn.Module()
        policy.config = _dense_config("bailing_moe_v3")
        policy.config.no_kda_lora = True
        policy.config.num_experts = 2
        policy.config.num_experts_per_tok = 1
        policy.config.moe_router_bias_update_rate = 0.01
        policy.projection = ColumnParallelLinear(32, 32)
        policy.gate = BailingGate(policy.config, routing_layer_slot=0)
        policy.media = nn.Linear(32, 32, bias=False)
        policy.media.weight._areno_policy_sync = True
        return policy

    policy = make_policy()
    original_media = policy.media.weight.detach().clone()
    registry = initialize_lora(policy, LoraConfig(rank=2, alpha=2, target_modules=("projection",)), seed=7)
    assert policy.media.weight.requires_grad
    assert not policy.projection.weight.requires_grad
    assert id(policy.media.weight) in {id(parameter) for parameter in registry.parameters()}
    optimizer = torch.optim.SGD(registry.parameters(), lr=0.1)
    optimizer.zero_grad()
    policy.media(torch.ones(1, 32)).sum().backward()
    optimizer.step()
    policy.gate.local_tokens_per_expert.copy_(torch.tensor([10.0, 0.0]))
    policy.gate.finalize_expert_bias(None, None)
    assert torch.count_nonzero(policy.gate.expert_bias) == 2
    updated_media = policy.media.weight.detach().clone()
    updated_bias = policy.gate.expert_bias.clone()
    plan = build_adapter_policy_plan(registry)
    assert "areno_policy_state.media.weight" in plan
    assert "areno_policy_state.gate.expert_bias" in plan
    assert not any("tokens_per_expert" in name for name in plan)
    with registry.base_only():
        torch.testing.assert_close(policy.media.weight, original_media, rtol=0, atol=0)
        assert not policy.gate.expert_bias.any()
        assert not policy.gate.local_expert_bias.any()
        with registry.base_only():
            assert not policy.gate.expert_bias.any()
    torch.testing.assert_close(policy.media.weight, updated_media, rtol=0, atol=0)
    torch.testing.assert_close(policy.gate.expert_bias, updated_bias, rtol=0, atol=0)
    export_peft_adapter(registry, tmp_path, base_model_name_or_path=None)
    reloaded = make_policy()
    loaded_config = LoraConfig(adapter_path=str(tmp_path))
    loaded_registry = initialize_lora(reloaded, loaded_config, seed=7)
    load_peft_adapter(loaded_registry, tmp_path)
    for name, tensor in registry.policy_state.named_tensors():
        torch.testing.assert_close(dict(loaded_registry.policy_state.named_tensors())[name], tensor, rtol=0, atol=0)
    # A model dtype/device transition replaces buffers; the registry must follow
    # the live objects instead of publishing a stale tensor reference.
    reloaded.to(torch.float64)
    assert dict(loaded_registry.policy_state.named_tensors())["gate.expert_bias"].dtype == torch.float64


@pytest.mark.parametrize("family", ("qwen3", "gemma4", "bailing", "bailing_v3"))
def test_empty_expert_routes_preserve_input_and_adapter_gradients(monkeypatch, family):
    import importlib

    module = importlib.import_module(f"areno.models.{family}.model")
    config = _dense_config(
        {"qwen3": "qwen3_moe", "gemma4": "gemma4", "bailing": "bailing_moe_linear_v2", "bailing_v3": "bailing_moe_v3"}[
            family
        ]
    )
    config.num_experts = 2
    config.moe_intermediate_size = 16
    # Follow the working FFT module layout, independently of the checkpoint flag.
    config.no_kda_lora = False
    factory = getattr(
        module,
        {
            "qwen3": "Qwen3MoeExperts",
            "gemma4": "Gemma4MoeExperts",
            "bailing": "BailingGroupedExperts",
            "bailing_v3": "BailingGroupedExperts",
        }[family],
    )
    policy = nn.Module()
    policy.config = config
    policy.experts = factory(config)
    registry = initialize_lora(
        policy, LoraConfig(rank=2, alpha=2, target_modules=("experts.gate_proj", "experts.down_proj")), seed=7
    )

    def empty_routes(flat, *args):
        return (
            flat.new_empty((0, 32)),
            flat.new_empty(0),
            torch.empty(0, dtype=torch.long),
            torch.zeros(2, dtype=torch.int32),
        )

    monkeypatch.setattr(module, "_areno_moe_topk_permute_no_compile", empty_routes)
    flat = torch.randn(4, 32, requires_grad=True)
    out = policy.experts(flat, torch.zeros(4, 1, dtype=torch.long), torch.ones(4, 1))
    out.sum().backward()
    assert flat.grad is not None
    assert torch.equal(flat.grad, torch.zeros_like(flat))
    assert all(parameter.grad is not None and not parameter.grad.any() for parameter in registry.parameters())


def test_replicated_projection_follows_fft_gradient_ownership():
    from areno.models.gemma4.model import Gemma4ReplicatedLinear

    set_tp_context(TPContext(rank=0, world_size=2, device=torch.device("cpu"), group=None))
    policy = nn.Module()
    policy.config = _dense_config("gemma4")
    policy.ple = Gemma4ReplicatedLinear(32, 8, sequence_parallel=False)
    registry = initialize_lora(policy, LoraConfig(rank=2, alpha=2, target_modules=("ple",)), seed=7)
    slot = registry.slots["ple"]
    for parameter in (slot.lora_A, slot.lora_B):
        assert parameter.tensor_model_parallel == policy.ple.weight.tensor_model_parallel
        assert parameter.sequence_parallel == policy.ple.weight.sequence_parallel
        assert parameter.tp_grad_allreduce == policy.ple.weight.tp_grad_allreduce
        assert not hasattr(parameter, "tp_replicated_output_range")


def test_standard_peft_adapter_initializes_existing_fft_router_updates(tmp_path):
    from areno.adapters.peft import export_peft_adapter, load_peft_adapter
    from areno.engine.layers.linear import ColumnParallelLinear
    from areno.models.bailing_v3.model import BailingGate

    def make_policy(rate):
        policy = nn.Module()
        policy.config = _dense_config("bailing_moe_v3")
        policy.config.num_experts = 2
        policy.config.num_experts_per_tok = 1
        policy.config.moe_router_bias_update_rate = rate
        policy.projection = ColumnParallelLinear(32, 32)
        policy.gate = BailingGate(policy.config, routing_layer_slot=0)
        return policy

    source = initialize_lora(make_policy(0), LoraConfig(rank=2, target_modules=("projection",)), seed=7)
    with torch.no_grad():
        source.slots["projection"].lora_B.fill_(0.1)
    export_peft_adapter(source, tmp_path, base_model_name_or_path=None)
    policy = make_policy(0.01)
    target = initialize_lora(policy, LoraConfig(adapter_path=str(tmp_path)), seed=7)
    assert target.policy_state.active
    load_peft_adapter(target, tmp_path)
    torch.testing.assert_close(target.slots["projection"].lora_B, source.slots["projection"].lora_B)
    assert not policy.gate.expert_bias.any()


@pytest.mark.parametrize("policy_metadata", [None, {"version": 2}, {"version": 1, "buffers": ["projection.weight"]}])
def test_policy_artifact_rejects_missing_unsupported_or_foreign_state(tmp_path, policy_metadata):
    import json

    from areno.engine.layers.linear import ColumnParallelLinear

    metadata = {
        "peft_type": "areno_lora_policy",
        "r": 2,
        "lora_alpha": 2,
        "target_modules": ["projection"],
    }
    if policy_metadata is not None:
        metadata["areno_policy_state"] = policy_metadata
    (tmp_path / "adapter_config.json").write_text(json.dumps(metadata))
    policy = nn.Module()
    policy.config = _dense_config("qwen3")
    policy.projection = ColumnParallelLinear(32, 32)
    with pytest.raises(ValueError, match="policy-state|routing state"):
        initialize_lora(policy, LoraConfig(adapter_path=str(tmp_path)), seed=7)
    assert policy.projection.weight.requires_grad
    assert policy.projection.lora_slot is None
