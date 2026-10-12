"""Hybrid policy ownership, native layouts, and differentiable router contract."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from areno.adapters import LoraConfig
from areno.adapters.lora import initialize_lora
from areno.adapters.peft import export_peft_adapter, load_peft_adapter
from areno.engine.config import EngineConfig, ModelConfig
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.engine.policy_sync import build_policy_plan
from areno.models.qwen3.model import Qwen3ForCausalLM


@pytest.fixture(autouse=True)
def cpu_context():
    previous = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    yield
    set_tp_context(previous)


def make_model():
    config = ModelConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        dtype=torch.float32,
        sequence_parallel=False,
        attn_backend="native",
    )
    model = Qwen3ForCausalLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1)
    return model


@pytest.mark.parametrize("with_lora", (False, True))
def test_hybrid_fused_full_parameter_ownership_sync_and_roundtrip(tmp_path, with_lora):
    model = make_model()
    targets = ("layers.0.mlp.down_proj",) if with_lora else ()
    full = ("layers.0.self_attn.qkv_proj",)
    registry = initialize_lora(model, LoraConfig(target_modules=targets, full_parameter_targets=full), seed=4)
    selected = model.layers[0].self_attn.qkv_proj.weight
    assert selected.requires_grad
    assert len({id(p) for p in registry.parameters()}) == len(registry.parameters())
    assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in registry.parameters()}
    with torch.no_grad():
        selected.add_(0.3)
    worker = SimpleNamespace(model=model, config=SimpleNamespace(model=model.config), adapter_registry=registry)
    plan, metadata = build_policy_plan(worker)
    full_keys = {m.key for m in metadata if ".lora_" not in m.key}
    assert len(full_keys) == 3  # One fused native parameter maps to canonical q/k/v.
    for key in full_keys:
        layout = plan[key].policy_layout()
        chunk = torch.empty(layout.numel, dtype=layout.dtype)
        layout.read_chunk(0, chunk)
        assert torch.allclose(chunk, torch.full_like(chunk, 0.4))
    export_peft_adapter(registry, tmp_path, model=model, model_config=model.config, base_model_name_or_path=None)
    restored = make_model()
    config = LoraConfig(adapter_path=str(tmp_path))
    assert config.full_parameter_targets == full and config.target_modules == targets
    other = initialize_lora(restored, config, seed=4)
    load_peft_adapter(other, tmp_path, model=restored, model_config=restored.config)
    torch.testing.assert_close(restored.layers[0].self_attn.qkv_proj.weight, selected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="frozen reference"):
        with registry.base_only():
            pass
    with pytest.raises(ValueError, match="actor base is trainable"):
        EngineConfig(model=model.config, lora=config, devices=[0], reference_mode="reuse_actor_base")


@pytest.mark.parametrize(
    "selectors,match",
    [
        (("missing",), "not present"),
        (("layers.0.mlp", "layers.0.mlp.down_proj.weight"), "overlap"),
        (("layers.0.self_attn.qkv_proj.weight",), "both in full and LoRA"),
    ],
)
def test_hybrid_rejects_invalid_selection_before_binding(selectors, match):
    model = make_model()
    with pytest.raises(ValueError, match=match):
        initialize_lora(
            model, LoraConfig(target_modules=("layers.0.self_attn.q_proj",), full_parameter_targets=selectors), seed=4
        )
    assert all(p.requires_grad for p in model.parameters())
    assert not any("lora_" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("family", ("bailing", "bailing_v3"))
def test_trainable_router_differentiates_selected_scores_without_changing_routes(monkeypatch, family):
    module = importlib.import_module(f"areno.models.{family}.model")
    config = ModelConfig(
        hidden_size=4, num_experts=4, num_experts_per_tok=2, n_group=1, topk_group=1, moe_router_bias_update_rate=0.0
    )
    gate = module.BailingGate(config, routing_layer_slot=0)
    with torch.no_grad():
        gate.weight.copy_(torch.arange(16).reshape(4, 4) / 16)
    monkeypatch.setattr(module, "_areno_linear_no_compile", F.linear)

    def fused(logits):
        scores = logits.detach().sigmoid()
        weights, indices = scores.topk(2, dim=-1)
        return indices, weights / weights.sum(dim=-1, keepdim=True)

    monkeypatch.setattr(gate, "_forward_grouped_topk", fused)
    x = torch.tensor([[0.2, -0.1, 0.3, 0.4]])
    indices, weights, _ = gate(x)
    expected_indices, expected_weights = fused(F.linear(x, gate.weight))
    assert torch.equal(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights)
    (weights * torch.tensor([[1.0, 2.0]])).sum().backward()
    assert torch.isfinite(gate.weight.grad).all() and gate.weight.grad.abs().sum() > 0
    gate.weight.requires_grad_(False)
    _, frozen_weights, _ = gate(x)
    torch.testing.assert_close(frozen_weights, expected_weights)


def test_fullweight_cli_and_qlora_default_modes():
    from areno.cli.train import _lora_config_from_options

    args = SimpleNamespace(
        lora_rank=None,
        lora_adapter_path=None,
        full_parameter_targets="layers.0.mlp.down_proj",
        lora_alpha=16,
        lora_dropout=0,
        lora_target_modules="q_proj",
        qlora=False,
    )
    config = _lora_config_from_options(args)
    assert config.target_modules == () and config.full_parameter_targets == ("layers.0.mlp.down_proj",)
    args.full_parameter_targets = ""
    args.qlora = True
    assert _lora_config_from_options(args).target_modules == ("q_proj",)
    with pytest.raises(ValueError, match="native checkpoint layout"):
        LoraConfig(qlora=True, full_parameter_targets=("norm",))
