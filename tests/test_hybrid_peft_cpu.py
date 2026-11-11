"""Hybrid PEFT training: LoRA slots plus a full-parameter subset (issue #555)."""

import json
from types import SimpleNamespace

import click
import pytest
import torch
from safetensors.torch import load_file

from areno.adapters import LoraConfig
from areno.adapters.lora import initialize_lora
from areno.adapters.peft import export_peft_adapter, load_peft_adapter
from areno.adapters.qlora import initialize_qlora
from areno.cli.train import _lora_config_from_options
from areno.engine.config import EngineConfig, ModelConfig
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.engine.policy_sync import build_policy_plan
from areno.models.qwen3.model import Qwen3ForCausalLM


@pytest.fixture(autouse=True)
def single_rank():
    old = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    yield
    set_tp_context(old)


def build_model():
    config = ModelConfig(
        model_type="qwen3",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        dtype=torch.float32,
        attn_backend="native",
    )
    model = Qwen3ForCausalLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    return model


def test_hybrid_selects_and_unfreezes_only_matching_params_cpu():
    model = build_model()
    config = LoraConfig(rank=2, alpha=4, full_param_modules=("norm",))
    registry = initialize_lora(model, config, seed=3)
    expected_full = {
        "norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
        "layers.1.input_layernorm.weight",
        "layers.1.post_attention_layernorm.weight",
        "layers.1.self_attn.q_norm.weight",
        "layers.1.self_attn.k_norm.weight",
    }
    assert set(registry.full_params) == expected_full
    assert all(parameter.requires_grad for parameter in registry.full_parameters())
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    wanted = {id(parameter) for parameter in registry.parameters()} | {
        id(parameter) for parameter in registry.full_parameters()
    }
    assert trainable == wanted


def test_hybrid_overlap_with_lora_targets_rejected_cpu():
    model = build_model()
    config = LoraConfig(rank=2, alpha=4, full_param_modules=("q_proj",))
    with pytest.raises(ValueError, match="overlap"):
        initialize_lora(model, config, seed=3)
    config = LoraConfig(rank=2, alpha=4, full_param_modules=("lora_A",))
    with pytest.raises(ValueError, match="overlap"):
        initialize_lora(model, config, seed=3)


def test_hybrid_unmatched_pattern_rejected_cpu():
    model = build_model()
    config = LoraConfig(rank=2, alpha=4, full_param_modules=("norm", "no_such_module"))
    with pytest.raises(ValueError, match="matched no parameters: no_such_module"):
        initialize_lora(model, config, seed=3)


def test_lora_config_rejects_empty_full_param_pattern_cpu():
    with pytest.raises(ValueError, match="non-empty"):
        LoraConfig(rank=2, alpha=4, full_param_modules=("norm", ""))


def test_hybrid_gradients_flow_to_slots_and_full_params_cpu():
    # Model forwards need CUDA/NPU kernels on this runtime, so exercise the
    # exact two update terms of a hybrid projection: base(x) + slot(x).
    model = build_model()
    config = LoraConfig(
        rank=2, alpha=4, target_modules=("q_proj", "k_proj", "v_proj"), full_param_modules=("o_proj",)
    )
    registry = initialize_lora(model, config, seed=3)
    x = torch.randn(2, 5, 32)
    slot = registry.slots["layers.0.self_attn.q_proj"]
    base = model.layers[0].self_attn.o_proj.weight
    with torch.no_grad():
        slot.lora_B.normal_(0, 0.03)
    (torch.nn.functional.linear(x, base) + slot(x)).square().sum().backward()
    assert base.grad is not None and base.grad.abs().sum() > 0
    assert slot.lora_A.grad is not None and slot.lora_A.grad.abs().sum() > 0
    assert slot.lora_B.grad is not None and slot.lora_B.grad.abs().sum() > 0
    frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
    assert frozen and all(parameter.grad is None for parameter in frozen)


def test_hybrid_policy_plan_carries_slots_and_full_params_cpu():
    model = build_model()
    config = LoraConfig(rank=2, alpha=4, full_param_modules=("norm",))
    registry = initialize_lora(model, config, seed=3)
    worker = SimpleNamespace(adapter_registry=registry, model=model, config=SimpleNamespace(model=model.config))
    plan, metadata = build_policy_plan(worker)
    assert set(registry.full_params) <= set(plan)
    for logical_name in registry.slots:
        assert f"{logical_name}.lora_A.weight" in plan
        assert f"{logical_name}.lora_B.weight" in plan
    assert len(metadata) == 2 * len(registry.slots) + len(registry.full_params)


def test_hybrid_plan_entries_write_back_into_live_params_cpu():
    model = build_model()
    config = LoraConfig(rank=2, alpha=4, full_param_modules=("lm_head",))
    registry = initialize_lora(model, config, seed=3)
    worker = SimpleNamespace(adapter_registry=registry, model=model, config=SimpleNamespace(model=model.config))
    plan, _ = build_policy_plan(worker)
    parameter = registry.full_params["lm_head.weight"]
    layout = plan["lm_head.weight"].policy_layout()
    assert (
        layout.pieces[0].tensor.untyped_storage().data_ptr() == parameter.untyped_storage().data_ptr()
    )
    flat = parameter.reshape(-1).clone()
    buffer = torch.empty_like(flat)
    layout.read_chunk(0, buffer)
    torch.testing.assert_close(buffer, flat, atol=0, rtol=0)
    layout.write_chunk(0, torch.zeros_like(flat))
    assert parameter.abs().sum() == 0


def test_hybrid_qlora_keeps_full_params_unquantized_cpu():
    torch.manual_seed(9)
    model = build_model()
    config = LoraConfig(
        rank=2,
        alpha=4,
        target_modules=("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"),
        qlora=True,
        full_param_modules=("o_proj",),
    )
    registry = initialize_lora(model, config, seed=3)
    EngineConfig(model=model.config, lora=config, devices=[0])
    initialize_qlora(model)
    for layer in model.layers:
        assert layer.self_attn.o_proj.weight is not None and layer.self_attn.o_proj.weight.requires_grad
        assert getattr(layer.self_attn.o_proj, "quantized_weight", None) is None
        assert layer.self_attn.qkv_proj.weight is None
        assert layer.self_attn.qkv_proj.quantized_weight is not None
    assert set(registry.full_params) == {"layers.0.self_attn.o_proj.weight", "layers.1.self_attn.o_proj.weight"}


def test_hybrid_peft_export_writes_full_params_sidecar_cpu(tmp_path):
    model = build_model()
    config = LoraConfig(rank=2, alpha=4, full_param_modules=("norm",))
    registry = initialize_lora(model, config, seed=3)
    exported = export_peft_adapter(registry, tmp_path, base_model_name_or_path="test-base")
    assert exported == str(tmp_path)
    assert (tmp_path / "adapter_model.safetensors").exists()
    assert (tmp_path / "areno_full_params.safetensors").exists()
    adapter_config = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert "full_param_modules" not in adapter_config
    saved = load_file(tmp_path / "areno_full_params.safetensors", device="cpu")
    assert set(saved) == set(registry.full_params)
    for name, parameter in registry.full_params.items():
        torch.testing.assert_close(saved[name], parameter.detach().float().cpu(), atol=0, rtol=0)


def test_hybrid_peft_roundtrip_restores_full_params_cpu(tmp_path):
    source = build_model()
    config = LoraConfig(rank=2, alpha=4, full_param_modules=("norm",))
    source_registry = initialize_lora(source, config, seed=3)
    export_peft_adapter(source_registry, tmp_path, base_model_name_or_path="test-base")

    target_registry = initialize_lora(build_model(), config, seed=7)
    load_peft_adapter(target_registry, tmp_path)
    assert set(target_registry.slots) == set(source_registry.slots)
    for name, slot in source_registry.slots.items():
        other = target_registry.slots[name]
        torch.testing.assert_close(other.lora_A, slot.lora_A, atol=0, rtol=0)
        torch.testing.assert_close(other.lora_B, slot.lora_B, atol=0, rtol=0)
    for name, parameter in source_registry.full_params.items():
        torch.testing.assert_close(target_registry.full_params[name], parameter, atol=0, rtol=0)


def test_hybrid_peft_load_rejects_sidecar_mismatch_cpu(tmp_path):
    config = LoraConfig(rank=2, alpha=4, full_param_modules=("norm",))
    source_registry = initialize_lora(build_model(), config, seed=3)
    export_peft_adapter(source_registry, tmp_path, base_model_name_or_path="test-base")

    (tmp_path / "areno_full_params.safetensors").unlink()
    hybrid_registry = initialize_lora(build_model(), config, seed=5)
    with pytest.raises(ValueError, match="areno_full_params"):
        load_peft_adapter(hybrid_registry, tmp_path)

    export_peft_adapter(source_registry, tmp_path, base_model_name_or_path="test-base")
    plain_registry = initialize_lora(build_model(), LoraConfig(rank=2, alpha=4), seed=5)
    with pytest.raises(ValueError, match="no full_param_modules"):
        load_peft_adapter(plain_registry, tmp_path)


def test_cli_full_param_modules_requires_lora_cpu():
    args = SimpleNamespace(lora_rank=None, lora_adapter_path=None, qlora=False, full_param_modules="router")
    with pytest.raises(click.UsageError, match="requires LoRA"):
        _lora_config_from_options(args)


def test_cli_full_param_modules_parses_cpu():
    args = SimpleNamespace(
        lora_rank=4,
        lora_adapter_path=None,
        qlora=False,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_target_modules="q_proj,k_proj",
        full_param_modules=" norm , q_norm ",
    )
    config = _lora_config_from_options(args)
    assert config.full_param_modules == ("norm", "q_norm")
    args.full_param_modules = None
    assert _lora_config_from_options(args).full_param_modules == ()
