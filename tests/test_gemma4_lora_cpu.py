"""Gemma4 adapter layouts, K=V semantics, frozen bases and PEFT persistence."""

import copy
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from areno.adapters import LoraConfig
from areno.adapters.lora import initialize_lora
from areno.adapters.peft import export_peft_adapter, load_peft_adapter
from areno.adapters.qlora import initialize_qlora
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context


@pytest.fixture(autouse=True)
def single_rank():
    old = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    yield
    set_tp_context(old)


@pytest.fixture(scope="module")
def gemma4_module():
    import areno.models
    from areno.models import registry

    imported = set(sys.modules)
    adapters, groups = registry._ADAPTERS.copy(), areno.models._REGISTERED_GROUPS.copy()
    with pytest.MonkeyPatch.context() as patch:
        if importlib.util.find_spec("triton") is None:
            from areno.accel.utils import log_once

            def unavailable(*args, **kwargs):
                raise AssertionError("CPU layout tests must not execute GPU-only kernels")

            ops = ModuleType("areno.accel.ops")
            from areno.accel import areno_silu_and_mul

            ops.SegLaMeta = SimpleNamespace
            ops.areno_silu_and_mul = areno_silu_and_mul
            ops.chunk_lightning_attn = unavailable
            ops.seg_la_fwd = unavailable
            ops.FusedMoeConfig = SimpleNamespace
            ops.areno_fused_experts = unavailable
            ops.areno_gelu_tanh_and_mul = lambda x: F.gelu(x.chunk(2, -1)[0], approximate="tanh") * x.chunk(2, -1)[1]
            ops.log_once = log_once
            patch.setitem(sys.modules, "areno.accel.ops", ops)
        from areno.models.gemma4 import model

        yield model
        registry._ADAPTERS.clear()
        registry._ADAPTERS.update(adapters)
        areno.models._REGISTERED_GROUPS.clear()
        areno.models._REGISTERED_GROUPS.update(groups)
        if importlib.util.find_spec("triton") is None:
            for name in list(sys.modules):
                if name.startswith("areno.") and name not in imported:
                    module = sys.modules.pop(name)
                    parent_name, _, child_name = name.rpartition(".")
                    parent = sys.modules.get(parent_name)
                    if parent is not None and getattr(parent, child_name, None) is module:
                        delattr(parent, child_name)


def build_model(module, *, multimodal=False, shared_tail=False):
    cfg = {
        "model_type": "gemma4",
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 4,
        "global_head_dim": 8,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 8,
        "attention_k_eq_v": True,
        "max_position_embeddings": 32,
    }
    if shared_tail:
        cfg.update(num_hidden_layers=4, num_kv_shared_layers=2, layer_types=cfg["layer_types"] * 2)
    hf = {"model_type": "gemma4_unified", "text_config": cfg} if multimodal else cfg
    config = module.Gemma4Adapter().config_from_hf(hf)
    config.dtype = torch.float32
    config.attn_backend = "native"
    model = module.Gemma4ForCausalLM(config)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.1, 0.1)
    return model


@pytest.mark.parametrize("multimodal", [False, True], ids=["text", "unified"])
@pytest.mark.parametrize("qlora", [False, True], ids=["lora", "qlora"])
def test_targets_frozen_base_and_peft_roundtrip(gemma4_module, tmp_path, multimodal, qlora):
    model = build_model(gemma4_module, multimodal=multimodal)
    restored = copy.deepcopy(model)
    registry = initialize_lora(model, LoraConfig(rank=2, alpha=4, qlora=qlora), seed=7)
    assert len(registry.slots) == 13  # Seven sliding targets, six full targets (K=V).
    assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in registry.parameters()}
    root = "language_model." if multimodal else ""
    name = f"{root}layers.1.self_attn.k_proj"
    assert registry.slots[name].lora_B.shape == (8, 2)
    assert f"{root}layers.1.self_attn.v_proj" not in registry.slots
    full = model.layers[1].self_attn.qkv_proj
    assert full.lora_slots["k_proj"] is full.lora_slots["shared_v_proj"]
    if qlora:
        memory = initialize_qlora(model)
        assert memory["projections"] == 8
        assert memory["quantized_weight_bytes"] < memory["original_weight_bytes"]
    with torch.no_grad():
        for slot in registry.slots.values():
            slot.lora_B.normal_(0, 0.02)
    export_peft_adapter(registry, tmp_path, base_model_name_or_path="test-gemma4")
    artifact = load_file(tmp_path / "adapter_model.safetensors")
    assert f"base_model.model.model.{name}.lora_B.weight" in artifact
    assert not any("layers.1.self_attn.v_proj" in key or "shared_v_proj" in key for key in artifact)
    saved = LoraConfig(adapter_path=str(tmp_path))
    restored_registry = initialize_lora(restored, saved, seed=23)
    if qlora:
        initialize_qlora(restored)
    load_peft_adapter(restored_registry, tmp_path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize("qlora", [False, True], ids=["lora", "qlora"])
def test_shared_kv_output_and_gradients(gemma4_module, qlora):
    model = build_model(gemma4_module)
    registry = initialize_lora(model, LoraConfig(rank=2, alpha=4, target_modules=("k_proj",), qlora=qlora), seed=9)
    if qlora:
        initialize_qlora(model)
    full = model.layers[1].self_attn.qkv_proj
    x = torch.randn(2, 3, 32, requires_grad=True)
    with registry.base_only():
        base = full(x)
    slot = registry.slots["layers.1.self_attn.k_proj"]
    with torch.no_grad():
        slot.lora_B.normal_(0, 0.02)
    q, k, v = (full(x) - base).split(full.local_out_features, dim=-1)
    torch.testing.assert_close(q, torch.zeros_like(q))
    torch.testing.assert_close(k, slot(x), atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(v, slot(x), atol=1e-7, rtol=1e-5)
    (k.square().sum() + 2 * v.square().sum()).backward()
    assert slot.lora_A.grad.abs().sum() > 0 and torch.isfinite(slot.lora_A.grad).all()
    assert slot.lora_B.grad.abs().sum() > 0 and torch.isfinite(slot.lora_B.grad).all()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_shared_tail_skips_dead_kv_targets(gemma4_module):
    model = build_model(gemma4_module, shared_tail=True)
    registry = initialize_lora(model, LoraConfig(rank=2, alpha=4), seed=3)
    assert len(registry.slots) == 23
    for index in (2, 3):
        assert f"layers.{index}.self_attn.q_proj" in registry.slots
        assert f"layers.{index}.self_attn.k_proj" not in registry.slots
        assert f"layers.{index}.self_attn.v_proj" not in registry.slots


def test_rejects_unvalidated_moe_and_unfrozen_media(gemma4_module):
    model = build_model(gemma4_module)
    model.config.enable_moe_block = True
    with pytest.raises(ValueError, match="dense text trunks"):
        initialize_lora(model, LoraConfig(), seed=1)
    model.config.enable_moe_block = False
    model._train_multimodal_projector = True
    with pytest.raises(ValueError, match="frozen multimodal"):
        initialize_lora(model, LoraConfig(), seed=1)


@pytest.mark.parametrize("rank", [0, 1])
def test_tp_replicates_single_global_kv_head(gemma4_module, rank):
    set_tp_context(TPContext(rank=rank, world_size=2, device=torch.device("cpu"), group=None))
    model = build_model(gemma4_module)
    registry = initialize_lora(model, LoraConfig(rank=2, alpha=4), seed=7)
    shared = registry.slots["layers.1.self_attn.k_proj"]
    assert shared.global_out_features == shared.local_out_features == 8
    assert shared.output_ranges == ((0, 8),)
    assert shared.output_replicated
    q = registry.slots["layers.1.self_attn.q_proj"]
    assert q.global_out_features == 32 and q.local_out_features == 16
    assert q.output_ranges == ((rank * 16, (rank + 1) * 16),)
    assert model.layers[1].self_attn.qkv_proj.lora_slots["shared_v_proj"] is shared
