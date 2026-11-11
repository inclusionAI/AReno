"""Qwen3.5 canonical targets, frozen-base gradients and PEFT tensor layouts."""

import copy
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from areno.adapters import LoraConfig
from areno.adapters.config import QWEN35_TARGETS
from areno.adapters.lora import AdapterRegistry, LoraSlot, _AdapterRuntimeState, initialize_lora
from areno.adapters.peft import export_peft_adapter, load_peft_adapter
from areno.adapters.qlora import initialize_qlora
from areno.engine.config import EngineConfig, ModelConfig
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context


@pytest.fixture(autouse=True)
def single_rank():
    old = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    yield
    set_tp_context(old)


@pytest.fixture(scope="module")
def qwen35_module():
    import areno.models
    from areno.models import registry

    imported = set(sys.modules)
    adapters, groups = registry._ADAPTERS.copy(), areno.models._REGISTERED_GROUPS.copy()
    with pytest.MonkeyPatch.context() as patch:
        if importlib.util.find_spec("triton") is None:
            from areno.accel import areno_silu_and_mul
            from areno.accel.utils import log_once

            def unavailable(*args, **kwargs):
                raise AssertionError("CPU target-layout tests must not execute GPU-only kernels")

            ops = ModuleType("areno.accel.ops")
            ops.FusedMoeConfig = SimpleNamespace
            ops.SegLaMeta = SimpleNamespace
            ops.areno_fused_experts = unavailable
            ops.rms_norm_gate_fwd = unavailable
            ops.seg_la_fwd = unavailable
            ops.chunk_lightning_attn = unavailable
            ops.areno_silu_and_mul = areno_silu_and_mul
            ops.log_once = log_once
            patch.setitem(sys.modules, "areno.accel.ops", ops)
        from areno.models.qwen3_5 import model

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


def build_model(module, *, vision=False):
    config = ModelConfig(
        model_type="qwen3_5_vl" if vision else "qwen3_5",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attn_output_gate=True,
        layer_types=("linear_attention", "full_attention"),
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        dtype=torch.float32,
        attn_backend="native",
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "in_channels": 3,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "out_hidden_size": 32,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
        }
        if vision
        else None,
    )
    cls = module.Qwen35VLForConditionalGeneration if vision else module.Qwen35ForCausalLM
    model = cls(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    return model


@pytest.mark.parametrize("vision", [False, True], ids=["text", "vl"])
@pytest.mark.parametrize("qlora", [False, True], ids=["lora", "qlora"])
def test_qwen35_targets_frozen_base_and_roundtrip_cpu(qwen35_module, tmp_path, vision, qlora):
    torch.manual_seed(417)
    model = build_model(qwen35_module, vision=vision)
    restored = copy.deepcopy(model)
    config = LoraConfig(rank=2, alpha=4, target_modules=QWEN35_TARGETS, qlora=qlora)
    registry = initialize_lora(model, config, seed=9)
    assert len(registry.slots) == 15
    assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in registry.parameters()}
    root = "language_model." if vision else ""
    qkv = registry.slots[f"{root}layers.0.linear_attn.in_proj_qkv"]
    assert qkv.lora_B.shape == (32, 2)  # K=8, K=8, V=16 in canonical HF order.
    if qlora:
        EngineConfig(model=model.config, lora=config, devices=[0])
        memory = initialize_qlora(model)
        assert memory["projections"] == 9
        assert memory["quantized_weight_bytes"] < memory["original_weight_bytes"]
    with torch.no_grad():
        for slot in registry.slots.values():
            slot.lora_B.normal_(0, 0.02)
    export_peft_adapter(registry, tmp_path, base_model_name_or_path="test-base")
    artifact = load_file(tmp_path / "adapter_model.safetensors")
    assert f"base_model.model.model.{root}layers.0.linear_attn.in_proj_qkv.lora_B.weight" in artifact
    saved = LoraConfig(adapter_path=str(tmp_path))
    assert saved.qlora == qlora
    restored_registry = initialize_lora(restored, saved, seed=23)
    if qlora:
        initialize_qlora(restored)
    load_peft_adapter(restored_registry, tmp_path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    if vision:
        assert all(not p.requires_grad for p in model.visual.parameters())


@pytest.mark.parametrize("qlora", [False, True], ids=["lora", "qlora"])
def test_qwen35_fused_gdn_projection_and_gradients_cpu(qwen35_module, qlora):
    torch.manual_seed(519)
    model = build_model(qwen35_module)
    owner = model.layers[0].attention.in_proj_qkvz
    before = copy.deepcopy(owner)
    config = LoraConfig(rank=2, alpha=4, target_modules=("in_proj_qkv", "in_proj_z"), qlora=qlora)
    registry = initialize_lora(model, config, seed=2)
    x = torch.randn(2, 3, 32, requires_grad=True)
    torch.testing.assert_close(owner(x), before(x), atol=0, rtol=0)
    if qlora:
        initialize_qlora(model)
        weight = owner.quantized_weight.dequantize()
    else:
        weight = owner.weight
    with torch.no_grad():
        for slot in registry.slots.values():
            slot.lora_B.normal_(0, 0.03)
    qkv, z = registry.slots.values()
    expected = F.linear(x, weight) + torch.cat((qkv(x), z(x)), dim=-1)
    actual = owner(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    parameters = (x, *registry.parameters())
    wanted = torch.autograd.grad(expected.square().sum(), parameters)
    grads = torch.autograd.grad(actual.square().sum(), parameters)
    for got, reference in zip(grads, wanted, strict=True):
        assert torch.isfinite(got).all() and got.abs().sum() > 0
        torch.testing.assert_close(got, reference, atol=1e-5, rtol=1e-5)
    with registry.base_only():
        torch.testing.assert_close(owner(x), F.linear(x, weight), atol=0, rtol=0)


def test_fused_component_peft_tp2_canonical_row_order_cpu(monkeypatch, tmp_path):
    import areno.adapters.peft as peft
    from areno.engine.policy_sync import build_adapter_policy_plan

    config = LoraConfig(rank=2, alpha=4, target_modules=("in_proj_qkv",))
    name = "layers.0.linear_attn.in_proj_qkv"
    canonical_A = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    canonical_B = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    save_file(
        {
            f"base_model.model.model.{name}.lora_A.weight": canonical_A,
            f"base_model.model.model.{name}.lora_B.weight": canonical_B,
        },
        tmp_path / "adapter_model.safetensors",
    )
    registries, layouts, contributions = [], [], []
    for rank in range(2):
        set_tp_context(TPContext(rank=rank, world_size=2, device=torch.device("cpu"), group=None))
        state = _AdapterRuntimeState()
        slot = LoraSlot(
            logical_name=name,
            base_weight=torch.nn.Parameter(torch.zeros(1)),
            global_in_features=16,
            global_out_features=16,
            local_in_features=16,
            local_out_features=8,
            row_parallel=False,
            config=config,
            seed=3,
            runtime_state=state,
            output_splits=(4, 4, 8),
        )
        registry = AdapterRegistry({name: slot}, config, state)
        load_peft_adapter(registry, tmp_path)
        expected_rows = [rank * 2, rank * 2 + 1, 4 + rank * 2, 5 + rank * 2, *range(8 + rank * 4, 12 + rank * 4)]
        torch.testing.assert_close(slot.lora_B, canonical_B[expected_rows], atol=0, rtol=0)
        layout = build_adapter_policy_plan(registry)[f"{name}.lora_B.weight"].policy_layout()
        assert [(piece.shard_start, piece.shard_end) for piece in layout.pieces] == list(slot.output_ranges)
        contribution = torch.empty(canonical_B.numel())
        layout.read_chunk(0, contribution)
        contributions.append(contribution)
        layouts.append(layout)
        registries.append(registry)
    torch.testing.assert_close(torch.stack(contributions).sum(0), canonical_B.flatten(), atol=0, rtol=0)
    set_tp_context(TPContext(rank=0, world_size=2, device=torch.device("cpu"), group=None))
    monkeypatch.setattr(
        peft,
        "_all_gather",
        lambda tensor, *_: (
            [canonical_A, canonical_A]
            if tensor.shape == canonical_A.shape
            else [next(iter(registry.slots.values())).lora_B.detach() for registry in registries]
        ),
    )
    export_peft_adapter(registries[0], tmp_path / "export", base_model_name_or_path="test-base")
    exported = load_file(tmp_path / "export/adapter_model.safetensors")
    torch.testing.assert_close(exported[f"base_model.model.model.{name}.lora_B.weight"], canonical_B, atol=0, rtol=0)
    for registry, layout in zip(registries, layouts, strict=True):
        slot = registry.slots[name]
        expected = slot.lora_B.detach().clone() + 1
        layout.write_chunk(0, (canonical_B + 1).flatten())
        torch.testing.assert_close(slot.lora_B, expected, atol=0, rtol=0)


def test_qwen35_replicated_kv_projection_targets_cpu(qwen35_module):
    set_tp_context(TPContext(rank=3, world_size=4, device=torch.device("cpu"), group=None))
    model = qwen35_module.Qwen35ForCausalLM(
        ModelConfig(
            model_type="qwen3_5",
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            attn_output_gate=True,
            layer_types=("full_attention",),
            dtype=torch.float32,
        )
    )
    registry = initialize_lora(model, LoraConfig(rank=2, target_modules=("q_proj", "k_proj", "v_proj")), seed=2)
    assert registry.slots["layers.0.self_attn.k_proj"].output_ranges == ((8, 16),)
    assert registry.slots["layers.0.self_attn.k_proj"].output_replicated
    assert registry.slots["layers.0.self_attn.q_proj"].global_out_features == 64
    owner = model.layers[0].attention.qkv_proj
    with torch.no_grad():
        owner.weight.uniform_(-0.1, 0.1)
    x = torch.randn(2, 3, 32)
    torch.testing.assert_close(owner(x), F.linear(x, owner.weight), atol=0, rtol=0)
    initialize_qlora(model)
    assert owner.weight is None
    torch.testing.assert_close(owner(x), F.linear(x, owner.quantized_weight.dequantize()), atol=0, rtol=0)


def test_qwen35_lora_rejects_unfrozen_vision_cpu(qwen35_module):
    model = build_model(qwen35_module, vision=True)
    model.configure_multimodal_training(
        unfreeze_tower=False, unfreeze_projector=True, tower_lr=None, projector_lr=None, base_lr=1e-4
    )
    with pytest.raises(ValueError, match="frozen vision tower and projector"):
        initialize_lora(model, LoraConfig(), seed=1)
