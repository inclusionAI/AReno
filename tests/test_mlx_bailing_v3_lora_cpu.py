"""Ling MLX target coverage and PEFT contracts without importing MLX."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from areno.adapters import LoraConfig
from areno.api.backend.mlx.lora import export_peft_adapter, initialize_lora, load_peft_adapter
from tests.test_mlx_lora_cpu import (
    _fake_api,
    _FakeLinear,
    _FakeLoraLinear,
    _FakeModel,
    _FakeQuantizedLinear,
    _FakeQuantizedSwitchLinear,
    _FakeSwitchLinear,
)

KDA_TARGETS = ("q_proj", "k_proj", "v_proj", "f_proj", "o_proj")
MLA_TARGETS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "dense")
TARGETS = KDA_TARGETS + MLA_TARGETS
EXPANDED_TARGETS = TARGETS + ("g_proj", "gate_proj", "up_proj", "down_proj")
MODEL_CONFIG = {"model_type": "bailing_hybrid", "architectures": ["BailingMoeV3ForCausalLM"]}


class _LingModel(_FakeModel):
    def __init__(self, layers: int = 4) -> None:
        self.args = SimpleNamespace(
            num_hidden_layers=layers,
            hidden_size=8,
            num_attention_heads=2,
            head_dim=4,
            q_lora_rank=4,
            kv_lora_rank=4,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=4,
            no_kda_lora=True,
            kda_safe_gate=True,
            rope_interleave=True,
            first_k_dense_replace=1,
            intermediate_size=16,
            moe_shared_expert_intermediate_size=12,
            num_shared_experts=1,
            moe_intermediate_size=10,
            num_experts=3,
        )
        self.model = SimpleNamespace(layers=[SimpleNamespace(is_linear=(i % 4 != 3)) for i in range(layers)])
        modules = {}
        for index, layer in enumerate(self.model.layers):
            shapes = (
                {name: (8, 8) for name in KDA_TARGETS}
                if layer.is_linear
                else {"q_a_proj": (4, 8), "q_b_proj": (8, 4), "kv_a_proj_with_mqa": (6, 8), "dense": (8, 8)}
            )
            for name, (out_dims, in_dims) in shapes.items():
                module = _FakeLinear(in_dims, out_dims)
                module.weight = np.zeros((out_dims, in_dims), dtype=np.float32)
                modules[f"model.layers.{index}.attention.{name}"] = module
            modules[f"model.layers.{index}.attention.g_proj"] = _FakeLinear(8, 8 if layer.is_linear else 2)
            mlps = [("mlp", 16)] if index == 0 else [("mlp.shared_experts", 12), ("mlp.switch_mlp", 10)]
            for path, width in mlps:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    in_dims, out_dims = (width, 8) if name == "down_proj" else (8, width)
                    module = (
                        _FakeSwitchLinear(in_dims, out_dims, 3)
                        if "switch_mlp" in path
                        else _FakeLinear(in_dims, out_dims)
                    )
                    modules[f"model.layers.{index}.{path}.{name}"] = module
        super().__init__(modules)


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch):
    _FakeLoraLinear.events = []
    monkeypatch.setattr("areno.api.backend.mlx.lora._mlx_lora_api", _fake_api)


def _initialize(model, targets=TARGETS, *, config=None):
    return initialize_lora(
        model,
        LoraConfig(rank=2, alpha=8, target_modules=targets),
        model_type="bailing_hybrid",
        model_config=MODEL_CONFIG if config is None else config,
    )


def test_all_ling_layers_have_exact_attention_slots():
    model = _LingModel(24)
    state = _initialize(model)
    expected = {
        f"layers.{index}.attention.{name}"
        for index in range(24)
        for name in (MLA_TARGETS if index % 4 == 3 else KDA_TARGETS)
    }
    assert len(expected) == 114
    assert set(state.slots) == expected
    assert set(model.trainable_parameters()) == {
        f"model.{path}.{ab}" for path in expected for ab in ("lora_a", "lora_b")
    }
    assert all(slot.scale == 4 for slot in state.slots.values())
    assert not any("mlp" in path or "g_proj" in path for path in state.slots)


@pytest.mark.parametrize("target", TARGETS)
def test_projection_subset_covers_every_applicable_layer(target):
    state = _initialize(_LingModel(), (target,))
    indices = range(3) if target in KDA_TARGETS else (3,)
    assert set(state.slots) == {f"layers.{i}.attention.{target}" for i in indices}


@pytest.mark.parametrize("targets", [("kv_b_proj",), ("g_proj", "kv_b_proj")])
def test_unsupported_targets_fail_before_freeze(targets):
    model = _LingModel()
    with pytest.raises(ValueError, match="unsupported MLX Ling LoRA targets.*--lora-target-modules"):
        _initialize(model, targets)
    assert not model.frozen and not model.updated


def test_expanded_targets_cover_attention_dense_shared_and_routed_experts(caplog):
    with caplog.at_level("INFO"):
        state = _initialize(_LingModel(), EXPANDED_TARGETS)
    assert len(state.slots) == 44
    assert len(state.expert_slots) == 9
    assert all(".experts.{expert}." in name for name in state.expert_slots)
    assert "layers.0.mlp.gate_proj" in state.slots
    assert "layers.3.mlp.shared_experts.down_proj" in state.slots
    assert "layers.3.attention.g_proj" in state.slots
    assert "adapter_parameters=" in caplog.text and "routed_experts': 9" in caplog.text
    # Existing default suffixes now cover MLP too, without enabling new gates.
    assert _initialize(_LingModel(), LoraConfig().target_modules).expert_slots


@pytest.mark.parametrize("mutation", ["missing_shared", "expert_count", "wrong_type", "outside"])
def test_invalid_mlp_structure_fails_before_freeze(mutation):
    model = _LingModel()
    path = "model.layers.3.mlp.switch_mlp.down_proj"
    if mutation == "missing_shared":
        del model.modules["model.layers.3.mlp.shared_experts.up_proj"]
    elif mutation == "expert_count":
        model.modules[path].weight = np.zeros((2, 8, 10))
    elif mutation == "wrong_type":
        model.modules[path] = SimpleNamespace(weight=np.zeros((3, 8, 10)))
    else:
        model.modules["model.layers.3.mlp.router.gate_proj"] = _FakeLinear(8, 3)
    with pytest.raises((ValueError, TypeError)):
        _initialize(model, EXPANDED_TARGETS)
    assert not model.frozen and not model.updated and not _FakeLoraLinear.events


def test_expert_peft_roundtrip_and_failed_load_are_atomic(monkeypatch, tmp_path):
    state = _initialize(_LingModel(), EXPANDED_TARGETS)
    for index, slot in enumerate(state.slots.values()):
        slot.lora_a[:] = np.arange(slot.lora_a.size).reshape(slot.lora_a.shape) + index
        slot.lora_b[:] = np.arange(slot.lora_b.size).reshape(slot.lora_b.shape) - index
    export_peft_adapter(state, tmp_path, base_model_name_or_path="synthetic-ling")
    tensors = load_file(tmp_path / "adapter_model.safetensors")
    assert len(tensors) == (35 + 9 * 3) * 2
    for name in state.expert_slots:
        for expert in range(3):
            prefix = f"base_model.model.model.{name.format(expert=expert)}"
            np.testing.assert_array_equal(tensors[f"{prefix}.lora_A.weight"], state.slots[name].lora_a[expert])
            np.testing.assert_array_equal(tensors[f"{prefix}.lora_B.weight"], state.slots[name].lora_b[expert])
    monkeypatch.setattr("areno.api.backend.mlx.lora._mlx_lora_api", lambda: _fake_api(tensors=tensors))
    reloaded = initialize_lora(
        _LingModel(), LoraConfig(adapter_path=str(tmp_path)), model_type="bailing_hybrid", model_config=MODEL_CONFIG
    )
    for name, slot in reloaded.slots.items():
        np.testing.assert_array_equal(slot.lora_a, state.slots[name].lora_a)
        np.testing.assert_array_equal(slot.lora_b, state.slots[name].lora_b)
    bad_key = "base_model.model.model.layers.3.mlp.experts.2.up_proj.lora_B.weight"
    tensors[bad_key] = np.zeros((1, 1), dtype=np.float32)
    save_file(tensors, tmp_path / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="has shape"):
        load_peft_adapter(reloaded, tmp_path)
    del tensors[bad_key]
    with pytest.raises(ValueError, match="tensor keys"):
        load_peft_adapter(reloaded, tmp_path)
    for name, slot in reloaded.slots.items():
        np.testing.assert_array_equal(slot.lora_a, state.slots[name].lora_a)
        np.testing.assert_array_equal(slot.lora_b, state.slots[name].lora_b)


@pytest.mark.parametrize("architectures", [None, [], ["BailingMoeForCausalLM"]])
def test_wrong_architecture_fails_before_mlx_import(monkeypatch, architectures):
    monkeypatch.setattr("areno.api.backend.mlx.lora._mlx_lora_api", lambda: pytest.fail("unexpected MLX import"))
    with pytest.raises(ValueError, match="requires architecture"):
        _initialize(_LingModel(), config={"architectures": architectures})


def test_architecture_string_is_accepted():
    assert _initialize(_LingModel(), config={"architectures": "BailingMoeV3ForCausalLM"}).slots


@pytest.mark.parametrize(
    "mutation", ["missing", "shape", "nonlinear", "quantized", "outside", "layer_type", "layer_count"]
)
def test_invalid_structure_never_partially_injects(mutation):
    model = _LingModel()
    path = "model.layers.2.attention.v_proj"
    if mutation == "missing":
        del model.modules[path]
    elif mutation == "shape":
        model.modules[path].weight = np.zeros((2, 8))
    elif mutation == "nonlinear":
        model.modules[path] = SimpleNamespace(weight=np.zeros((8, 8)))
    elif mutation == "quantized":
        module = _FakeQuantizedLinear(8, 8)
        module.bits = 4
        module.scales = np.ones((8, 1))
        module.weight = np.zeros((8, 8))
        model.modules[path] = module
    elif mutation == "outside":
        model.modules["model.layers.3.mlp.q_proj"] = _FakeLinear(8, 8)
    elif mutation == "layer_type":
        model.model.layers[2].is_linear = None
    else:
        model.model.layers.pop()
    with pytest.raises((ValueError, TypeError)):
        _initialize(model)
    assert not model.frozen and not model.updated
    assert not _FakeLoraLinear.events


def test_quantized_checkpoint_allows_dense_excluded_targets():
    model = _LingModel()
    assert _initialize(model, config={**MODEL_CONFIG, "quantization": {"bits": 4}}).slots
    assert model.frozen


def test_online_quantized_expert_allows_ordinary_lora():
    model = _LingModel()
    path = "model.layers.3.mlp.switch_mlp.down_proj"
    model.modules[path] = _FakeQuantizedSwitchLinear(10, 8, 3)
    state = _initialize(model, EXPANDED_TARGETS)
    assert "layers.3.mlp.experts.{expert}.down_proj" in state.expert_slots
    assert state.quantization is None and model.frozen


def test_target_with_no_applicable_layer_is_rejected():
    model = _LingModel(3)
    with pytest.raises(ValueError, match="not present.*q_a_proj"):
        _initialize(model, ("q_a_proj",))
    assert not model.frozen


@pytest.mark.parametrize("option", ["no_kda_lora", "kda_safe_gate", "rope_interleave"])
def test_unsupported_architecture_options(option):
    model = _LingModel()
    setattr(model.args, option, False)
    with pytest.raises(ValueError, match="requires no_kda_lora"):
        _initialize(model)
    assert not model.frozen


def test_ling_peft_export_and_metadata_driven_reload(monkeypatch, tmp_path):
    state = _initialize(_LingModel())
    for index, slot in enumerate(state.slots.values()):
        slot.lora_a[:] = index + 1
        slot.lora_b[:] = index / 10
    export_peft_adapter(state, tmp_path, base_model_name_or_path="inclusionAI/Ling-3.0-tiny")
    tensors = load_file(tmp_path / "adapter_model.safetensors")
    expected = {f"base_model.model.model.{path}.lora_{ab}.weight" for path in state.slots for ab in ("A", "B")}
    assert set(tensors) == expected
    for path, slot in state.slots.items():
        np.testing.assert_array_equal(tensors[f"base_model.model.model.{path}.lora_A.weight"], slot.lora_a.T)
        np.testing.assert_array_equal(tensors[f"base_model.model.model.{path}.lora_B.weight"], slot.lora_b.T)
    config = LoraConfig(adapter_path=str(tmp_path))
    assert config.target_modules == TARGETS
    assert config.rank == 2 and config.alpha == 8
    monkeypatch.setattr("areno.api.backend.mlx.lora._mlx_lora_api", lambda: _fake_api(tensors=tensors))
    reloaded = initialize_lora(_LingModel(), config, model_type="bailing_hybrid", model_config=MODEL_CONFIG)
    for path, slot in reloaded.slots.items():
        np.testing.assert_array_equal(slot.lora_a, state.slots[path].lora_a)
        np.testing.assert_array_equal(slot.lora_b, state.slots[path].lora_b)

    # A bad late MLA tensor must not modify any earlier KDA slot.
    snapshots = {path: (slot.lora_a.copy(), slot.lora_b.copy()) for path, slot in reloaded.slots.items()}
    tensors["base_model.model.model.layers.3.attention.dense.lora_B.weight"] = np.zeros((1, 1))
    with pytest.raises(ValueError, match="has shape"):
        load_peft_adapter(reloaded, tmp_path)
    for path, slot in reloaded.slots.items():
        np.testing.assert_array_equal(slot.lora_a, snapshots[path][0])
        np.testing.assert_array_equal(slot.lora_b, snapshots[path][1])


def test_repeated_backend_initialization_does_not_nest_checkpoint_wrappers(monkeypatch):
    from areno.api.backend.mlx.backend import MlxBackend

    class Layer:
        def __call__(self, x):
            return x

    calls = []
    trainer_module = ModuleType("mlx_lm.tuner.trainer")

    def grad_checkpoint(layer):
        calls.append(layer)
        original = type(layer).__call__

        def wrapped(self, x):
            return original(self, x)

        type(layer).__call__ = wrapped

    trainer_module.grad_checkpoint = grad_checkpoint
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.trainer", trainer_module)
    for _ in range(2):
        backend = MlxBackend()
        backend.provider = SimpleNamespace(generation_model=SimpleNamespace(layers=[Layer(), Layer()]))
        backend._enable_gradient_checkpointing()
    assert len(calls) == 1
    assert Layer()(3) == 3
