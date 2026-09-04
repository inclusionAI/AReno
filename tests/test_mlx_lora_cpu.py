"""CPU-only contract tests for PEFT-compatible MLX LoRA support."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file

from areno.adapters import LoraConfig
from areno.api.backend.mlx.lora import (
    MlxLoraState,
    export_peft_adapter,
    initialize_lora,
    load_peft_adapter,
)


class _FakeLinear:
    def __init__(self, input_dims: int, output_dims: int) -> None:
        self.input_dims = input_dims
        self.output_dims = output_dims


class _FakeQuantizedLinear(_FakeLinear):
    pass


class _FakeLoraLinear:
    events: list[str] = []

    def __init__(self, base: _FakeLinear, *, rank: int, dropout: float, scale: float) -> None:
        self.linear = base
        self.lora_a = np.zeros((base.input_dims, rank), dtype=np.float32)
        self.lora_b = np.zeros((rank, base.output_dims), dtype=np.float32)
        self.dropout = dropout
        self.scale = scale

    @classmethod
    def from_base(cls, base: _FakeLinear, *, r: int, dropout: float, scale: float):
        assert cls.events and cls.events[0] == "freeze"
        cls.events.append("wrap")
        return cls(base, rank=r, dropout=dropout, scale=scale)


class _FakeModel:
    def __init__(self, modules: dict[str, object], *, extra_trainable: str | None = None) -> None:
        self.modules = modules
        self.extra_trainable = extra_trainable
        self.frozen = False
        self.updated = False

    def named_modules(self):
        return list(self.modules.items())

    def freeze(self) -> None:
        self.frozen = True
        _FakeLoraLinear.events.append("freeze")

    def update_modules(self, replacements: dict[str, object]) -> None:
        self.updated = True
        self.modules.update(replacements)

    def trainable_parameters(self) -> dict[str, np.ndarray]:
        parameters = {}
        for path, module in self.modules.items():
            if isinstance(module, _FakeLoraLinear):
                parameters[f"{path}.lora_a"] = module.lora_a
                parameters[f"{path}.lora_b"] = module.lora_b
        if self.extra_trainable is not None:
            parameters[self.extra_trainable] = np.zeros((1,), dtype=np.float32)
        return parameters


def _fake_api(*, tensors: dict[str, np.ndarray] | None = None, evaluated: list | None = None):
    def load(path: str):
        del path
        assert tensors is not None
        return tensors

    def evaluate(*arrays) -> None:
        if evaluated is not None:
            evaluated.extend(arrays)

    return SimpleNamespace(
        mx=SimpleNamespace(load=load, eval=evaluate, float32=np.float32),
        linear_type=_FakeLinear,
        quantized_linear_type=_FakeQuantizedLinear,
        lora_linear_type=_FakeLoraLinear,
        tree_flatten=lambda tree: list(tree.items()),
        tree_unflatten=lambda pairs: dict(pairs),
    )


@pytest.fixture(autouse=True)
def _reset_fake_lora_events():
    _FakeLoraLinear.events = []


def test_initialize_lora_freezes_base_and_exposes_only_adapter_parameters(monkeypatch):
    modules = {
        "model.layers.0.self_attn.q_proj": _FakeLinear(4, 6),
        "model.layers.0.self_attn.v_proj": _FakeLinear(4, 2),
        "model.layers.0.mlp.up_proj": _FakeLinear(4, 8),
    }
    model = _FakeModel(modules)
    config = LoraConfig(rank=2, alpha=8, target_modules=("q_proj", "v_proj"))
    monkeypatch.setattr("areno.api.backend.mlx.lora._mlx_lora_api", _fake_api)

    state = initialize_lora(model, config, model_type="qwen3")

    assert model.frozen
    assert model.updated
    assert _FakeLoraLinear.events == ["freeze", "wrap", "wrap"]
    assert set(state.slots) == {"layers.0.self_attn.q_proj", "layers.0.self_attn.v_proj"}
    assert all(slot.scale == 4.0 for slot in state.slots.values())
    assert all(slot.dropout == 0.0 for slot in state.slots.values())
    assert set(model.trainable_parameters()) == {
        "model.layers.0.self_attn.q_proj.lora_a",
        "model.layers.0.self_attn.q_proj.lora_b",
        "model.layers.0.self_attn.v_proj.lora_a",
        "model.layers.0.self_attn.v_proj.lora_b",
    }
    assert isinstance(modules["model.layers.0.mlp.up_proj"], _FakeLinear)


def test_initialize_lora_validates_all_targets_before_freezing(monkeypatch):
    model = _FakeModel({"model.layers.0.self_attn.q_proj": _FakeLinear(4, 6)})
    config = LoraConfig(rank=2, target_modules=("q_proj", "v_proj"))
    monkeypatch.setattr("areno.api.backend.mlx.lora._mlx_lora_api", _fake_api)

    with pytest.raises(ValueError, match="v_proj"):
        initialize_lora(model, config, model_type="qwen3")

    assert not model.frozen
    assert not model.updated


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3_moe"])
def test_initialize_lora_rejects_unsupported_model_families_before_import(monkeypatch, model_type):
    monkeypatch.setattr(
        "areno.api.backend.mlx.lora._mlx_lora_api",
        lambda: pytest.fail("MLX should not be imported for an unsupported model family"),
    )

    with pytest.raises(ValueError, match="qwen3"):
        initialize_lora(_FakeModel({}), LoraConfig(), model_type=model_type)


def test_initialize_lora_rejects_quantized_targets_before_freezing(monkeypatch):
    model = _FakeModel({"model.layers.0.self_attn.q_proj": _FakeQuantizedLinear(4, 6)})
    config = LoraConfig(rank=2, target_modules=("q_proj",))
    monkeypatch.setattr("areno.api.backend.mlx.lora._mlx_lora_api", _fake_api)

    with pytest.raises(ValueError, match="QLoRA"):
        initialize_lora(model, config, model_type="qwen3")

    assert not model.frozen


def test_initialize_lora_rejects_any_unexpected_trainable_parameter(monkeypatch):
    model = _FakeModel(
        {"model.layers.0.self_attn.q_proj": _FakeLinear(4, 6)},
        extra_trainable="model.embed_tokens.weight",
    )
    config = LoraConfig(rank=2, target_modules=("q_proj",))
    monkeypatch.setattr("areno.api.backend.mlx.lora._mlx_lora_api", _fake_api)

    with pytest.raises(RuntimeError, match="only adapter parameters trainable"):
        initialize_lora(model, config, model_type="qwen3")


def test_load_peft_adapter_transposes_dense_weights_atomically(monkeypatch, tmp_path):
    slot = _FakeLoraLinear(_FakeLinear(3, 4), rank=2, dropout=0.0, scale=1.0)
    state = MlxLoraState(
        slots={"layers.0.self_attn.q_proj": slot},
        config=LoraConfig(rank=2, target_modules=("q_proj",)),
    )
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    peft_a = np.arange(6, dtype=np.float64).reshape(2, 3)
    peft_b = np.arange(8, dtype=np.float64).reshape(4, 2)
    tensors = {
        f"{prefix}.lora_A.weight": peft_a,
        f"{prefix}.lora_B.weight": peft_b,
    }
    evaluated = []
    adapter_file = tmp_path / "adapter_model.safetensors"
    adapter_file.touch()
    monkeypatch.setattr(
        "areno.api.backend.mlx.lora._mlx_lora_api",
        lambda: _fake_api(tensors=tensors, evaluated=evaluated),
    )

    load_peft_adapter(state, tmp_path)

    np.testing.assert_array_equal(slot.lora_a, peft_a.T.astype(np.float32))
    np.testing.assert_array_equal(slot.lora_b, peft_b.T.astype(np.float32))
    assert slot.lora_a.dtype == np.float32
    assert slot.lora_b.dtype == np.float32
    assert len(evaluated) == 2


@pytest.mark.parametrize(
    "tensors, match",
    [
        ({"unexpected": np.zeros((1,), dtype=np.float32)}, "tensor keys do not match"),
        (
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.zeros((3, 3), dtype=np.float32),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.zeros((4, 2), dtype=np.float32),
            },
            "has shape",
        ),
    ],
)
def test_load_peft_adapter_prevalidates_before_mutating_slots(monkeypatch, tmp_path, tensors, match):
    slot = _FakeLoraLinear(_FakeLinear(3, 4), rank=2, dropout=0.0, scale=1.0)
    slot.lora_a.fill(7)
    slot.lora_b.fill(9)
    original_a = slot.lora_a.copy()
    original_b = slot.lora_b.copy()
    state = MlxLoraState(
        slots={"layers.0.self_attn.q_proj": slot},
        config=LoraConfig(rank=2, target_modules=("q_proj",)),
    )
    (tmp_path / "adapter_model.safetensors").touch()
    monkeypatch.setattr(
        "areno.api.backend.mlx.lora._mlx_lora_api",
        lambda: _fake_api(tensors=tensors),
    )

    with pytest.raises(ValueError, match=match):
        load_peft_adapter(state, tmp_path)

    np.testing.assert_array_equal(slot.lora_a, original_a)
    np.testing.assert_array_equal(slot.lora_b, original_b)


def test_export_peft_adapter_writes_reloadable_standard_artifact(monkeypatch, tmp_path):
    slot = _FakeLoraLinear(_FakeLinear(3, 4), rank=2, dropout=0.0, scale=4.0)
    slot.lora_a = np.arange(6, dtype=np.float64).reshape(3, 2)
    slot.lora_b = np.arange(8, dtype=np.float64).reshape(2, 4)
    config = LoraConfig(rank=2, alpha=8, target_modules=("q_proj",))
    state = MlxLoraState(slots={"layers.0.self_attn.q_proj": slot}, config=config)
    evaluated = []
    monkeypatch.setattr(
        "areno.api.backend.mlx.lora._mlx_lora_api",
        lambda: _fake_api(evaluated=evaluated),
    )
    output_path = tmp_path / "nested" / "adapter"

    exported = export_peft_adapter(
        state,
        output_path,
        base_model_name_or_path="Qwen/Qwen3-0.6B",
    )

    assert exported == str(output_path)
    metadata = json.loads((output_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert metadata == {
        "base_model_name_or_path": "Qwen/Qwen3-0.6B",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": 2,
        "target_modules": ["q_proj"],
        "task_type": "CAUSAL_LM",
    }
    tensors = load_file(output_path / "adapter_model.safetensors")
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    np.testing.assert_array_equal(tensors[f"{prefix}.lora_A.weight"], slot.lora_a.T.astype(np.float32))
    np.testing.assert_array_equal(tensors[f"{prefix}.lora_B.weight"], slot.lora_b.T.astype(np.float32))
    assert tensors[f"{prefix}.lora_A.weight"].flags.c_contiguous
    assert tensors[f"{prefix}.lora_B.weight"].flags.c_contiguous
    assert len(evaluated) == 2
    assert not list(output_path.glob(".areno-peft-*"))

    reloaded_slot = _FakeLoraLinear(_FakeLinear(3, 4), rank=2, dropout=0.0, scale=4.0)
    reloaded = MlxLoraState(slots={"layers.0.self_attn.q_proj": reloaded_slot}, config=config)
    monkeypatch.setattr(
        "areno.api.backend.mlx.lora._mlx_lora_api",
        lambda: _fake_api(tensors=tensors),
    )
    load_peft_adapter(reloaded, output_path)

    np.testing.assert_array_equal(reloaded_slot.lora_a, slot.lora_a.astype(np.float32))
    np.testing.assert_array_equal(reloaded_slot.lora_b, slot.lora_b.astype(np.float32))


def test_export_peft_adapter_preserves_existing_artifact_on_write_failure(monkeypatch, tmp_path):
    import safetensors.numpy as safetensors_numpy

    slot = _FakeLoraLinear(_FakeLinear(3, 4), rank=2, dropout=0.0, scale=1.0)
    state = MlxLoraState(
        slots={"layers.0.self_attn.q_proj": slot},
        config=LoraConfig(rank=2, target_modules=("q_proj",)),
    )
    output_path = tmp_path / "adapter"
    output_path.mkdir()
    config_path = output_path / "adapter_config.json"
    weights_path = output_path / "adapter_model.safetensors"
    config_path.write_text("old config", encoding="utf-8")
    weights_path.write_bytes(b"old weights")
    monkeypatch.setattr("areno.api.backend.mlx.lora._mlx_lora_api", _fake_api)

    def fail_save(*args, **kwargs):
        del args, kwargs
        raise OSError("disk")

    monkeypatch.setattr(safetensors_numpy, "save_file", fail_save)

    with pytest.raises(OSError, match="disk"):
        export_peft_adapter(state, output_path, base_model_name_or_path="base")

    assert config_path.read_text(encoding="utf-8") == "old config"
    assert weights_path.read_bytes() == b"old weights"
    assert not list(output_path.glob(".areno-peft-*"))
