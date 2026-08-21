from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import areno.engine.api as engine_api
from areno.adapters import LoraConfig
from areno.adapters.lora import AdapterRegistry, LoraSlot, RoutedExpertLoraSlot, _AdapterRuntimeState
from areno.adapters.peft import export_peft_adapter, load_peft_adapter
from areno.engine.config import ModelConfig, RuntimeConfig
from areno.engine.parallel.context import TPContext, set_tp_context

_PREFIX = "base_model.model.model."


def _set_cpu_tp(*, rank: int = 0, world_size: int = 1) -> None:
    set_tp_context(TPContext(rank=rank, world_size=world_size, device=torch.device("cpu"), group=None))


def _dense_registry() -> AdapterRegistry:
    _set_cpu_tp()
    state = _AdapterRuntimeState()
    config = LoraConfig(rank=2, alpha=4.0, target_modules=("q_proj",))
    slot = LoraSlot(
        logical_name="layers.0.self_attn.q_proj",
        base_weight=nn.Parameter(torch.zeros(1)),
        global_in_features=4,
        global_out_features=3,
        local_in_features=4,
        local_out_features=3,
        row_parallel=False,
        config=config,
        seed=1,
        runtime_state=state,
    )
    return AdapterRegistry({slot.logical_name: slot}, config, state)


def _expert_registry() -> AdapterRegistry:
    _set_cpu_tp(rank=1, world_size=2)
    state = _AdapterRuntimeState()
    config = LoraConfig(rank=2, alpha=4.0, target_modules=("down_proj",))
    slot = RoutedExpertLoraSlot(
        logical_name="layers.0.mlp.experts.{expert}.down_proj",
        base_weight=nn.Parameter(torch.zeros(1)),
        local_num_experts=2,
        local_expert_start=2,
        in_features=3,
        out_features=4,
        config=config,
        seed=1,
        runtime_state=state,
    )
    return AdapterRegistry({slot.logical_name: slot}, config, state)


def _key(logical_name: str, component: str) -> str:
    return f"{_PREFIX}{logical_name}.lora_{component}.weight"


def _dense_state() -> dict[str, torch.Tensor]:
    logical_name = "layers.0.self_attn.q_proj"
    return {
        _key(logical_name, "A"): torch.arange(8, dtype=torch.float32).reshape(2, 4),
        _key(logical_name, "B"): torch.arange(6, dtype=torch.float32).reshape(3, 2),
    }


def test_lora_config_rejects_semantic_modifiers_and_allows_neutral_values(tmp_path) -> None:
    base_config = {
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0,
        "bias": "none",
        "target_modules": ["q_proj"],
    }
    modifiers = {
        "alora_invocation_tokens": [1, 2],
        "layer_replication": [[0, 1]],
        "trainable_token_indices": [0],
        "target_parameters": ["layers.0.weight"],
        "use_qalora": True,
    }
    for modifier, value in modifiers.items():
        config = {**base_config, modifier: value}
        (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(ValueError, match=modifier):
            LoraConfig(adapter_path=str(tmp_path))

    neutral_config = {
        **base_config,
        "alora_invocation_tokens": None,
        "layer_replication": [],
        "trainable_token_indices": None,
        "target_parameters": [],
        "use_qalora": False,
        "qalora_group_size": 16,
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(neutral_config), encoding="utf-8")

    loaded = LoraConfig(adapter_path=str(tmp_path))

    assert loaded.target_modules == ("q_proj",)


def test_peft_load_requires_exact_expert_key_set(tmp_path) -> None:
    registry = _expert_registry()
    state = {}
    for expert_id in range(4):
        logical_name = f"layers.0.mlp.experts.{expert_id}.down_proj"
        state[_key(logical_name, "A")] = torch.full((2, 3), float(expert_id + 1))
        state[_key(logical_name, "B")] = torch.full((4, 2), float(expert_id + 2))
    save_file(state, tmp_path / "adapter_model.safetensors")
    load_peft_adapter(registry, tmp_path)

    slot = registry.slots["layers.0.mlp.experts.{expert}.down_proj"]
    torch.testing.assert_close(slot.lora_A[0], state[_key("layers.0.mlp.experts.2.down_proj", "A")])
    torch.testing.assert_close(slot.lora_B[1], state[_key("layers.0.mlp.experts.3.down_proj", "B")])

    del state[_key("layers.0.mlp.experts.3.down_proj", "B")]
    state[_key("layers.0.mlp.experts.4.down_proj", "A")] = torch.zeros(2, 3)
    (tmp_path / "adapter_model.safetensors").unlink()
    save_file(state, tmp_path / "adapter_model.safetensors")

    with pytest.raises(ValueError, match="tensor keys"):
        load_peft_adapter(registry, tmp_path)


def test_peft_load_rejects_broadcastable_shape_before_copy(tmp_path) -> None:
    registry = _dense_registry()
    slot = registry.slots["layers.0.self_attn.q_proj"]
    original_A = slot.lora_A.detach().clone()
    original_B = slot.lora_B.detach().clone()
    state = _dense_state()
    state[_key("layers.0.self_attn.q_proj", "A")] = torch.full((2, 1), 9.0)
    state[_key("layers.0.self_attn.q_proj", "B")].fill_(7.0)
    save_file(state, tmp_path / "adapter_model.safetensors")

    with pytest.raises(ValueError, match="has shape"):
        load_peft_adapter(registry, tmp_path)

    torch.testing.assert_close(slot.lora_A, original_A)
    torch.testing.assert_close(slot.lora_B, original_B)


def test_export_preserves_model_reference_before_resolution(tmp_path, monkeypatch) -> None:
    resolved_path = "/cache/models--example--base/snapshots/revision"
    monkeypatch.setattr(engine_api, "resolve_model_path", lambda _model: resolved_path)
    monkeypatch.setattr(engine_api, "config_from_hf", lambda _path: ModelConfig())

    def fake_init(self, config, **_kwargs):
        self.config = config

    monkeypatch.setattr(engine_api.ArenoEngine, "__init__", fake_init)
    engine = engine_api.ArenoEngine.from_pretrained(
        resolved_path,
        devices=[0],
        start=False,
        loss_fn=lambda _pack, logprobs: logprobs.sum(),
        runtime_config=RuntimeConfig(attn_backend="native", compile_model=False),
        base_model_name_or_path="example/base",
    )

    assert engine.config.model_path == resolved_path
    assert engine.config.base_model_name_or_path == "example/base"

    registry = _dense_registry()
    exported = export_peft_adapter(
        registry,
        tmp_path,
        base_model_name_or_path=engine.config.base_model_name_or_path,
    )
    adapter_config = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))

    assert exported == str(tmp_path)
    assert adapter_config["base_model_name_or_path"] == "example/base"
