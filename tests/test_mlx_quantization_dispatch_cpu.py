"""Keep online LoRA, explicit QLoRA, and quantized inference independent."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from areno.adapters import LoraConfig
from areno.api.backend.mlx.backend import MlxBackend
from areno.api.config import MlxConfig


@pytest.fixture
def runtime(monkeypatch):
    core = ModuleType("mlx.core")
    core.metal = SimpleNamespace(is_available=lambda: False)
    core.eval = lambda *args: None
    core.clear_cache = lambda: None
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr("areno.api.backend.mlx.gated_delta.install_parallel_gated_delta", lambda: None)
    tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
    provider = SimpleNamespace(
        model=SimpleNamespace(parameters=lambda: {}, train=lambda: None),
        tokenizer=tokenizer,
        processor=None,
        # This deliberately exercises a family and group size outside strict QLoRA.
        config={"model_type": "arbitrary", "quantization": {"bits": 4, "group_size": 32}},
        is_multimodal=False,
        configure_trainability=lambda config: None,
        optimizer_state_precision=lambda *args: "fp32",
    )
    calls = []

    def load(path, **kwargs):
        calls.append(kwargs)
        return provider

    monkeypatch.setattr("areno.api.backend.mlx.backend.load_provider", load)
    monkeypatch.setattr(
        "areno.api.backend.mlx.backend.build_optimizer", lambda *a, **kw: (SimpleNamespace(state={}), [])
    )
    monkeypatch.setattr(
        "areno.api.backend.mlx.backend.initialize_lora",
        lambda *a, **kw: SimpleNamespace(quantization=kw["quantization"]),
    )
    return tokenizer, calls


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("inference,with_lora", [(True, False), (True, True), (False, True), (False, False)])
def test_quantized_backend_preserves_online_and_inference_paths(runtime, monkeypatch, online, inference, with_lora):
    tokenizer, calls = runtime
    monkeypatch.setattr(
        "areno.api.backend.mlx.backend.validate_quantized_base",
        lambda *a: pytest.fail("ordinary LoRA and inference must not invoke strict QLoRA validation"),
    )
    config = MlxConfig(
        lora=LoraConfig() if with_lora else None,
        inference_only=inference,
        quantization_bits=4 if online else None,
        quantization_group_size=32,
        gradient_checkpointing=False,
    )
    ctx = SimpleNamespace(world_size=1, custom_config=config, model_path="base", tokenizer=tokenizer)
    backend = MlxBackend()
    try:
        if not inference and not with_lora:
            with pytest.raises(ValueError, match="full quantized-weight training is unsupported"):
                backend.initialize(ctx)
        else:
            backend.initialize(ctx)
            if with_lora:
                assert backend._lora_state.quantization is None
        assert calls == [{"adapter_path": None, **({"quantization": {"bits": 4, "group_size": 32}} if online else {})}]
    finally:
        backend.close()


def test_explicit_qlora_still_validates_base_before_injection(runtime, monkeypatch):
    tokenizer, _ = runtime
    descriptor = {"format": "mlx-affine-v1"}
    validated = []

    def validate(model, config, lora, path):
        validated.append((lora.qlora, path))
        return descriptor

    monkeypatch.setattr("areno.api.backend.mlx.backend.validate_quantized_base", validate)
    config = MlxConfig(lora=LoraConfig(qlora=True), gradient_checkpointing=False)
    ctx = SimpleNamespace(world_size=1, custom_config=config, model_path="base", tokenizer=tokenizer)
    backend = MlxBackend()
    try:
        backend.initialize(ctx)
        assert validated == [(True, "base")]
        assert backend._lora_state.quantization is descriptor
    finally:
        backend.close()


def test_explicit_qlora_rejects_online_conversion_before_loading(runtime):
    tokenizer, calls = runtime
    config = MlxConfig(lora=LoraConfig(qlora=True), quantization_bits=4)
    ctx = SimpleNamespace(world_size=1, custom_config=config, model_path="base", tokenizer=tokenizer)
    with pytest.raises(ValueError, match="cannot combine with online quantization"):
        MlxBackend().initialize(ctx)
    assert not calls
