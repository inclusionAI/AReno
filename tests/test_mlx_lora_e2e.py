"""Opt-in Apple Silicon E2E for Qwen3 MLX LoRA training."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from areno.adapters import LoraConfig
from areno.api import MlxConfig, Trainer, TrainSequence, sft_loss_fn


def _require_mlx_device(mx) -> None:
    try:
        probe = mx.zeros((1,))
        mx.eval(probe)
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip("MLX Metal device is unavailable in this environment")
        raise


def test_mlx_lora_backend_loads_peft_and_runs_two_synthetic_sft_steps(monkeypatch, tmp_path) -> None:
    if os.getenv("ARENO_E2E_MLX_SYNTHETIC") != "1":
        pytest.skip("set ARENO_E2E_MLX_SYNTHETIC=1 to run the Apple Silicon synthetic MLX LoRA E2E")
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    from mlx.utils import tree_flatten
    from safetensors.numpy import load_file, save_file

    from areno.api.backend.mlx.backend import MlxBackend
    from areno.api.backend.mlx.provider import MlxModelProvider

    _require_mlx_device(mx)

    class SelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.v_proj = nn.Linear(8, 8, bias=False)

        def __call__(self, hidden_states):
            return self.q_proj(hidden_states) + self.v_proj(hidden_states)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = SelfAttention()

        def __call__(self, hidden_states):
            return self.self_attn(hidden_states)

    class Body(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 8)
            self.layers = [Block()]

        def __call__(self, input_ids):
            hidden_states = self.embed_tokens(input_ids)
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Body()
            self.lm_head = nn.Linear(8, 16, bias=False)

        def __call__(self, input_ids):
            return self.lm_head(self.model(input_ids))

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, *, add_special_tokens):
            del text, add_special_tokens
            return [1, 2, 3]

    model = Model()
    base_weights = [(name, mx.array(value)) for name, value in tree_flatten(model.parameters())]
    mx.eval(*(value for _, value in base_weights))
    tokenizer = Tokenizer()
    provider = MlxModelProvider(model, tokenizer, None, {"model_type": "qwen3"})
    monkeypatch.setattr("areno.api.backend.mlx.backend.load_provider", lambda *_args, **_kwargs: provider)
    adapter_config = {
        "base_model_name_or_path": "synthetic",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": 2,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
    adapter_tensors = {}
    for index, target in enumerate(adapter_config["target_modules"], start=1):
        prefix = f"base_model.model.model.layers.0.self_attn.{target}"
        adapter_tensors[f"{prefix}.lora_A.weight"] = np.full((2, 8), index / 10, dtype=np.float32)
        adapter_tensors[f"{prefix}.lora_B.weight"] = np.full((8, 2), index / 20, dtype=np.float32)
    save_file(adapter_tensors, tmp_path / "adapter_model.safetensors")
    config = MlxConfig(
        lora=LoraConfig(adapter_path=os.fspath(tmp_path)),
        optimizer={
            "lr": 1.0e-2,
            "min_lr": 1.0e-2,
            "lr_decay_style": "constant",
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
        },
        compile_train_step=False,
        gradient_checkpointing=False,
    )
    ctx = type(
        "Context",
        (),
        {"world_size": 1, "custom_config": config, "model_path": "synthetic", "tokenizer": tokenizer, "global_step": 0},
    )()
    backend = MlxBackend()
    backend.initialize(ctx)
    try:
        assert backend._lora_state is not None
        slot = backend._lora_state.slots["layers.0.self_attn.q_proj"]
        mx.eval(slot.linear.weight, slot.lora_a, slot.lora_b)
        initial_base = np.array(slot.linear.weight)
        initial_a = np.array(slot.lora_a)
        initial_b = np.array(slot.lora_b)
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        np.testing.assert_array_equal(initial_a, adapter_tensors[f"{prefix}.lora_A.weight"].T)
        np.testing.assert_array_equal(initial_b, adapter_tensors[f"{prefix}.lora_B.weight"].T)
        row = TrainSequence(tokens=[1, 2, 3, 4], prompt_mask=[True, False, False, False], eos_token_id=0)

        results = []
        for step in range(2):
            ctx.global_step = step
            results.append(backend.train(ctx, [row], sft_loss_fn, mini_bs=1))

        trained_logprobs = backend.score_logprobs(ctx, "actor", [[1, 2, 3, 4]], microbatch_size=1)
        mx.eval(slot.linear.weight, slot.lora_a, slot.lora_b)
        final_base = np.array(slot.linear.weight)
        final_a = np.array(slot.lora_a)
        final_b = np.array(slot.lora_b)
        exported_path = tmp_path / "exported"
        assert backend.save_checkpoint(ctx, os.fspath(exported_path)) == os.fspath(exported_path)
    finally:
        backend.close()

    exported = load_file(exported_path / "adapter_model.safetensors")
    np.testing.assert_array_equal(exported[f"{prefix}.lora_A.weight"], final_a.T.astype(np.float32))
    np.testing.assert_array_equal(exported[f"{prefix}.lora_B.weight"], final_b.T.astype(np.float32))
    np.testing.assert_array_equal(final_base, initial_base)
    assert not np.array_equal(final_b, initial_b)
    assert not np.array_equal(final_a, initial_a)
    assert all(np.isfinite(result["loss"]) for result in results)
    assert all(np.isfinite(result["grad_norm"]) for result in results)

    reloaded_model = Model()
    reloaded_model.load_weights(base_weights)
    reloaded_provider = MlxModelProvider(reloaded_model, tokenizer, None, {"model_type": "qwen3"})
    monkeypatch.setattr("areno.api.backend.mlx.backend.load_provider", lambda *_args, **_kwargs: reloaded_provider)
    reloaded_config = MlxConfig(
        lora=LoraConfig(adapter_path=os.fspath(exported_path)),
        optimizer=config.optimizer,
        compile_train_step=False,
        gradient_checkpointing=False,
    )
    reloaded_ctx = type(
        "Context",
        (),
        {
            "world_size": 1,
            "custom_config": reloaded_config,
            "model_path": "synthetic",
            "tokenizer": tokenizer,
            "global_step": 0,
        },
    )()
    reloaded_backend = MlxBackend()
    reloaded_backend.initialize(reloaded_ctx)
    try:
        assert reloaded_backend._lora_state is not None
        reloaded_slot = reloaded_backend._lora_state.slots["layers.0.self_attn.q_proj"]
        mx.eval(reloaded_slot.lora_a, reloaded_slot.lora_b)
        np.testing.assert_array_equal(np.array(reloaded_slot.lora_a), final_a.astype(np.float32))
        np.testing.assert_array_equal(np.array(reloaded_slot.lora_b), final_b.astype(np.float32))
        reloaded_logprobs = reloaded_backend.score_logprobs(
            reloaded_ctx,
            "actor",
            [[1, 2, 3, 4]],
            microbatch_size=1,
        )
    finally:
        reloaded_backend.close()

    np.testing.assert_allclose(reloaded_logprobs, trained_logprobs, rtol=0.0, atol=1.0e-6)


def test_mlx_qwen3_lora_runs_two_sft_steps() -> None:
    model_path_value = os.getenv("ARENO_E2E_MLX_QWEN3_MODEL")
    if not model_path_value:
        pytest.skip("set ARENO_E2E_MLX_QWEN3_MODEL to run the Apple Silicon MLX LoRA E2E")
    model_path = Path(model_path_value)
    if not model_path.is_dir():
        pytest.fail(f"ARENO_E2E_MLX_QWEN3_MODEL is not a local checkpoint directory: {model_path}")

    import mlx.core as mx

    _require_mlx_device(mx)

    config = MlxConfig(
        lora=LoraConfig(rank=2, alpha=4, target_modules=("q_proj", "v_proj", "up_proj")),
        optimizer={
            "lr": 1.0e-3,
            "min_lr": 1.0e-3,
            "lr_decay_style": "constant",
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
        },
        compile_train_step=False,
        gradient_checkpointing=False,
    )
    trainer = Trainer(1, os.fspath(model_path), custom_config=config)
    trainer.init()
    try:
        tokenizer = trainer.get_tokenizer()
        tokens = list(tokenizer.encode("MLX LoRA should learn this short sequence.", add_special_tokens=True))
        if len(tokens) < 2:
            pytest.fail("Qwen3 tokenizer returned fewer than two tokens for the MLX LoRA E2E prompt")
        row = TrainSequence(
            tokens=tokens,
            prompt_mask=[True, *([False] * (len(tokens) - 1))],
            eos_token_id=int(tokenizer.eos_token_id or 0),
        )

        backend = trainer._backend
        assert backend is not None
        state = backend._lora_state
        assert state is not None
        initial_arrays = [array for slot in state.slots.values() for array in (slot.lora_a, slot.lora_b)]
        mx.eval(*initial_arrays)
        initial = [np.array(array) for array in initial_arrays]

        results = [trainer.train([row], sft_loss_fn, mini_bs=1) for _ in range(2)]
        final_arrays = [array for slot in state.slots.values() for array in (slot.lora_a, slot.lora_b)]
        mx.eval(*final_arrays)
        final = [np.array(array) for array in final_arrays]
    finally:
        trainer.close()

    assert all(np.isfinite(result["loss"]) for result in results)
    assert all(np.isfinite(result["grad_norm"]) for result in results)
    assert any(not np.array_equal(before, after) for before, after in zip(initial, final, strict=True))
