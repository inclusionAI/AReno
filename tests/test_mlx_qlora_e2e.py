"""Opt-in upstream quantized model gates: ARENO_E2E_MLX_QLORA_SYNTHETIC=1."""

import importlib
import json
import os

import numpy as np
import pytest

from areno.adapters import LoraConfig
from areno.api import MlxConfig, TrainSequence, sft_loss_fn


@pytest.mark.parametrize("family", ["qwen3", "bailing_moe_v3"])
@pytest.mark.parametrize("compile_step,checkpointing", [(False, False), (False, True), (True, False), (True, True)])
def test_quantized_upstream_train_save_reload(monkeypatch, tmp_path, family, compile_step, checkpointing):
    if os.getenv("ARENO_E2E_MLX_QLORA_SYNTHETIC") != "1":
        pytest.skip("set ARENO_E2E_MLX_QLORA_SYNTHETIC=1")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model, quantize_model

    from areno.api.backend.mlx.backend import MlxBackend
    from areno.api.backend.mlx.provider import MlxModelProvider
    from areno.api.context import Context

    upstream = importlib.import_module(f"mlx_lm.models.{family}")
    layer_type = upstream.TransformerBlock if family == "qwen3" else upstream.BailingDecoderLayer
    monkeypatch.setattr(layer_type, "__call__", layer_type.__call__)
    mx.random.seed(42)
    common = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        head_dim=32,
        vocab_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_theta=10000,
        tie_word_embeddings=False,
    )
    if family == "qwen3":
        args = upstream.ModelArgs(model_type="qwen3", num_key_value_heads=1, **common)
        targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        architecture = "Qwen3ForCausalLM"
    else:
        args = upstream.ModelArgs(
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=32,
            num_experts=4,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            moe_intermediate_size=64,
            moe_shared_expert_intermediate_size=64,
            **common,
        )
        targets = (
            "q_proj",
            "k_proj",
            "v_proj",
            "f_proj",
            "o_proj",
            "g_proj",
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "dense",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        architecture = "BailingMoeV3ForCausalLM"
    model, config = quantize_model(upstream.Model(args), {**vars(args), "architectures": [architecture]}, 64, 4)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(base / "model.safetensors"), dict(tree_flatten(model.parameters())))
    original = {name: np.array(weight) for name, weight in tree_flatten(model.parameters())}
    tokens = mx.array([[i % 120 + 1 for i in range(40)]])  # Exercise sorted expert routing too.
    initial = np.array(model(tokens))
    tokenizer = type("Tokenizer", (), {"encode": lambda self, text, **kw: [1, 2], "eos_token_id": 0})()

    def initialize(adapter_path=None):
        policy, loaded_config = load_model(base)
        provider = MlxModelProvider(policy, tokenizer, None, loaded_config)
        monkeypatch.setattr("areno.api.backend.mlx.backend.load_provider", lambda *a, **kw: provider)
        backend = MlxBackend()
        ctx = Context(
            1,
            str(base),
            tokenizer,
            MlxConfig(
                lora=LoraConfig(rank=4, alpha=8, target_modules=targets, qlora=True, adapter_path=adapter_path),
                compile_train_step=compile_step,
                gradient_checkpointing=checkpointing,
                optimizer={"lr": 1e-3, "min_lr": 1e-3, "lr_decay_style": "constant", "weight_decay": 0.0},
            ),
        )
        backend.initialize(ctx)
        return backend, ctx

    def unchanged(policy):
        for name, weight in tree_flatten(policy.parameters()):
            if not name.endswith((".lora_a", ".lora_b")):
                np.testing.assert_array_equal(np.array(weight), original[name.replace(".linear.", ".")])

    row = TrainSequence(tokens=tokens[0].tolist(), prompt_mask=[True] + [False] * 39, eos_token_id=0)
    backend, ctx = initialize()
    try:
        np.testing.assert_allclose(np.array(backend.model(tokens)), initial, atol=1e-5, rtol=1e-4)
        assert len(backend._lora_state.slots) == (28 if family == "qwen3" else 44)
        if family != "qwen3":
            slots = backend._lora_state.slots
            assert isinstance(slots["layers.3.attention.q_a_proj"].linear, nn.Linear)
            assert isinstance(slots["layers.0.attention.g_proj"].linear, nn.QuantizedLinear)
            assert slots["layers.3.mlp.experts.{expert}.down_proj"].linear.bits == 4
        before = {name: np.array(slot.lora_b) for name, slot in backend._lora_state.slots.items()}
        for _ in range(2):
            ctx.step()
            metrics = backend.train(ctx, [row], sft_loss_fn, mini_bs=1)
            assert np.isfinite(metrics["loss"]) and np.isfinite(metrics["grad_norm"]) and metrics["grad_norm"] > 0
        assert all(
            not np.array_equal(before[name], np.array(slot.lora_b)) for name, slot in backend._lora_state.slots.items()
        )
        unchanged(backend.model)
        trained = backend.score_logprobs(ctx, "actor", [row.tokens], microbatch_size=1)
        exported = backend.save_checkpoint(ctx, str(tmp_path / "adapter"))
        assert LoraConfig(adapter_path=exported).qlora
        backend.model.eval()
        cache = make_prompt_cache(backend.model)
        incremental = mx.concatenate([backend.model(tokens[:, i : i + 1], cache=cache) for i in range(40)], axis=1)
        np.testing.assert_allclose(np.array(incremental), np.array(backend.model(tokens)), atol=1e-4, rtol=1e-4)
    finally:
        backend.close()

    backend, ctx = initialize(exported)
    try:
        np.testing.assert_allclose(
            backend.score_logprobs(ctx, "actor", [row.tokens], microbatch_size=1), trained, atol=1e-6, rtol=0
        )
        ctx.step()
        metrics = backend.train(ctx, [row], sft_loss_fn, mini_bs=1)
        assert np.isfinite(metrics["loss"]) and metrics["grad_norm"] > 0
        unchanged(backend.model)
    finally:
        backend.close()


@pytest.mark.parametrize("online", [False, True], ids=["preconverted", "online"])
def test_real_qwen_qlora_two_steps_and_reload(tmp_path, online):
    variable = "ARENO_E2E_MLX_QWEN_ONLINE_MODEL" if online else "ARENO_E2E_MLX_QWEN_QLORA_MODEL"
    path = os.getenv(variable)
    if not path:
        pytest.skip(f"set {variable} to a local Qwen3 checkpoint")
    import mlx.core as mx

    from areno import Trainer
    from areno.api import MLX, SamplingParams
    from tests.test_mlx_bailing_v3_lora_e2e import _assert_step, _base_digest

    def trainer(adapter_path=None):
        return Trainer(
            1,
            path,
            backend_type=MLX,
            custom_config=MlxConfig(
                lora=LoraConfig(qlora=not online, rank=8, alpha=16, adapter_path=adapter_path),
                # Reload online quantization from the exported adapter metadata.
                quantization_bits=4 if online and adapter_path is None else None,
                compile_train_step=False,
                gradient_checkpointing=True,
                optimizer={"lr": 1e-3, "min_lr": 1e-3, "lr_decay_style": "constant", "weight_decay": 0.0},
            ),
        )

    report = {"model_path": path, "online": online, "tokens": 512, "steps": [], "status": "running"}
    mx.random.seed(42)
    active = trainer()
    try:
        active.init()
        original = _base_digest(active._backend.model, mx)
        report["adapter_modules"] = len(active._backend._lora_state.slots)
        report["quantized_modules"] = (
            active._backend.model_config["areno_online_quantization"]["modules"]
            if online
            else len(active._backend._lora_state.quantization["layers"])
        )
        active.begin_rollout_session()
        try:
            result = active.rollout_batch(
                ["The capital of France is"], 1, SamplingParams(temperature=0, max_new_tokens=32)
            )
        finally:
            active.end_rollout_session()
        sequence = result[0].sequences[0]
        assert sequence.resp_tokens and all(np.isfinite(sequence.resp_logprobs))
        report["generation"] = active.get_tokenizer().decode(sequence.resp_tokens)
        seed = active.get_tokenizer().encode("The capital of France is Paris. ", add_special_tokens=False)
        tokens = (seed * (512 // len(seed) + 1))[:512]
        row = TrainSequence(
            tokens=tokens,
            prompt_mask=[True] + [False] * 511,
            eos_token_id=int(active.get_tokenizer().eos_token_id or 0),
        )
        for _ in range(2):
            metrics = active.train([row], sft_loss_fn, mini_bs=1)
            _assert_step(metrics)
            report["steps"].append(metrics)
        assert _base_digest(active._backend.model, mx) == original
        trained = active._backend.score_logprobs(active._ctx, "actor", [tokens], microbatch_size=1)
        exported = active.export_adapter(str(tmp_path / "adapter"))
        active.close()
        active = trainer(exported)
        active.init()
        reloaded = active._backend.score_logprobs(active._ctx, "actor", [tokens], microbatch_size=1)
        report["reload_max_abs_error"] = float(np.max(np.abs(np.asarray(trained) - np.asarray(reloaded))))
        np.testing.assert_allclose(reloaded, trained, atol=1e-6, rtol=0)
        report["continued_step"] = active.train([row], sft_loss_fn, mini_bs=1)
        _assert_step(report["continued_step"])
        assert _base_digest(active._backend.model, mx) == original
        report["status"] = "passed"
    finally:
        if report["status"] == "running":
            report["status"] = "failed"
        report["peak_bytes"] = mx.get_peak_memory()
        (tmp_path / "qwen-validation.json").write_text(json.dumps(report, indent=2))
        active.close()


@pytest.mark.parametrize("online", [False, True], ids=["preconverted", "online"])
def test_real_qwen_quantized_inference_without_adapter(tmp_path, online):
    variable = "ARENO_E2E_MLX_QWEN_ONLINE_MODEL" if online else "ARENO_E2E_MLX_QWEN_QLORA_MODEL"
    path = os.getenv(variable)
    if not path:
        pytest.skip(f"set {variable} to a local Qwen3 checkpoint")

    from areno import Trainer
    from areno.api import MLX, SamplingParams

    trainer = Trainer(
        1,
        path,
        backend_type=MLX,
        custom_config=MlxConfig(
            inference_only=True,
            gradient_checkpointing=False,
            quantization_bits=4 if online else None,
        ),
    )
    try:
        trainer.init()
        assert trainer._backend._lora_state is None
        assert trainer._backend.model_config["quantization"]["bits"] == 4
        trainer.begin_rollout_session()
        try:
            result = trainer.rollout_batch(
                ["The capital of France is"], 1, SamplingParams(temperature=0, max_new_tokens=32)
            )
        finally:
            trainer.end_rollout_session()
        sequence = result[0].sequences[0]
        assert sequence.resp_tokens and all(np.isfinite(sequence.resp_logprobs))
        (tmp_path / "qwen-inference.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "online": online,
                    "tokens": len(sequence.resp_tokens),
                    "generation": trainer.get_tokenizer().decode(sequence.resp_tokens),
                },
                indent=2,
            )
        )
    finally:
        trainer.close()
