"""Opt-in Ling MLX numerical and real-checkpoint LoRA validation.

Set ARENO_E2E_MLX_BAILING_SYNTHETIC=1 for reduced upstream models, or
ARENO_E2E_MLX_BAILING_MODEL to a local Ling checkpoint (set ARENO_E2E_MLX_QLORA=1 for a quantized base).
Set ARENO_E2E_MLX_BAILING_METRICS_DIR to also register the real-model run
with the dashboard and record training/continuation metrics.
No test downloads weights. These gates are intentionally separate from CPU contracts.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import resource
from pathlib import Path

import numpy as np
import pytest

from areno import Trainer
from areno.adapters import LoraConfig
from areno.api import MLX, MlxConfig, TrainSequence, sft_loss_fn

TARGETS = ("q_proj", "k_proj", "v_proj", "f_proj", "o_proj", "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "dense")
EXPANDED_TARGETS = TARGETS + ("g_proj", "gate_proj", "up_proj", "down_proj")


def _runtime():
    try:
        import mlx.core as mx
    except ImportError as exc:
        pytest.skip(f"MLX runtime unavailable: {exc}")
    if os.getenv("ARENO_E2E_MLX_DEVICE") == "cpu":
        mx.set_default_device(mx.cpu)
    try:
        module = importlib.import_module("mlx_lm.models.bailing_moe_v3")
    except ImportError as exc:
        pytest.fail(f"Ling tests require MLX-LM with bailing_moe_v3 support: {exc}")
    return mx, module


def _config(*, adapter_path=None, compile_step=False, checkpointing=False, targets=TARGETS, qlora=False):
    return MlxConfig(
        lora=LoraConfig(rank=8, alpha=16, target_modules=targets, adapter_path=adapter_path, qlora=qlora),
        optimizer={"lr": 1e-3, "min_lr": 1e-3, "lr_decay_style": "constant", "weight_decay": 0.0},
        compile_train_step=compile_step,
        gradient_checkpointing=checkpointing,
    )


def _base_digest(model, mx):
    """Hash frozen arrays in small chunks without duplicating the full base."""
    from mlx.utils import tree_flatten

    result = {}
    for name, value in tree_flatten(model.parameters()):
        if name.endswith((".lora_a", ".lora_b")):
            continue
        digest = hashlib.sha256()
        flat = value.reshape(-1)
        for start in range(0, value.size, 1 << 20):
            part = flat[start : start + (1 << 20)]
            if value.dtype != mx.uint32:
                part = part.astype(mx.float32)
            mx.eval(part)
            digest.update(np.array(part).tobytes())
        result[name.replace(".linear.", ".")] = (tuple(value.shape), str(value.dtype), digest.hexdigest())
    return result


def _assert_step(metrics):
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["grad_norm"]) and metrics["grad_norm"] > 0


@pytest.mark.parametrize("token_count", [3, 40])
@pytest.mark.parametrize("quantized", [False, True])
def test_routed_mlp_matches_explicit_expert_updates(token_count, quantized):
    if os.getenv("ARENO_E2E_MLX_BAILING_SYNTHETIC") != "1":
        pytest.skip("set ARENO_E2E_MLX_BAILING_SYNTHETIC=1")
    mx, _ = _runtime()
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.switch_layers import SwitchGLU
    from mlx_lm.tuner.lora import LoRASwitchLinear

    mx.random.seed(11)
    width = 64 if quantized else 8
    mlp = SwitchGLU(width, 128 if quantized else 12, 4)
    mlp.freeze()
    for name in ("gate_proj", "up_proj", "down_proj"):
        base = getattr(mlp, name)
        if quantized:
            base = base.to_quantized(group_size=64, bits=4)
        slot = LoRASwitchLinear.from_base(base, r=2, scale=2)
        slot.lora_b = mx.random.normal(slot.lora_b.shape) * 0.05
        setattr(mlp, name, slot)
    x = mx.random.normal((1, token_count, width))
    indices = mx.array([[[i % 3, (i + 1) % 3] for i in range(token_count)]])

    def reference(value):
        def project(name, inputs):
            slot = getattr(mlp, name)
            weight = slot.linear.weight
            if quantized:
                weight = mx.dequantize(weight, slot.linear.scales, slot.linear.biases, group_size=64, bits=4)
            weight = weight + slot.scale * (slot.lora_b @ slot.lora_a)
            selected = weight[indices]
            return mx.squeeze(mx.expand_dims(inputs, -2) @ selected.swapaxes(-1, -2), -2)

        inputs = mx.broadcast_to(value[:, :, None, :], (1, token_count, 2, width))
        gate, up = project("gate_proj", inputs), project("up_proj", inputs)
        return project("down_proj", nn.silu(gate) * up)

    np.testing.assert_allclose(np.array(mlp(x, indices)), np.array(reference(x)), atol=1e-5, rtol=1e-4)
    actual = mx.grad(lambda value: mx.sum(mlp(value, indices) ** 2))(x)
    expected = mx.grad(lambda value: mx.sum(reference(value) ** 2))(x)
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-5, rtol=1e-4)
    _, gradients = nn.value_and_grad(mlp, lambda model: mx.sum(model(x, indices) ** 2))(mlp)
    for _, grad in tree_flatten(gradients):
        assert bool(mx.all(mx.isfinite(grad)).item())
        assert float(mx.sum(mx.abs(grad[:3])).item()) > 0
        np.testing.assert_array_equal(np.array(grad[3]), 0)


@pytest.mark.parametrize("compile_step,checkpointing", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("targets", [TARGETS, EXPANDED_TARGETS], ids=["attention", "expanded"])
def test_reduced_ling_train_export_reload(monkeypatch, tmp_path, compile_step, checkpointing, targets):
    if os.getenv("ARENO_E2E_MLX_BAILING_SYNTHETIC") != "1":
        pytest.skip("set ARENO_E2E_MLX_BAILING_SYNTHETIC=1")
    mx, upstream = _runtime()
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from areno.api.backend.mlx.backend import MlxBackend
    from areno.api.backend.mlx.provider import MlxModelProvider
    from areno.api.context import Context

    # Upstream checkpointing patches a class globally; isolate test cases.
    monkeypatch.setattr(upstream.BailingDecoderLayer, "__call__", upstream.BailingDecoderLayer.__call__)
    args = upstream.ModelArgs(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        head_dim=8,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=16,
    )
    mx.random.seed(42)
    model = upstream.Model(args)
    tokens = mx.array([[1, 2, 3, 4, 5]])
    model.train()
    initial_logits = np.array(model(tokens))
    original = _base_digest(model, mx)
    weights = list(tree_flatten(model.parameters()))
    tokenizer = type("Tokenizer", (), {"encode": lambda self, text, **kw: [1, 2], "eos_token_id": 0})()
    model_config = {**vars(args), "architectures": ["BailingMoeV3ForCausalLM"]}

    def make_backend(policy, config):
        provider = MlxModelProvider(policy, tokenizer, None, model_config)
        monkeypatch.setattr("areno.api.backend.mlx.backend.load_provider", lambda *a, **kw: provider)
        ctx = Context(1, "synthetic", tokenizer, config)
        backend = MlxBackend()
        try:
            backend.initialize(ctx)
        except Exception:
            backend.close()
            raise
        return backend, ctx

    backend, ctx = make_backend(model, _config(compile_step=compile_step, checkpointing=checkpointing, targets=targets))
    row = TrainSequence(tokens=[1, 2, 3, 4, 5], prompt_mask=[True, True, False, False, False], eos_token_id=0)
    try:
        np.testing.assert_allclose(np.array(model(tokens)), initial_logits, atol=1e-5, rtol=1e-4)
        assert len(backend._lora_state.slots) == (44 if targets == EXPANDED_TARGETS else 19)
        # Each wrapped projection must match an explicit dense low-rank update,
        # including input gradients (the base stays frozen, its input does not).
        for name, slot in backend._lora_state.slots.items():
            slot.lora_b = mx.random.normal(slot.lora_b.shape) * 0.01
            if name in backend._lora_state.expert_slots:
                # Packed expert output/gradient semantics are covered below.
                continue
            x = mx.random.normal((2, slot.lora_a.shape[0]))
            weight = slot.linear.weight + slot.scale * (slot.lora_b.T @ slot.lora_a.T)

            def reference(value):
                return value @ weight.T

            np.testing.assert_allclose(np.array(slot(x)), np.array(reference(x)), atol=1e-5, rtol=1e-4)
            actual_grad = mx.grad(lambda value: mx.sum(slot(value) ** 2))(x)
            reference_grad = mx.grad(lambda value: mx.sum(reference(value) ** 2))(x)
            np.testing.assert_allclose(np.array(actual_grad), np.array(reference_grad), atol=1e-5, rtol=1e-4)

        # Selected early adapters must receive gradients through later frozen MoE.
        def loss(policy):
            return nn.losses.cross_entropy(policy(tokens)[:, :-1], tokens[:, 1:]).mean()

        _, gradients = nn.value_and_grad(model, loss)(model)
        flat_gradients = dict(tree_flatten(gradients))
        for index, name in ((0, "v_proj"), (3, "q_a_proj")):
            grad = flat_gradients[f"model.layers.{index}.attention.{name}.lora_b"]
            assert bool(mx.all(mx.isfinite(grad)).item())
            assert float(mx.sum(mx.abs(grad)).item()) > 0
        if targets == EXPANDED_TARGETS:
            for path in (
                "0.attention.g_proj",
                "3.attention.g_proj",
                "0.mlp.gate_proj",
                "3.mlp.shared_experts.up_proj",
                "3.mlp.switch_mlp.down_proj",
            ):
                grad = flat_gradients[f"model.layers.{path}.lora_b"]
                assert bool(mx.all(mx.isfinite(grad)).item())
                assert float(mx.sum(mx.abs(grad)).item()) > 0
        before = {name: np.array(slot.lora_b) for name, slot in backend._lora_state.slots.items()}
        for _ in range(2):
            ctx.step()
            _assert_step(backend.train(ctx, [row], sft_loss_fn, mini_bs=1))
        assert any(
            not np.array_equal(before[name], np.array(slot.lora_b)) for name, slot in backend._lora_state.slots.items()
        )
        assert _base_digest(model, mx) == original
        trained = backend.score_logprobs(ctx, "actor", [row.tokens], microbatch_size=1)
        exported = backend.save_checkpoint(ctx, str(tmp_path / "adapter"))
        # Prefill and token-at-a-time decode must agree after LoRA updates.
        model.eval()
        full = np.array(model(tokens))
        cache = model.make_cache()
        incremental = mx.concatenate([model(tokens[:, i : i + 1], cache=cache) for i in range(tokens.shape[1])], axis=1)
        np.testing.assert_allclose(np.array(incremental), full, atol=1e-5, rtol=1e-4)
    finally:
        backend.close()

    reloaded = upstream.Model(args)
    reloaded.load_weights(weights)
    backend, ctx = make_backend(
        reloaded, _config(adapter_path=exported, compile_step=compile_step, checkpointing=checkpointing)
    )
    try:
        np.testing.assert_allclose(
            backend.score_logprobs(ctx, "actor", [row.tokens], microbatch_size=1), trained, atol=1e-6, rtol=0
        )
        ctx.step()
        _assert_step(backend.train(ctx, [row], sft_loss_fn, mini_bs=1))
        assert _base_digest(reloaded, mx) == original
    finally:
        backend.close()


def test_real_ling_two_steps_and_reload(tmp_path, request):
    value = os.getenv("ARENO_E2E_MLX_BAILING_MODEL")
    if not value:
        pytest.skip("set ARENO_E2E_MLX_BAILING_MODEL to a local non-quantized checkpoint")
    path = Path(value)
    assert path.is_dir(), f"model directory does not exist: {path}"
    mx, _ = _runtime()
    steps = int(os.getenv("ARENO_E2E_MLX_BAILING_STEPS", "2"))
    assert steps >= 2
    length = int(os.getenv("ARENO_E2E_MLX_BAILING_TOKENS", "128"))
    assert length >= 2
    report = {"model_path": str(path), "tokens": length, "steps": [], "memory": [], "status": "running"}

    def finish_report():
        if report["status"] == "running":
            report["status"] = "failed"
        report["final_peak_bytes"] = mx.get_peak_memory()
        (tmp_path / "ling-validation.json").write_text(json.dumps(report, indent=2))

    request.addfinalizer(finish_report)
    recorder = None
    metrics_dir = os.getenv("ARENO_E2E_MLX_BAILING_METRICS_DIR")
    if metrics_dir:
        from areno.api.metrics import MetricsRecorder
        from areno.cli.dashboard_registry import register_dashboard_job

        recorder = MetricsRecorder(metrics_dir)

        def finish_recording():
            try:
                recorder.record_dashboard_state(
                    stage="validation_passed" if report["status"] == "passed" else "validation_failed",
                    status="succeeded" if report["status"] == "passed" else "failed",
                    step=len(report["steps"]) + int("continued_step" in report),
                )
            finally:
                recorder.close()

        request.addfinalizer(finish_recording)
        register_dashboard_job(
            kind="train",
            name=f"Ling MLX validation {length} tokens x {steps} steps + reload",
            metrics_dir=metrics_dir,
            config={
                "algo": "sft",
                "ckpt": str(path),
                "tokens": length,
                "max_steps": steps,
                "mini_bs": 1,
                "lora_rank": 8,
                "compile_train_step": False,
                "gradient_checkpointing": True,
            },
        )

    def record(phase):
        report["memory"].append(
            {
                "phase": phase,
                "mlx_active_bytes": mx.get_active_memory(),
                "mlx_peak_bytes": mx.get_peak_memory(),
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        (tmp_path / "ling-validation.json").write_text(json.dumps(report, indent=2))
        if recorder is not None:
            recorder.record_dashboard_state(stage=phase, step=len(report["steps"]) + int("continued_step" in report))

    mx.random.seed(42)
    targets = EXPANDED_TARGETS if os.getenv("ARENO_E2E_MLX_BAILING_EXPANDED") == "1" else TARGETS
    report["targets"] = list(targets)
    qlora = os.getenv("ARENO_E2E_MLX_QLORA") == "1"
    report["qlora"] = qlora
    trainer = Trainer(
        1, str(path), backend_type=MLX, custom_config=_config(checkpointing=True, targets=targets, qlora=qlora)
    )
    try:
        trainer.init()
        record("initialized")
        backend = trainer._backend
        assert len(backend._lora_state.slots) == (279 if targets == EXPANDED_TARGETS else 114)
        original = _base_digest(backend.model, mx)
        if qlora:
            from areno.api import SamplingParams

            report["quantized_modules"] = len(backend._lora_state.quantization["layers"])
            trainer.begin_rollout_session()
            try:
                generated = trainer.rollout_batch(
                    ["The capital of France is"], 1, SamplingParams(temperature=0, max_new_tokens=32)
                )
            finally:
                trainer.end_rollout_session()
            sequence = generated[0].sequences[0]
            assert sequence.resp_tokens and all(np.isfinite(sequence.resp_logprobs))
            report["generation"] = trainer.get_tokenizer().decode(sequence.resp_tokens)
            record("generated")
        seed_tokens = trainer.get_tokenizer().encode("The capital of France is Paris. ", add_special_tokens=False)
        assert seed_tokens
        # Purpose-built bounded synthetic text, not truncation of a user dataset.
        tokens = (list(seed_tokens) * (length // len(seed_tokens) + 1))[:length]
        row = TrainSequence(
            tokens=tokens,
            prompt_mask=[True] + [False] * (length - 1),
            eos_token_id=int(trainer.get_tokenizer().eos_token_id or 0),
        )
        before = {name: np.array(slot.lora_b) for name, slot in backend._lora_state.slots.items()}
        for step in range(steps):
            metrics = trainer.train([row], sft_loss_fn, mini_bs=1)
            _assert_step(metrics)
            report["steps"].append(metrics)
            if recorder is not None:
                recorder.record_train_step(step=step + 1, train_result=metrics, train_batch=[row])
            record(f"step_{step + 1}")
        assert _base_digest(backend.model, mx) == original
        assert any(
            not np.array_equal(before[name], np.array(slot.lora_b)) for name, slot in backend._lora_state.slots.items()
        )
        trained = backend.score_logprobs(trainer._ctx, "actor", [tokens], microbatch_size=1)
        exported = trainer.export_adapter(str(tmp_path / "adapter"))
        record("exported")
    finally:
        trainer.close()
        del trainer
        gc.collect()
        mx.clear_cache()
        (tmp_path / "ling-validation.json").write_text(json.dumps(report, indent=2))

    trainer = Trainer(1, str(path), backend_type=MLX, custom_config=_config(adapter_path=exported, checkpointing=True))
    try:
        trainer.init()
        record("reloaded")
        reloaded = trainer._backend.score_logprobs(trainer._ctx, "actor", [tokens], microbatch_size=1)
        report["reload_max_abs_error"] = float(np.max(np.abs(np.asarray(reloaded) - np.asarray(trained))))
        np.savez(tmp_path / "reload-logprobs.npz", tokens=tokens, trained=trained, reloaded=reloaded)
        np.testing.assert_allclose(reloaded, trained, atol=1e-6, rtol=0)
        metrics = trainer.train([row], sft_loss_fn, mini_bs=1)
        _assert_step(metrics)
        report["continued_step"] = metrics
        if recorder is not None:
            recorder.record_train_step(step=steps + 1, train_result=metrics, train_batch=[row])
        assert _base_digest(trainer._backend.model, mx) == original
        report["status"] = "passed"
        record("continued")
    finally:
        trainer.close()
        (tmp_path / "ling-validation.json").write_text(json.dumps(report, indent=2))
