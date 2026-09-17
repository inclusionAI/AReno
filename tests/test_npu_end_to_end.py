"""Opt-in Ascend SFT/rollout/checkpoint/HTTP acceptance against a local model.

Run in a fresh process, separately from single-process kernel tests, so the
coordinator has not already acquired a worker's NPU. No assets are downloaded.
"""

import math
import os
from pathlib import Path

import pytest

from areno import Trainer
from areno.api import NPU, NpuConfig, SamplingParams, TrainSequence
from areno.api.algorithms import sft_loss_fn


@pytest.mark.parametrize("optimizer", ["master", "8bit", "4bit"])
def test_sft_rollout_and_checkpoint_reload(tmp_path, optimizer):
    model = os.environ.get("ARENO_NPU_TEST_MODEL")
    if not model:
        pytest.skip("Set ARENO_NPU_TEST_MODEL to a local Ascend validation checkpoint")
    assert Path(model).is_dir(), "ARENO_NPU_TEST_MODEL must be a local checkpoint directory"
    world = int(os.environ.get("ARENO_NPU_TEST_WORLD_SIZE", "1"))
    tp = int(os.environ.get("ARENO_NPU_TEST_TP_SIZE", "1"))
    devices = [int(x) for x in os.environ.get("ARENO_NPU_TEST_DEVICES", ",".join(map(str, range(world)))).split(",")]
    rollout = os.environ.get("ARENO_NPU_TEST_ROLLOUT_DEVICES")
    config = NpuConfig(
        tp_size=tp,
        devices=devices,
        rollout_devices=[int(x) for x in rollout.split(",")] if rollout else None,
        rollout_tp_size=int(os.environ.get("ARENO_NPU_TEST_ROLLOUT_TP_SIZE", "1")) if rollout else None,
        optimizer={"adam_8bit": optimizer == "8bit", "adam_4bit": optimizer == "4bit"},
        runtime={"activation_checkpointing": True},
    )
    checkpoint = tmp_path / "checkpoint"
    trainer = Trainer(world, model, backend_type=NPU, custom_config=config)
    sampling = SamplingParams(greedy=True, max_new_tokens=3, ignore_eos=True)
    prompts = ["Hello", "One plus one is", "The sky is", "A small test"]
    try:
        trainer.init()
        tokenizer = trainer.get_tokenizer()
        tokens = [
            tokenizer.encode(text, add_special_tokens=True)
            for text in (
                "Hello, this is a training example.",
                "One plus one is two.",
                "The sky is blue.",
                "A small test checks training and inference.",
            )
        ]
        batch = [
            TrainSequence(tokens=row, prompt_mask=[False] * len(row), loss_mask=[True] * len(row)) for row in tokens
        ]
        metrics = trainer.train(batch, sft_loss_fn, mini_bs=len(batch))
        assert metrics and all(
            math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (float, int))
        )
        trainer.begin_rollout_session()
        try:
            before = trainer.rollout_batch(prompts, 1, sampling)
        finally:
            trainer.end_rollout_session()
        assert len(before) == len(prompts)
        assert all(result.sequences and len(result.sequences[0].resp_tokens) == 3 for result in before)
        trainer.save_checkpoint(str(checkpoint))
        assert list(checkpoint.glob("*.safetensors"))
    finally:
        trainer.close()
    # Reload through the same public checkpoint path and compare greedy output.
    restored = Trainer(world, str(checkpoint), backend_type=NPU, custom_config=config)
    try:
        restored.init()
        restored.begin_rollout_session()
        try:
            after = restored.rollout_batch(prompts, 1, sampling)
        finally:
            restored.end_rollout_session()
        assert [r.sequences[0].resp_tokens for r in after] == [r.sequences[0].resp_tokens for r in before]
    finally:
        restored.close()


def test_http_serve_greedy_generation_and_shutdown():
    model = os.environ.get("ARENO_NPU_TEST_MODEL")
    if not model:
        pytest.skip("Set ARENO_NPU_TEST_MODEL to a local Ascend validation checkpoint")
    assert Path(model).is_dir(), "ARENO_NPU_TEST_MODEL must be a local checkpoint directory"
    pytest.importorskip("httpx", reason="FastAPI HTTP acceptance requires httpx")
    from fastapi.testclient import TestClient

    from areno.cli.serve import create_app

    app = create_app(
        model_path=model,
        backend_type=NPU,
        tp_size=int(os.environ.get("ARENO_NPU_TEST_TP_SIZE", "1")),
        world_size=int(os.environ.get("ARENO_NPU_TEST_WORLD_SIZE", "1")),
        max_running_prompts=2,
        default_max_tokens=4,
        decode_progress_interval_s=0,
        chat_template_enable_thinking=False,
    )
    state = app.state.areno_serve
    engine = state.engine._engine
    processes = list(engine.cluster.processes)
    assert engine.config.role == "rollout"
    assert engine.config.runtime.device_type == "npu"
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/v1/models").json()["data"][0]["id"] == model
        request = {
            "model": model,
            "messages": [{"role": "user", "content": "Count from one to five."}],
            "max_tokens": 4,
            "temperature": 0,
        }
        first = client.post("/v1/chat/completions", json=request)
        assert first.status_code == 200, first.text
        result = first.json()
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["finish_reason"] in {"stop", "length"}
        assert 1 <= result["usage"]["completion_tokens"] <= 4
        assert result["usage"]["prompt_tokens"] > 0
        repeated = client.post("/v1/chat/completions", json=request)
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["choices"] == result["choices"]
        batched = client.post("/v1/chat/completions", json={**request, "n": 2})
        assert batched.status_code == 200, batched.text
        assert len(batched.json()["choices"]) == 2
        assert 2 <= batched.json()["usage"]["completion_tokens"] <= 8
    assert state.closing and not state.active_tasks
    assert all(not process.is_alive() for process in processes)
