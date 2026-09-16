"""Opt-in Gaudi SFT/rollout/checkpoint acceptance against a local model.

Run in a fresh process, separately from single-process kernel tests, so the
coordinator has not already acquired a worker's HPU. No assets are downloaded.
"""

import math
import os
from pathlib import Path

import pytest

from areno import Trainer
from areno.api import HPU, HpuConfig, SamplingParams, TrainSequence
from areno.api.algorithms import sft_loss_fn


@pytest.mark.parametrize("optimizer", ["master", "8bit", "4bit"])
def test_sft_rollout_and_checkpoint_reload(tmp_path, optimizer):
    model = os.environ.get("ARENO_HPU_TEST_MODEL")
    if not model:
        pytest.skip("Set ARENO_HPU_TEST_MODEL to a local Gaudi validation checkpoint")
    assert Path(model).is_dir(), "ARENO_HPU_TEST_MODEL must be a local checkpoint directory"
    assert os.environ.get("PT_ENABLE_INT64_SUPPORT", "").lower() in {"1", "true"}
    assert os.environ.get("PT_HPU_LAZY_MODE") in {"0", "1"}
    world = int(os.environ.get("ARENO_HPU_TEST_WORLD_SIZE", "1"))
    tp = int(os.environ.get("ARENO_HPU_TEST_TP_SIZE", "1"))
    devices = [int(x) for x in os.environ.get("ARENO_HPU_TEST_DEVICES", ",".join(map(str, range(world)))).split(",")]
    rollout = os.environ.get("ARENO_HPU_TEST_ROLLOUT_DEVICES")
    config = HpuConfig(
        tp_size=tp,
        devices=devices,
        rollout_devices=[int(x) for x in rollout.split(",")] if rollout else None,
        rollout_tp_size=int(os.environ.get("ARENO_HPU_TEST_ROLLOUT_TP_SIZE", "1")) if rollout else None,
        optimizer={"adam_8bit": optimizer == "8bit", "adam_4bit": optimizer == "4bit"},
        runtime={"activation_checkpointing": True},
    )
    checkpoint = tmp_path / "checkpoint"
    trainer = Trainer(world, model, backend_type=HPU, custom_config=config)
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
    restored = Trainer(world, str(checkpoint), backend_type=HPU, custom_config=config)
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
