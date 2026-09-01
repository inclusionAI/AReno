"""Opt-in TP8 Native LoRA E2E for a production Flash V3 checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from areno import Trainer
from areno.adapters import LoraConfig
from areno.api import CudaConfig, SamplingParams
from areno.api.algorithms import get_algorithm
from areno.api.trainer_config import PolicyTrainerConfig
from areno.api.trainers.policy_only import PolicyOnlyTrainer

FLASH_V3_TARGETS = (
    "layers.0.attention.q_proj",
    "layers.0.attention.k_proj",
    "layers.2.mlp.experts.linear_fc1",
    "layers.2.mlp.experts.linear_fc2",
)


class _ObservedTrainer:
    def __init__(self, inner: Trainer) -> None:
        self.inner = inner
        self.rollout_versions: list[int | None] = []
        self.train_versions: list[int | None] = []
        self.train_results: list[dict[str, float]] = []

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    async def rollout_token_batch_async(self, prompt_tokens, n_samples, sampling_params, *, prompt_features=None):
        results = await self.inner.rollout_token_batch_async(
            prompt_tokens,
            n_samples,
            sampling_params,
            prompt_features=prompt_features,
        )
        self.rollout_versions.extend(result.adapter_version for result in results)
        return results

    def train(self, batch_data, loss_fn, mini_bs=8, gradient_accumulation_steps=None):
        result = self.inner.train(batch_data, loss_fn, mini_bs, gradient_accumulation_steps)
        self.train_versions.append(result.get("adapter_version"))
        self.train_results.append(result)
        return result


def test_flash_v3_tp8_sp_rollout_train_next_rollout(tmp_path: Path) -> None:
    model_path = os.getenv("ARENO_E2E_FLASH_V3_MODEL")
    if not model_path:
        pytest.skip("set ARENO_E2E_FLASH_V3_MODEL to run the TP8 Flash V3 E2E")

    initial_path = tmp_path / "adapter-initial"
    trained_path = tmp_path / "adapter-trained"
    lora = LoraConfig(rank=4, alpha=4.0, target_modules=FLASH_V3_TARGETS)
    backend_config = CudaConfig(
        tp_size=8,
        dp_size=1,
        devices=list(range(8)),
        lora=lora,
        max_running_prompts=2,
        optimizer={
            "lr": 1.0e-4,
            "min_lr": 1.0e-4,
            "lr_decay_style": "constant",
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
        },
        runtime={
            "compile_model": False,
            "activation_checkpointing": True,
            "keep_rollout_state": False,
            "eager_decode": False,
        },
    )
    observed = _ObservedTrainer(Trainer(8, model_path, custom_config=backend_config))
    config = PolicyTrainerConfig(
        algo="grpo",
        ckpt=model_path,
        dataset_path="e2e://flash-v3-version-closure",
        epochs=1,
        max_steps=1,
        world_size=8,
        tp_size=8,
        train_devices=list(range(8)),
        batch_size=1,
        mini_bs=2,
        n_samples=2,
        greedy=True,
        max_running_prompts=2,
        max_prompt_tokens=32,
        max_new_tokens=2,
        optimizer_lr=1.0e-4,
        optimizer_min_lr=1.0e-4,
        lr_decay_style="constant",
        weight_decay=0.0,
        activation_checkpointing=True,
        keep_rollout_state=False,
        eager_decode=False,
        metrics_log_dir=None,
        lora=lora,
    )

    def reward_fn(record) -> float:
        return float(record.metadata["sample_index"])

    policy = PolicyOnlyTrainer(
        config,
        instance=observed,
        dataset=[{"prompt": "Compute 17 + 25. Output only the answer."}],
        reward_fn=reward_fn,
        loss_fn=get_algorithm("grpo").make_loss_fn(config),
    )

    observed.init()
    try:
        observed.export_adapter(os.fspath(initial_path))
        policy._fit_initialized()
        observed.export_adapter(os.fspath(trained_path))

        prompt_tokens = observed.get_tokenizer().encode("Compute 9 times 7.", add_special_tokens=True)
        sampling_params = SamplingParams(greedy=True, max_new_tokens=1, max_prompt_len=32)
        observed.begin_rollout_session()
        try:
            next_rollout = observed.rollout_token_batch([prompt_tokens], 1, sampling_params)
        finally:
            observed.end_rollout_session()
            observed.finish_step()
    finally:
        observed.close()

    assert observed.rollout_versions == [0]
    assert observed.train_versions == [1]
    assert len(next_rollout) == 1
    assert next_rollout[0].adapter_version == 1
    assert observed.train_results[0]["packed_training"]
    assert observed.train_results[0]["sequence_parallel"]

    initial = load_file(initial_path / "adapter_model.safetensors")
    trained = load_file(trained_path / "adapter_model.safetensors")
    changed = {key for key in initial if not torch.equal(initial[key], trained[key])}
    assert any("layers.0.attention" in key for key in changed)
    assert any("layers.2.mlp.experts" in key for key in changed)
