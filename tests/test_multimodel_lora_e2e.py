"""One opt-in real-checkpoint Native LoRA E2E across model families."""

from __future__ import annotations

import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _Case:
    model_env: str
    targets: tuple[str, ...]
    changed_fragments: tuple[str, ...]


_CASES = {
    "minicpmv46": _Case(
        model_env="ARENO_E2E_MINICPMV46_MODEL",
        targets=(
            "layers.0.attention.in_proj_q",
            "layers.0.attention.in_proj_k",
            "layers.0.attention.in_proj_v",
            "layers.0.attention.in_proj_z",
            "layers.0.attention.in_proj_b",
            "layers.0.attention.in_proj_a",
            "layers.0.attention.out_proj",
            "layers.3.attention.q_proj",
            "layers.3.attention.q_gate_proj",
            "layers.3.attention.k_proj",
            "layers.3.attention.v_proj",
            "layers.3.attention.o_proj",
            "layers.0.mlp.gate_proj",
            "layers.0.mlp.up_proj",
            "layers.0.mlp.down_proj",
        ),
        changed_fragments=(".attention.", ".mlp."),
    ),
    "phi4mm": _Case(
        model_env="ARENO_E2E_PHI4MM_MODEL",
        targets=(
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        ),
        changed_fragments=(".self_attn.", ".mlp."),
    ),
    "qwen35_vl": _Case(
        model_env="ARENO_E2E_QWEN35_MODEL",
        targets=(
            "language_model.layers.0.attention.in_proj_q",
            "language_model.layers.0.attention.in_proj_a",
            "language_model.layers.0.attention.out_proj",
            "language_model.layers.3.attention.q_proj",
            "language_model.layers.3.attention.k_proj",
            "language_model.layers.3.attention.o_proj",
            "language_model.layers.0.mlp.gate_proj",
            "language_model.layers.0.mlp.up_proj",
            "language_model.layers.0.mlp.down_proj",
        ),
        changed_fragments=(".attention.", ".mlp."),
    ),
}


class _ObservedTrainer:
    def __init__(self, inner: Trainer) -> None:
        self.inner = inner
        self.rollout_versions: list[int | None] = []
        self.train_versions: list[int | None] = []

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
        return result


def _cuda_config(lora: LoraConfig) -> CudaConfig:
    return CudaConfig(
        tp_size=2,
        dp_size=1,
        devices=[0, 1],
        sequence_parallel=True,
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
            "activation_checkpointing": False,
            "keep_rollout_state": False,
            "eager_decode": False,
        },
    )


def test_multimodel_lora_rollout_train_reload(tmp_path: Path) -> None:
    case_name = os.getenv("ARENO_E2E_MULTIMODEL_CASE")
    if not case_name:
        pytest.skip("set ARENO_E2E_MULTIMODEL_CASE to select a real-checkpoint case")
    if case_name not in _CASES:
        pytest.fail(f"unknown ARENO_E2E_MULTIMODEL_CASE={case_name!r}; expected one of {sorted(_CASES)}")
    case = _CASES[case_name]
    model_path_value = os.getenv(case.model_env)
    if not model_path_value:
        pytest.fail(f"set {case.model_env} to the local checkpoint for {case_name}")
    model_path = Path(model_path_value)
    if not model_path.is_dir():
        pytest.fail(f"checkpoint is not a directory: {model_path}")

    initial_path = tmp_path / "adapter-initial"
    trained_path = tmp_path / "adapter-trained"
    lora = LoraConfig(rank=4, alpha=4.0, target_modules=case.targets)
    observed = _ObservedTrainer(Trainer(2, os.fspath(model_path), custom_config=_cuda_config(lora)))
    config = PolicyTrainerConfig(
        algo="grpo",
        ckpt=os.fspath(model_path),
        dataset_path=f"e2e://{case_name}",
        epochs=1,
        max_steps=1,
        world_size=2,
        tp_size=2,
        sequence_parallel=True,
        train_devices=[0, 1],
        batch_size=1,
        mini_bs=2,
        n_samples=2,
        greedy=True,
        max_running_prompts=2,
        max_prompt_tokens=64,
        max_new_tokens=2,
        optimizer_lr=1.0e-4,
        optimizer_min_lr=1.0e-4,
        lr_decay_style="constant",
        weight_decay=0.0,
        activation_checkpointing=False,
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
        dataset=[{"prompt": "Write one short English noun. Output only the noun."}],
        reward_fn=reward_fn,
        loss_fn=get_algorithm("grpo").make_loss_fn(config),
    )

    observed.init()
    try:
        parity_tokens = observed.get_tokenizer().encode("A fixed adapter parity check.", add_special_tokens=True)
        observed.export_adapter(os.fspath(initial_path))
        policy._fit_initialized()
        observed.export_adapter(os.fspath(trained_path))
        trained_logprobs = observed.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
        sampling = SamplingParams(greedy=True, max_new_tokens=1, max_prompt_len=64)
        observed.begin_rollout_session()
        try:
            final_rollout = observed.rollout_token_batch([parity_tokens], 1, sampling)
        finally:
            observed.end_rollout_session()
            observed.finish_step()
    finally:
        observed.close()

    assert observed.rollout_versions == [0]
    assert observed.train_versions == [1]
    assert final_rollout[0].adapter_version == 1
    initial = load_file(initial_path / "adapter_model.safetensors")
    trained = load_file(trained_path / "adapter_model.safetensors")
    changed = {name for name in initial if not torch.equal(initial[name], trained[name])}
    assert all(any(fragment in name for name in changed) for fragment in case.changed_fragments)

    reloaded = Trainer(
        2,
        os.fspath(model_path),
        custom_config=_cuda_config(LoraConfig(adapter_path=os.fspath(trained_path))),
    )
    reloaded.init()
    try:
        reloaded_logprobs = reloaded.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
        reloaded.begin_rollout_session()
        try:
            reloaded_rollout = reloaded.rollout_token_batch([parity_tokens], 1, sampling)
        finally:
            reloaded.end_rollout_session()
            reloaded.finish_step()
    finally:
        reloaded.close()

    torch.testing.assert_close(torch.tensor(reloaded_logprobs), torch.tensor(trained_logprobs), rtol=0.0, atol=1.0e-5)
    assert reloaded_rollout[0].sequences[0].resp_tokens == final_rollout[0].sequences[0].resp_tokens
