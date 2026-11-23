"""Opt-in full-parameter CUDA regression for rollout and train logprob agreement."""

import os

import pytest
import torch

from areno import Trainer
from areno.api import CudaConfig, SamplingParams, TrainSequence
from areno.api.algorithms import grpo_loss_fn


@pytest.mark.parametrize("model_env", ["ARENO_E2E_LING_MODEL", "ARENO_E2E_MINICPM5_1B_MODEL"])
def test_native_rollout_matches_train_logprobs(model_env, tmp_path):
    model_path = os.getenv(model_env)
    if not model_path or not torch.cuda.is_available():
        pytest.skip(f"requires CUDA and a checkpoint in {model_env}")

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    config = CudaConfig(
        devices=[0],
        max_running_prompts=1,
        optimizer={"adam_8bit": True},
        runtime={"compile_model": False, "decode_graph_buckets": [1]},
    )
    trainer = Trainer(1, model_path, custom_config=config, metrics_log_dir=str(tmp_path))
    trainer.init()
    try:
        tokenizer = trainer.get_tokenizer()
        prompt = tokenizer.encode(
            "Describe how to measure rainfall at a weather station. " * 1000, add_special_tokens=False
        )[:4096]
        assert len(prompt) == 4096
        params = SamplingParams(
            max_prompt_len=4096, max_context_len=8192, max_new_tokens=4096, seed=635, ignore_eos=True
        )
        trainer.begin_rollout_session()
        try:
            (result,) = trainer.rollout_token_batch([prompt], 1, params)
        finally:
            trainer.end_rollout_session()
        (response,) = result.sequences
        assert len(response.resp_tokens) == len(response.resp_logprobs) == 4096
        sample = TrainSequence(
            tokens=prompt + response.resp_tokens,
            prompt_len=len(prompt),
            logprobs=[0.0] * len(prompt) + response.resp_logprobs,
            scalar_advantage=1.0,
            reward=1.0,
            eos_token_id=tokenizer.eos_token_id,
            routed_experts=response.routed_experts,
        )
        metrics = trainer.train([sample], grpo_loss_fn, mini_bs=1)
        assert metrics["response_len"] == 4096
    finally:
        trainer.close()

    (event,) = EventAccumulator(str(tmp_path)).Reload().Scalars("train/logp_abs_diff_mean")
    assert event.value == 0.0
