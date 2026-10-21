"""Opt-in real-checkpoint two-step adapter regression on one CUDA device."""

import asyncio
import json
import math
import os

import pytest
import torch
from safetensors.torch import load_file

from areno import Trainer
from areno.adapters import LoraConfig
from areno.adapters.config import QWEN3_DENSE_TARGETS, QWEN35_TARGETS
from areno.api import CudaConfig, SamplingParams, TrainSequence
from areno.api.algorithms import sft_loss_fn


@pytest.mark.parametrize(
    "model_env,qlora,targets",
    [
        ("ARENO_E2E_QWEN35_MODEL", False, QWEN35_TARGETS),
        ("ARENO_E2E_QWEN35_MODEL", True, QWEN35_TARGETS),
        ("ARENO_E2E_QWEN3_MODEL", True, QWEN3_DENSE_TARGETS),
        ("ARENO_E2E_LING_MODEL", True, QWEN3_DENSE_TARGETS),
    ],
    ids=["qwen35-lora", "qwen35-qlora", "qwen3-qlora", "ling-qlora"],
)
def test_native_adapter_two_step_save_reload(tmp_path, model_env, qlora, targets):
    model_path = os.getenv(model_env)
    if not model_path or not torch.cuda.is_available():
        pytest.skip(f"set {model_env} to a local checkpoint on a CUDA host")

    def build(lora):
        return Trainer(
            1,
            model_path,
            metrics_log_dir=str(tmp_path / "metrics"),
            custom_config=CudaConfig(
                tp_size=1,
                devices=[0],
                lora=lora,
                max_running_prompts=2,
                optimizer={
                    "lr": 1e-4,
                    "min_lr": 1e-4,
                    "lr_decay_style": "constant",
                    "weight_decay": 0.0,
                    "adam_4bit": True,
                },
                runtime={"compile_model": False, "activation_checkpointing": True, "keep_rollout_state": False},
            ),
        )

    trainer = build(LoraConfig(rank=8, alpha=16, target_modules=targets, qlora=qlora))
    report = {"model_env": model_env, "model_path": model_path, "qlora": qlora, "steps": []}
    trainer.init()
    try:
        tokenizer = trainer.get_tokenizer()
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write the single word BLUE."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
        response = tokenizer.encode("BLUE", add_special_tokens=False) + [tokenizer.eos_token_id]
        sequence = TrainSequence(
            tokens=prompt + response,
            prompt_len=len(prompt),
            prompt_mask=[True] * len(prompt) + [False] * len(response),
            loss_mask=[False] * len(prompt) + [True] * len(response),
            eos_token_id=tokenizer.eos_token_id,
        )
        trainer.export_adapter(str(tmp_path / "initial"))

        async def sample():
            sampling = SamplingParams(greedy=True, max_new_tokens=16, max_prompt_len=512)
            async with trainer.rollout_session(sampling_params=sampling, max_running_prompts=2, proxy=False):
                return await trainer.rollout_token_batch_async([prompt], 1, sampling)

        for step in range(2):
            rollouts = asyncio.run(sample())
            assert rollouts[0].adapter_version == step
            sampled = rollouts[0].sequences[0]
            assert sampled.resp_tokens and len(sampled.resp_tokens) == len(sampled.resp_logprobs)
            scores = trainer.score_logprobs("actor", [prompt + sampled.resp_tokens], microbatch_size=1)[0]
            scored = scores[len(prompt) :]
            assert len(scored) == len(sampled.resp_logprobs)
            difference = torch.tensor(scored) - torch.tensor(sampled.resp_logprobs)
            mean_abs_diff = difference.abs().mean().item()
            assert math.isfinite(mean_abs_diff) and mean_abs_diff < 0.05
            result = trainer.train([sequence], sft_loss_fn, mini_bs=1)
            assert result["adapter_version"] == step + 1
            assert math.isfinite(result["loss"]) and result["grad_norm"] > 0 and math.isfinite(result["grad_norm"])
            report["steps"].append(
                {
                    "step": step + 1,
                    "metrics": result,
                    "logp_abs_diff_mean": mean_abs_diff,
                    "decoded": tokenizer.decode(sampled.resp_tokens),
                }
            )
            (tmp_path / "results.json").write_text(json.dumps(report, indent=2))
        trainer.export_adapter(str(tmp_path / "final"))
        final_scores = trainer.score_logprobs("actor", [sequence.tokens], microbatch_size=1)[0]
    finally:
        trainer.close()
    initial = load_file(tmp_path / "initial/adapter_model.safetensors")
    final = load_file(tmp_path / "final/adapter_model.safetensors")
    changed = [name for name in initial if not torch.equal(initial[name], final[name])]
    assert any(".mlp." in name for name in changed)
    if model_env == "ARENO_E2E_QWEN35_MODEL":
        assert any(".linear_attn." in name for name in changed)
        assert any(".self_attn." in name for name in changed)
    report["changed_adapter_tensors"] = len(changed)
    reloaded = build(LoraConfig(adapter_path=str(tmp_path / "final")))
    reloaded.init()
    try:
        restored_scores = reloaded.score_logprobs("actor", [sequence.tokens], microbatch_size=1)[0]
        reloaded.export_adapter(str(tmp_path / "reexported"))
    finally:
        reloaded.close()
    torch.testing.assert_close(torch.tensor(final_scores), torch.tensor(restored_scores), atol=1e-5, rtol=0)
    restored = load_file(tmp_path / "reexported/adapter_model.safetensors")
    assert restored.keys() == final.keys()
    assert all(torch.equal(restored[name], final[name]) for name in final)
    report["reload_max_abs_diff"] = (torch.tensor(final_scores) - torch.tensor(restored_scores)).abs().max().item()
    report["passed"] = True
    (tmp_path / "results.json").write_text(json.dumps(report, indent=2))
