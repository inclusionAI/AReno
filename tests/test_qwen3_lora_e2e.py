"""Qwen3 dense and MoE TP2/DP2 native-LoRA rollout/train/PEFT E2E."""

from __future__ import annotations

import os
import sys
from importlib import util as importlib_util
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from areno.adapters import LoraConfig
from areno.api import ArenoConfig, Trainer
from areno.api.algorithms import get_algorithm
from areno.api.roles import ModelRole
from areno.api.trainer_config import PolicyTrainerConfig
from areno.api.trainers.policy_only import PolicyOnlyTrainer


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


@pytest.mark.parametrize(
    ("model_env", "model_kind"),
    (("ARENO_E2E_QWEN3_MODEL", "dense"), ("ARENO_E2E_QWEN3_MOE_MODEL", "moe")),
)
def test_qwen3_lora_tp2_dp2_rollout_train_peft(tmp_path: Path, model_env: str, model_kind: str) -> None:
    model_path_value = os.getenv(model_env)
    if not model_path_value:
        pytest.skip(f"set {model_env} to run the 4-GPU Qwen3 {model_kind} LoRA E2E")
    model_path = Path(model_path_value)
    initial_path = tmp_path / "adapter-initial"
    checkpoint_path = tmp_path / "checkpoints"
    final_path = checkpoint_path / "step_000002"
    reexported_path = tmp_path / "adapter-reexported"
    lora = LoraConfig(rank=8, alpha=16.0)
    backend_config = ArenoConfig(
        tp_size=2,
        dp_size=2,
        devices=[0, 1, 2, 3],
        lora=lora,
        reference_mode="reuse_actor_base",
        max_running_prompts=4,
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
        },
    )
    inner = Trainer(4, os.fspath(model_path), custom_config=backend_config)
    observed = _ObservedTrainer(inner)
    config = PolicyTrainerConfig(
        algo="grpo",
        ckpt=os.fspath(model_path),
        dataset_path="e2e://in-memory",
        save_path=os.fspath(checkpoint_path),
        save_interval=2,
        epochs=1,
        max_steps=2,
        world_size=4,
        tp_size=2,
        train_devices=[0, 1, 2, 3],
        batch_size=1,
        mini_bs=4,
        n_samples=4,
        max_running_prompts=4,
        max_prompt_tokens=64,
        max_new_tokens=16,
        optimizer_lr=1.0e-4,
        optimizer_min_lr=1.0e-4,
        lr_decay_style="constant",
        weight_decay=0.0,
        activation_checkpointing=False,
        keep_rollout_state=False,
        metrics_log_dir=None,
        lora=lora,
    )
    dataset = [
        {"prompt": "Write one uncommon English noun. Output only the noun."},
        {"prompt": "Invent one short fictional name. Output only the name."},
    ]

    def reward_fn(record) -> float:
        return float(record.metadata["sample_index"])

    policy = PolicyOnlyTrainer(
        config,
        instance=observed,
        dataset=dataset,
        reward_fn=reward_fn,
        loss_fn=get_algorithm("grpo").make_loss_fn(config),
    )

    observed.init()
    try:
        parity_tokens = observed.get_tokenizer().encode("A short adapter parity check.", add_special_tokens=True)
        observed.export_adapter(os.fspath(initial_path))
        initial_native_logprobs = observed.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
        policy._fit_initialized()
        trained_logprobs = observed.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
        observed.ensure_roles({"ref": ModelRole("ref", os.fspath(model_path), trainable=False)})
        reference_logprobs = observed.score_logprobs("ref", [parity_tokens], microbatch_size=1)[0]
        restored_actor_logprobs = observed.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
    finally:
        observed.close()

    assert observed.rollout_versions == [0, 1]
    assert observed.train_versions == [1, 2]
    torch.testing.assert_close(
        torch.tensor(reference_logprobs), torch.tensor(initial_native_logprobs), rtol=0.0, atol=1.0e-5
    )
    torch.testing.assert_close(
        torch.tensor(restored_actor_logprobs), torch.tensor(trained_logprobs), rtol=0.0, atol=1.0e-5
    )
    assert (final_path / "adapter_config.json").is_file()
    assert (final_path / "adapter_model.safetensors").is_file()
    initial = load_file(initial_path / "adapter_model.safetensors")
    final = load_file(final_path / "adapter_model.safetensors")
    changed = {name for name in initial if not torch.equal(initial[name], final[name])}
    assert any(".self_attn." in name for name in changed)
    if model_kind == "moe":
        assert any(".experts." in name for name in changed)
    else:
        assert any(".mlp." in name for name in changed)

    if model_kind == "dense":
        initial_peft_logprobs = _peft_logprobs(model_path, initial_path, parity_tokens)
        peft_logprobs = _peft_logprobs(model_path, final_path, parity_tokens)
    else:
        expert_key = next(name for name in changed if ".experts." in name)
        peft_logprobs = _peft_logprobs(
            model_path,
            final_path,
            parity_tokens,
            expected_state=final,
            representative_key=expert_key,
        )
    imported = Trainer(
        4,
        os.fspath(model_path),
        custom_config=ArenoConfig(
            tp_size=2,
            dp_size=2,
            devices=[0, 1, 2, 3],
            lora=LoraConfig(adapter_path=os.fspath(final_path)),
            runtime={"compile_model": False, "activation_checkpointing": False},
        ),
    )
    imported.init()
    try:
        imported.export_adapter(os.fspath(reexported_path))
        areno_logprobs = imported.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
        repeated_logprobs = imported.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
    finally:
        imported.close()
    reexported = load_file(reexported_path / "adapter_model.safetensors")
    assert reexported.keys() == final.keys()
    assert all(torch.equal(reexported[name], final[name]) for name in final)
    torch.testing.assert_close(torch.tensor(repeated_logprobs), torch.tensor(areno_logprobs), rtol=0.0, atol=1.0e-5)
    torch.testing.assert_close(
        torch.tensor(areno_logprobs),
        torch.tensor(trained_logprobs),
        rtol=0.0,
        atol=1.0e-5,
    )
    if model_kind == "dense":
        native_delta = torch.tensor(areno_logprobs[1:]) - torch.tensor(initial_native_logprobs[1:])
        peft_delta = torch.tensor(peft_logprobs) - torch.tensor(initial_peft_logprobs)
        torch.testing.assert_close(native_delta, peft_delta, rtol=0.0, atol=1.5e-1)
    else:
        assert torch.isfinite(torch.tensor(peft_logprobs)).all()


def _peft_logprobs(
    model_path: Path,
    adapter_path: Path,
    token_ids: list[int],
    *,
    expected_state: dict[str, torch.Tensor] | None = None,
    representative_key: str | None = None,
) -> list[float]:
    peft_source = os.getenv("ARENO_E2E_PEFT_SOURCE")
    if peft_source:
        sys.path.insert(0, peft_source)
    original_find_spec = importlib_util.find_spec

    def find_spec_without_torchao(name, *args, **kwargs):
        if name == "torchao":
            return None
        return original_find_spec(name, *args, **kwargs)

    importlib_util.find_spec = find_spec_without_torchao
    try:
        from peft import PeftModel, get_peft_model_state_dict
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda:0")
        model = PeftModel.from_pretrained(base, os.fspath(adapter_path), autocast_adapter_dtype=False).eval()
        if expected_state is not None:
            loaded = get_peft_model_state_dict(model, save_embedding_layers=False)
            assert loaded.keys() == expected_state.keys()
            assert representative_key is not None
            torch.testing.assert_close(
                loaded[representative_key].detach().cpu().float(),
                expected_state[representative_key].float(),
                rtol=0.0,
                atol=0.0,
            )
        tokens = torch.tensor([token_ids], device="cuda:0", dtype=torch.long)
        with torch.inference_mode():
            logits = model(input_ids=tokens).logits[0, :-1].float()
            selected = logits.log_softmax(dim=-1).gather(-1, tokens[0, 1:].unsqueeze(-1)).squeeze(-1)
        result = selected.cpu().tolist()
        del model, base, tokens, logits, selected
        torch.cuda.empty_cache()
        return result
    finally:
        importlib_util.find_spec = original_find_spec
        if peft_source:
            sys.path.remove(peft_source)
