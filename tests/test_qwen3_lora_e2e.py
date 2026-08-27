"""Qwen3 dense and MoE native-LoRA rollout/train/PEFT E2E."""

from __future__ import annotations

import os
import sys
from importlib import util as importlib_util
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from areno.adapters import LoraConfig
from areno.api import CudaConfig, SamplingParams, Trainer
from areno.api.algorithms import get_algorithm
from areno.api.roles import ModelRole
from areno.api.trainer_config import DPOTrainerConfig, PolicyTrainerConfig, PPOTrainerConfig
from areno.api.trainer_factory import build_trainer
from areno.api.trainers.policy_only import PolicyOnlyTrainer


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


class _ObservedReferenceTrainer(_ObservedTrainer):
    """Observe the public PPO/DPO lifecycle without inspecting adapter slots."""

    def __init__(self, inner: Trainer) -> None:
        super().__init__(inner)
        self.reference_versions: list[int] = []
        self.critic_train_count = 0
        self.roles: set[str] = set()
        self.parity_scores: dict[str, list[float]] = {}
        self.parity_tokens: list[int] | None = None

    def ensure_roles(self, roles: dict[str, ModelRole]) -> None:
        self.roles = set(roles)
        self.inner.ensure_roles(roles)
        self.parity_tokens = self.inner.get_tokenizer().encode(
            "A fixed actor-base reference check.", add_special_tokens=True
        )
        self.parity_scores["initial_actor"] = self.inner.score_logprobs(
            "actor", [self.parity_tokens], microbatch_size=1
        )[0]
        self.parity_scores["initial_reference"] = self.inner.score_logprobs(
            "ref", [self.parity_tokens], microbatch_size=1
        )[0]

    def score_logprobs(self, role, token_rows, *, features=None, microbatch_size=None):
        if role == "ref":
            self.reference_versions.append(len(self.train_versions))
        return self.inner.score_logprobs(
            role,
            token_rows,
            features=features,
            microbatch_size=microbatch_size,
        )

    def train_values(
        self,
        role,
        batch_data,
        mini_bs,
        gradient_accumulation_steps=None,
        *,
        cliprange_value=0.5,
        value_loss_coef=0.5,
    ):
        result = self.inner.train_values(
            role,
            batch_data,
            mini_bs,
            gradient_accumulation_steps,
            cliprange_value=cliprange_value,
            value_loss_coef=value_loss_coef,
        )
        self.critic_train_count += 1
        return result

    def close(self) -> None:
        if self.parity_tokens is not None and self.train_versions == [1, 2]:
            self.parity_scores["final_actor_before_reference"] = self.inner.score_logprobs(
                "actor", [self.parity_tokens], microbatch_size=1
            )[0]
            self.parity_scores["final_reference"] = self.inner.score_logprobs(
                "ref", [self.parity_tokens], microbatch_size=1
            )[0]
            self.parity_scores["final_actor_after_reference"] = self.inner.score_logprobs(
                "actor", [self.parity_tokens], microbatch_size=1
            )[0]
        self.inner.close()


@pytest.mark.parametrize("algorithm", ("ppo", "dpo"))
def test_qwen3_lora_tp2_dp2_reference_two_step(algorithm: str) -> None:
    model_path_value = os.getenv("ARENO_E2E_QWEN3_MODEL")
    if not model_path_value:
        pytest.skip("set ARENO_E2E_QWEN3_MODEL to run the 4-GPU Qwen3 LoRA PPO/DPO E2E")
    model_path = Path(model_path_value)
    common = {
        "algo": algorithm,
        "ckpt": os.fspath(model_path),
        "dataset_path": f"e2e://{algorithm}-in-memory",
        "epochs": 1,
        "max_steps": 2,
        "world_size": 4,
        "tp_size": 2,
        "train_devices": [0, 1, 2, 3],
        "batch_size": 2,
        "score_micro_bs": 2,
        "gradient_accumulation_steps": 1,
        "max_prompt_tokens": 64,
        "optimizer_lr": 1.0e-4,
        "optimizer_min_lr": 1.0e-4,
        "lr_decay_style": "constant",
        "weight_decay": 0.0,
        "activation_checkpointing": False,
        "keep_rollout_state": False,
        "eager_decode": True,
        "metrics_log_dir": None,
        "lora": LoraConfig(rank=8, alpha=16.0),
        "reference_mode": "reuse_actor_base",
        "ref_ckpt": os.fspath(model_path),
    }
    if algorithm == "ppo":
        config = PPOTrainerConfig(
            **common,
            mini_bs=2,
            n_samples=1,
            greedy=True,
            max_new_tokens=3,
            max_running_prompts=2,
            critic_ckpt=os.fspath(model_path),
            critic_lr=1.0e-4,
            critic_warmup_steps=0,
        )
        dataset = [
            {"prompt": "Reply with the English word for the number one."},
            {"prompt": "Reply with the English word for the number two."},
            {"prompt": "Reply with the English word for the number three."},
            {"prompt": "Reply with the English word for the number four."},
        ]

        def reward_fn(record) -> float:
            return float(record.metadata["prompt_index"])

    else:
        config = DPOTrainerConfig(**common, mini_bs=4, max_new_tokens=8, dpo_beta=0.1)
        dataset = [
            {"prompt": "What is 1 + 1?", "chosen": "2", "rejected": "3"},
            {"prompt": "What is 2 + 2?", "chosen": "4", "rejected": "5"},
            {"prompt": "What is 3 + 3?", "chosen": "6", "rejected": "7"},
            {"prompt": "What is 4 + 4?", "chosen": "8", "rejected": "9"},
        ]
        reward_fn = None

    backend_config = config.cuda_config()
    backend_config.dp_size = 2
    backend_config.runtime["compile_model"] = False
    observed = _ObservedReferenceTrainer(
        Trainer(
            config.world_size,
            config.ckpt,
            custom_config=backend_config,
            metrics_log_dir=None,
            score_micro_bs=config.score_micro_bs,
        )
    )
    trainer = build_trainer(
        config,
        instance=observed,
        dataset=dataset,
        reward_fn=reward_fn,
        loss_fn=get_algorithm(algorithm).make_loss_fn(config),
    )
    trainer.fit()

    assert observed.reference_versions == [0, 1]
    assert observed.train_versions == [1, 2]
    assert observed.roles == ({"actor", "ref", "critic"} if algorithm == "ppo" else {"ref"})
    assert observed.rollout_versions == ([0, 0, 1, 1] if algorithm == "ppo" else [])
    assert observed.critic_train_count == (2 if algorithm == "ppo" else 0)
    torch.testing.assert_close(
        torch.tensor(observed.parity_scores["initial_reference"]),
        torch.tensor(observed.parity_scores["initial_actor"]),
        rtol=0.0,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        torch.tensor(observed.parity_scores["final_reference"]),
        torch.tensor(observed.parity_scores["initial_reference"]),
        rtol=0.0,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        torch.tensor(observed.parity_scores["final_actor_after_reference"]),
        torch.tensor(observed.parity_scores["final_actor_before_reference"]),
        rtol=0.0,
        atol=1.0e-5,
    )


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
    backend_config = CudaConfig(
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
        custom_config=CudaConfig(
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


def test_qwen3_moe_lora_tp8_replicated_kv_roundtrip(tmp_path: Path) -> None:
    model_path_value = os.getenv("ARENO_E2E_QWEN3_MOE_MODEL")
    if not model_path_value:
        pytest.skip("set ARENO_E2E_QWEN3_MOE_MODEL to run the 8-GPU replicated-KV LoRA E2E")
    model_path = Path(model_path_value)
    initial_path = tmp_path / "adapter-initial"
    checkpoint_path = tmp_path / "checkpoints"
    final_path = checkpoint_path / "step_000001"
    reexported_path = tmp_path / "adapter-reexported"
    lora = LoraConfig(rank=8, alpha=16.0)
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
        },
        runtime={
            "compile_model": False,
            "activation_checkpointing": False,
            "keep_rollout_state": False,
            "eager_decode": True,
        },
    )
    observed = _ObservedTrainer(Trainer(8, os.fspath(model_path), custom_config=backend_config))
    config = PolicyTrainerConfig(
        algo="grpo",
        ckpt=os.fspath(model_path),
        dataset_path="e2e://replicated-kv",
        save_path=os.fspath(checkpoint_path),
        save_interval=1,
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
        max_prompt_tokens=64,
        max_new_tokens=4,
        optimizer_lr=1.0e-4,
        optimizer_min_lr=1.0e-4,
        lr_decay_style="constant",
        weight_decay=0.0,
        activation_checkpointing=False,
        keep_rollout_state=False,
        eager_decode=True,
        metrics_log_dir=None,
        lora=lora,
    )

    def reward_fn(record) -> float:
        return float(record.metadata["sample_index"])

    policy = PolicyOnlyTrainer(
        config,
        instance=observed,
        dataset=[{"prompt": "Write one uncommon English noun. Output only the noun."}],
        reward_fn=reward_fn,
        loss_fn=get_algorithm("grpo").make_loss_fn(config),
    )
    observed.init()
    try:
        parity_tokens = observed.get_tokenizer().encode("A replicated KV adapter check.", add_special_tokens=True)
        observed.export_adapter(os.fspath(initial_path))
        policy._fit_initialized()
        trained_logprobs = observed.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
    finally:
        observed.close()

    initial = load_file(initial_path / "adapter_model.safetensors")
    final = load_file(final_path / "adapter_model.safetensors")
    kv_keys = [name for name in final if ".self_attn.k_proj." in name or ".self_attn.v_proj." in name]
    assert kv_keys
    assert any(not torch.equal(initial[name], final[name]) for name in kv_keys)
    assert all(final[name].shape[0] == 512 for name in kv_keys if name.endswith("lora_B.weight"))

    imported = Trainer(
        8,
        os.fspath(model_path),
        custom_config=CudaConfig(
            tp_size=8,
            dp_size=1,
            devices=list(range(8)),
            lora=LoraConfig(adapter_path=os.fspath(final_path)),
            runtime={"compile_model": False, "activation_checkpointing": False, "eager_decode": True},
        ),
    )
    imported.init()
    try:
        imported.export_adapter(os.fspath(reexported_path))
        imported_logprobs = imported.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
    finally:
        imported.close()
    reexported = load_file(reexported_path / "adapter_model.safetensors")
    assert reexported.keys() == final.keys()
    assert all(torch.equal(reexported[name], final[name]) for name in final)
    torch.testing.assert_close(torch.tensor(imported_logprobs), torch.tensor(trained_logprobs), rtol=0.0, atol=1.0e-5)


@pytest.mark.parametrize(
    ("model_env", "model_kind", "rollout_tp_size", "rollout_devices"),
    (
        ("ARENO_E2E_QWEN3_MODEL", "dense", 1, [2]),
        ("ARENO_E2E_QWEN3_MODEL", "dense", 2, [2, 3]),
        ("ARENO_E2E_QWEN3_MOE_MODEL", "moe", 2, [2, 3]),
    ),
)
def test_qwen3_lora_independent_rollout_two_step(
    tmp_path: Path,
    model_env: str,
    model_kind: str,
    rollout_tp_size: int,
    rollout_devices: list[int],
) -> None:
    model_path_value = os.getenv(model_env)
    if not model_path_value:
        pytest.skip(f"set {model_env} to run the Qwen3 {model_kind} independent-rollout LoRA E2E")
    model_path = Path(model_path_value)
    initial_path = tmp_path / "adapter-initial"
    trained_path = tmp_path / "adapter-trained"
    rollout_path = tmp_path / "adapter-rollout"
    lora = LoraConfig(rank=8, alpha=16.0)
    config = PolicyTrainerConfig(
        algo="grpo",
        ckpt=os.fspath(model_path),
        dataset_path="e2e://independent-rollout",
        epochs=1,
        max_steps=2,
        world_size=2,
        tp_size=2,
        train_devices=[0, 1],
        rollout_tp_size=rollout_tp_size,
        rollout_devices=rollout_devices,
        batch_size=1,
        mini_bs=4,
        n_samples=4,
        greedy=True,
        max_running_prompts=4,
        max_prompt_tokens=64,
        max_new_tokens=4,
        optimizer_lr=1.0e-4,
        optimizer_min_lr=1.0e-4,
        lr_decay_style="constant",
        weight_decay=0.0,
        activation_checkpointing=False,
        keep_rollout_state=False,
        eager_decode=True,
        metrics_log_dir=None,
        lora=lora,
    )
    backend_config = config.cuda_config()
    backend_config.dp_size = 1
    backend_config.runtime["compile_model"] = False
    inner = Trainer(2, os.fspath(model_path), custom_config=backend_config)
    observed = _ObservedTrainer(inner)
    dataset = [
        {"prompt": "Write one uncommon English noun. Output only the noun."},
        {"prompt": "Write one uncommon English noun. Output only the noun."},
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
        observed.export_adapter(os.fspath(initial_path))
        policy._fit_initialized()
        observed.export_adapter(os.fspath(trained_path))

        sampling_params = SamplingParams(greedy=True, max_new_tokens=2, max_prompt_len=64)
        prompt_tokens = observed.get_tokenizer().encode(
            "A fixed independent-rollout adapter check.", add_special_tokens=True
        )
        observed.begin_rollout_session()
        try:
            final_rollout = observed.rollout_token_batch([prompt_tokens], 1, sampling_params)
        finally:
            observed.end_rollout_session()
            observed.finish_step()

        backend = inner._backend
        assert backend is not None
        backend._require_rollout_engine().export_adapter(os.fspath(rollout_path))
    finally:
        observed.close()

    assert observed.rollout_versions == [0, 1]
    assert observed.train_versions == [1, 2]
    assert final_rollout[0].adapter_version == 2
    assert observed.train_results[1]["policy_sync_tensors"] > 0
    assert observed.train_results[1]["policy_sync_bytes"] > 0

    initial = load_file(initial_path / "adapter_model.safetensors")
    trained = load_file(trained_path / "adapter_model.safetensors")
    rollout = load_file(rollout_path / "adapter_model.safetensors")
    changed = {name for name in initial if not torch.equal(initial[name], trained[name])}
    assert any(".self_attn." in name for name in changed)
    if model_kind == "moe":
        assert any(".experts." in name for name in changed)
        representative_keys = [
            next(name for name in changed if ".self_attn." in name),
            next(name for name in changed if ".experts." in name),
        ]
    else:
        assert any(".mlp." in name for name in changed)
        representative_keys = [
            next(name for name in changed if ".self_attn." in name),
            next(name for name in changed if ".mlp." in name),
        ]
    assert rollout.keys() == trained.keys()
    assert all(torch.equal(rollout[name], trained[name]) for name in representative_keys)


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
