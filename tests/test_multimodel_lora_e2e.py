"""One opt-in real-checkpoint Native LoRA E2E across model families."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from areno import Trainer
from areno.adapters import LoraConfig
from areno.api import CudaConfig, SamplingParams
from areno.api.roles import ModelRole
from areno.api.trainer_config import PolicyTrainerConfig
from areno.api.trainers.policy_only import PolicyOnlyTrainer
from areno.engine.worker import ArenoWorker


@dataclass(frozen=True)
class _Case:
    model_env: str
    targets: tuple[str, ...]
    changed_fragments: tuple[str, ...]
    tp_size: int = 2
    steps: int = 2
    activation_checkpointing: bool = False
    attn_backend: str = "flash"


_CASES = {
    "flash_v3": _Case(
        model_env="ARENO_E2E_FLASH_V3_MODEL",
        targets=(
            "layers.0.attention.q_proj",
            "layers.0.attention.k_proj",
            "layers.0.attention.v_proj",
            "layers.0.attention.f_proj",
            "layers.0.attention.g_proj",
            "layers.2.mlp.experts.linear_fc1",
            "layers.2.mlp.experts.linear_fc2",
        ),
        changed_fragments=(".attention.", ".mlp.experts."),
        tp_size=8,
        steps=2,
        activation_checkpointing=True,
    ),
    "olmo2": _Case(
        model_env="ARENO_E2E_OLMO2_MODEL",
        attn_backend="native",
        targets=(
            "layers.0.self_attn.q_proj",
            "layers.0.self_attn.k_proj",
            "layers.0.self_attn.v_proj",
            "layers.0.self_attn.o_proj",
            "layers.0.mlp.gate_proj",
            "layers.0.mlp.up_proj",
            "layers.0.mlp.down_proj",
        ),
        changed_fragments=(".self_attn.", ".mlp."),
    ),
    "bailing_legacy": _Case(
        model_env="ARENO_E2E_BAILING_LEGACY_MODEL",
        targets=(
            "layers.0.attention.query_key_value",
            "layers.0.attention.g_proj",
            "layers.0.attention.dense",
            "layers.4.attention.query_key_value",
            "layers.4.attention.dense",
            "layers.0.mlp.gate_proj",
            "layers.0.mlp.up_proj",
            "layers.0.mlp.down_proj",
            "layers.1.mlp.experts.linear_fc1",
            "layers.1.mlp.experts.linear_fc2",
        ),
        changed_fragments=(".attention.", ".mlp.", ".experts."),
    ),
    "gemma4": _Case(
        model_env="ARENO_E2E_GEMMA4_MODEL",
        targets=(
            "per_layer_model_projection",
            "layers.0.self_attn.q_proj",
            "layers.0.self_attn.k_proj",
            "layers.0.self_attn.v_proj",
            "layers.0.self_attn.o_proj",
            "layers.4.self_attn.q_proj",
            "layers.4.self_attn.k_proj",
            "layers.4.self_attn.v_proj",
            "layers.4.self_attn.o_proj",
            "layers.0.mlp.gate_proj",
            "layers.0.mlp.up_proj",
            "layers.0.mlp.down_proj",
        ),
        changed_fragments=(".self_attn.", ".mlp.", ".per_layer_model_projection."),
    ),
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


class _PackedOracleWorker(ArenoWorker):
    """Check the actual rollout cache against canonical projections on every rank."""

    def __init__(self, config):
        super().__init__(config)
        if self.adapter_registry is None:
            return
        attention = self.model.layers[0].attention
        binding = attention.lora_execution
        original_project = binding.project
        checked_versions = set()

        def project(x, projections, *, use_cache):
            projections = tuple(projections)
            outputs = original_project(x, projections, use_cache=use_cache)
            version = self.adapter_registry.version
            if (
                use_cache
                and version not in checked_versions
                and all(projection.lora_slot.enabled for projection in projections)
                and not torch.cuda.is_current_stream_capturing()
            ):
                assert binding.packed_A.numel() > 0, "rollout must consume the five-projection packed cache"
                expected_A = torch.cat([projection.lora_slot.lora_A for projection in projections])
                torch.testing.assert_close(binding.packed_A, expected_A, rtol=0, atol=0)
                with torch.no_grad():
                    canonical = original_project(x, projections, use_cache=False)
                for packed, direct in zip(outputs, canonical, strict=True):
                    # BF16 GEMM accumulation differs between packed and separate A.
                    torch.testing.assert_close(packed, direct, rtol=2e-2, atol=2e-2)
                checked_versions.add(version)
                stage = "reloaded" if config.lora.adapter_path else "trained"
                rank = torch.distributed.get_rank()
                evidence = Path(os.environ["ARENO_E2E_PACKED_EVIDENCE"])
                evidence.mkdir(parents=True, exist_ok=True)
                (evidence / f"{stage}-rank{rank}-v{version}.json").write_text(
                    json.dumps({"rank": rank, "version": version, "packed_shape": list(binding.packed_A.shape)}) + "\n"
                )
            return outputs

        binding.project = project


class _ObservedTrainer:
    def __init__(self, inner: Trainer) -> None:
        self.inner = inner
        self.rollout_versions: list[int | None] = []
        self.train_versions: list[int | None] = []
        self.train_results: list[dict] = []

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


def _train_world_size(case: _Case) -> int:
    return case.tp_size * int(os.getenv("ARENO_E2E_DP_SIZE", "1"))


def _cuda_config(lora: LoraConfig | None, case: _Case) -> CudaConfig:
    world = _train_world_size(case)
    separate = os.getenv("ARENO_E2E_SEPARATE_ROLLOUT") == "1"
    return CudaConfig(
        tp_size=case.tp_size,
        dp_size=world // case.tp_size,
        devices=list(range(world)),
        rollout_devices=list(range(world, world + 2)) if separate else None,
        rollout_tp_size=1 if separate else None,
        sequence_parallel=True,
        lora=lora,
        reference_mode="reuse_actor_base"
        if lora is not None and os.getenv("ARENO_E2E_BASE_REFERENCE") == "1"
        else "independent",
        max_running_prompts=2,
        optimizer={
            "lr": 1.0e-4,
            "min_lr": 1.0e-4,
            "lr_decay_style": "constant",
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "unfreeze_multimodal_tower": os.getenv("ARENO_E2E_MEDIA_TOWER") == "1",
            "unfreeze_multimodal_projector": os.getenv("ARENO_E2E_MEDIA_PROJECTOR") == "1",
        },
        runtime={
            "attn_backend": case.attn_backend,
            "compile_model": os.getenv("ARENO_E2E_COMPILE_MODEL") == "1",
            "activation_checkpointing": case.activation_checkpointing,
            "keep_rollout_state": False,
            "eager_decode": False,
        },
    )


def _update_loss(_pack, logprobs: torch.Tensor) -> torch.Tensor:
    """Exercise adapter updates without cancelling identical greedy samples.

    This is a lifecycle diagnostic loss, not a GRPO objective qualification.
    """

    return -logprobs.mean()


def test_multimodel_lora_rollout_train_reload(tmp_path: Path, monkeypatch) -> None:
    case_name = os.getenv("ARENO_E2E_MULTIMODEL_CASE")
    if not case_name and os.getenv("ARENO_E2E_FLASH_V3_MODEL"):
        case_name = "flash_v3"
    if not case_name:
        pytest.skip("set ARENO_E2E_MULTIMODEL_CASE to select a real-checkpoint case")
    if case_name not in _CASES:
        pytest.fail(f"unknown ARENO_E2E_MULTIMODEL_CASE={case_name!r}; expected one of {sorted(_CASES)}")
    case = _CASES[case_name]
    if os.getenv("ARENO_E2E_TP_SIZE"):
        case = replace(case, tp_size=int(os.environ["ARENO_E2E_TP_SIZE"]))
    model_path_value = os.getenv(case.model_env)
    if not model_path_value:
        pytest.fail(f"set {case.model_env} to the local checkpoint for {case_name}")
    model_path = Path(model_path_value)
    if not model_path.is_dir():
        pytest.fail(f"checkpoint is not a directory: {model_path}")
    if case_name == "flash_v3":
        import areno.engine.api as engine_api

        monkeypatch.setattr(engine_api, "ArenoWorker", _PackedOracleWorker)
        monkeypatch.setenv("ARENO_E2E_PACKED_EVIDENCE", os.fspath(tmp_path / "packed-evidence"))

    initial_path = tmp_path / "adapter-initial"
    trained_path = tmp_path / "adapter-trained"
    full_targets = tuple(filter(None, os.getenv("ARENO_E2E_FULL_PARAMETER_TARGETS", "").split(",")))
    lora = LoraConfig(rank=4, alpha=4.0, target_modules=case.targets, full_parameter_targets=full_targets)
    observed = _ObservedTrainer(
        Trainer(_train_world_size(case), os.fspath(model_path), custom_config=_cuda_config(lora, case))
    )
    config = PolicyTrainerConfig(
        algo="grpo",
        ckpt=os.fspath(model_path),
        dataset_path=f"e2e://{case_name}",
        epochs=case.steps,
        max_steps=case.steps,
        world_size=_train_world_size(case),
        tp_size=case.tp_size,
        sequence_parallel=True,
        train_devices=list(range(_train_world_size(case))),
        batch_size=int(os.getenv("ARENO_E2E_DP_SIZE", "1")),
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
        activation_checkpointing=case.activation_checkpointing,
        keep_rollout_state=False,
        eager_decode=False,
        metrics_log_dir=None,
        lora=lora,
    )

    def reward_fn(record) -> float:
        return float(record.metadata["sample_index"])

    dataset = [{"prompt": "Write one short English noun. Output only the noun."}]
    media = any(os.getenv(name) == "1" for name in ("ARENO_E2E_MEDIA_PROJECTOR", "ARENO_E2E_MEDIA_TOWER"))
    image_input = media or os.getenv("ARENO_E2E_IMAGE_INPUT") == "1"
    if image_input:
        import base64
        import io

        from PIL import Image

        image = io.BytesIO()
        Image.new("RGB", (64, 64), (220, 30, 30)).save(image, format="PNG")
        dataset = [
            {"prompt": "Name the main color in one word.", "image_base64": base64.b64encode(image.getvalue()).decode()}
        ]
        config.max_prompt_tokens = 2048

    policy = PolicyOnlyTrainer(
        config,
        instance=observed,
        dataset=dataset,
        reward_fn=reward_fn,
        loss_fn=_update_loss,
    )

    # Establish the FFT forward baseline before interpreting LoRA differences.
    baseline = Trainer(_train_world_size(case), os.fspath(model_path), custom_config=_cuda_config(None, case))
    baseline.init()
    reference_scores = None
    try:
        parity_tokens = baseline.get_tokenizer().encode("A fixed adapter parity check.", add_special_tokens=True)
        base_logprobs = baseline.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
        if os.getenv("ARENO_E2E_BASE_REFERENCE") == "1":
            reference_tokens, reference_features = parity_tokens, None
            if image_input:
                from areno.api.multimodal import encode_multimodal_prompt

                reference_tokens, reference_features = encode_multimodal_prompt(
                    baseline.get_tokenizer(), baseline.get_processor(), dataset[0]
                )
            reference_baseline = baseline.score_logprobs(
                "actor", [reference_tokens], features=[reference_features], microbatch_size=1
            )[0]
        if os.getenv("ARENO_E2E_DIAGNOSTICS") == "1":
            baseline.begin_rollout_session()
            try:
                baseline.rollout_token_batch(
                    [parity_tokens], 1, SamplingParams(greedy=True, max_new_tokens=1, max_prompt_len=64)
                )
            finally:
                baseline.end_rollout_session()
                baseline.finish_step()
            base_after_rollout = baseline.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
            print("FFT_SCORE_AFTER_ROLLOUT", base_logprobs, base_after_rollout, flush=True)
    finally:
        baseline.close()
    observed.init()
    try:
        parity_tokens = observed.get_tokenizer().encode("A fixed adapter parity check.", add_special_tokens=True)
        initial_logprobs = observed.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
        torch.testing.assert_close(torch.tensor(initial_logprobs), torch.tensor(base_logprobs), rtol=0, atol=1e-5)
        observed.export_adapter(os.fspath(initial_path))
        policy._fit_initialized()
        observed.export_adapter(os.fspath(trained_path))
        trained_logprobs = observed.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
        rollout_tokens = parity_tokens
        rollout_features = None
        if image_input:
            from areno.api.multimodal import encode_multimodal_prompt

            rollout_tokens, features = encode_multimodal_prompt(
                observed.get_tokenizer(), observed.get_processor(), dataset[0]
            )
            rollout_features = [features]
        sampling = SamplingParams(greedy=True, max_new_tokens=1, max_prompt_len=config.max_prompt_tokens)
        observed.begin_rollout_session()
        try:
            final_rollout = observed.rollout_token_batch(
                [rollout_tokens], 1, sampling, prompt_features=rollout_features
            )
        finally:
            observed.end_rollout_session()
            observed.finish_step()
        trained_after_rollout = observed.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
        if os.getenv("ARENO_E2E_BASE_REFERENCE") == "1":
            observed.ensure_roles({"ref": ModelRole("ref", os.fspath(model_path), trainable=False)})
            reference_scores = {}
            for role, label in (("actor", "actor_before"), ("ref", "base_reference"), ("actor", "actor_after")):
                reference_scores[label] = observed.score_logprobs(
                    role, [reference_tokens], features=[reference_features], microbatch_size=1
                )[0]
            torch.testing.assert_close(
                torch.tensor(reference_scores["base_reference"]), torch.tensor(reference_baseline), rtol=0, atol=1e-5
            )
            torch.testing.assert_close(
                torch.tensor(reference_scores["actor_after"]),
                torch.tensor(reference_scores["actor_before"]),
                rtol=0,
                atol=1e-5,
            )
    finally:
        observed.close()

    assert observed.rollout_versions == list(range(case.steps))
    assert observed.train_versions == list(range(1, case.steps + 1))
    assert final_rollout[0].adapter_version == case.steps
    assert all(math.isfinite(result["grad_norm"]) and result["grad_norm"] > 0 for result in observed.train_results)
    assert all(math.isfinite(result["loss"]) for result in observed.train_results)
    assert all(result["sequence_parallel"] for result in observed.train_results)
    initial = load_file(initial_path / "adapter_model.safetensors")
    trained = load_file(trained_path / "adapter_model.safetensors")
    assert initial.keys() == trained.keys()
    assert all(torch.isfinite(value).all() for value in trained.values())
    changed = {name for name in initial if not torch.equal(initial[name], trained[name])}
    assert all(any(fragment in name for name in changed) for fragment in case.changed_fragments)
    if case_name == "flash_v3":
        for target in case.targets:
            owner, component = target.rsplit(".", 1)
            assert any(f"{owner}." in name and f".{component}.lora_B.weight" in name for name in changed), target
    if full_targets:
        full_keys = {name for name in trained if ".lora_" not in name}
        assert full_keys and full_keys <= changed, "every selected full parameter must update"
        metadata = json.loads((trained_path / "adapter_config.json").read_text())
        assert metadata["peft_type"] == "ARENO_HYBRID"
        assert tuple(metadata["full_parameter_targets"]) == full_targets
    before = torch.tensor(initial_logprobs)
    after = torch.tensor(trained_logprobs)
    assert torch.isfinite(before).all() and torch.isfinite(after).all()
    assert (after - before).abs().max() > 1e-6

    reloaded = Trainer(
        _train_world_size(case),
        os.fspath(model_path),
        custom_config=_cuda_config(LoraConfig(adapter_path=os.fspath(trained_path)), case),
    )
    reloaded.init()
    try:
        reloaded_logprobs = reloaded.score_logprobs("actor", [parity_tokens], microbatch_size=1)[0]
        reloaded.export_adapter(os.fspath(tmp_path / "adapter-reloaded"))
        reloaded.begin_rollout_session()
        try:
            reloaded_rollout = reloaded.rollout_token_batch(
                [rollout_tokens], 1, sampling, prompt_features=rollout_features
            )
        finally:
            reloaded.end_rollout_session()
            reloaded.finish_step()
    finally:
        reloaded.close()

    roundtrip = load_file(tmp_path / "adapter-reloaded" / "adapter_model.safetensors")
    assert roundtrip.keys() == trained.keys()
    for name in trained:
        torch.testing.assert_close(roundtrip[name], trained[name], rtol=0, atol=0)

    (tmp_path / "score-evidence.json").write_text(
        json.dumps(
            {
                "case": case_name,
                "full_parameter_targets": full_targets,
                "changed_policy_keys": sorted(changed),
                "tokens": parity_tokens,
                "rollout_prompt_tokens": rollout_tokens,
                "image_input": image_input,
                "reference_scores": reference_scores,
                "base": base_logprobs,
                "initial": initial_logprobs,
                "trained": trained_logprobs,
                "trained_after_rollout": trained_after_rollout,
                "reloaded": reloaded_logprobs,
                "reloaded_tokens": reloaded_rollout[0].sequences[0].resp_tokens,
                "trained_tokens": final_rollout[0].sequences[0].resp_tokens,
                "adapter_roundtrip_exact": True,
                "train_results": observed.train_results,
            },
            indent=2,
        )
        + "\n"
    )
    torch.testing.assert_close(torch.tensor(reloaded_logprobs), torch.tensor(trained_logprobs), rtol=0.0, atol=1.0e-5)
    assert reloaded_rollout[0].sequences[0].resp_tokens == final_rollout[0].sequences[0].resp_tokens
    if case_name == "flash_v3":
        for rank in range(_train_world_size(case)):
            for version in range(case.steps + 1):
                assert (tmp_path / "packed-evidence" / f"trained-rank{rank}-v{version}.json").exists()
            assert (tmp_path / "packed-evidence" / f"reloaded-rank{rank}-v0.json").exists()

    state_path = trained_path / "areno_policy_state.safetensors"
    if media:
        assert state_path.exists(), "media parameters must be included in the policy artifact"
    if state_path.exists():
        state = load_file(state_path)
        initial_state = load_file(initial_path / "areno_policy_state.safetensors")
        assert any(not torch.equal(initial_state[name], value) for name, value in state.items()), (
            "auxiliary policy must update"
        )
        restored_state = load_file(tmp_path / "adapter-reloaded" / "areno_policy_state.safetensors")
        assert state.keys() == restored_state.keys()
        assert all(torch.isfinite(value).all() for value in state.values())
        for name in state:
            torch.testing.assert_close(state[name], restored_state[name], rtol=0, atol=0)
