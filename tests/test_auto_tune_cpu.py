from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import areno.cli.auto_tune as auto_tune
from areno.api.trainer_config import PolicyTrainerConfig, TrainerConfig
from areno.cli.auto_tune import (
    AutoTuneCandidate,
    AutoTuneMeasurement,
    _dummy_policy_loss,
    _dummy_prompt_tokens,
    _dummy_response_tokens,
    _dummy_token_budgets,
    _dummy_train_rows,
    _is_oom_error,
    _peak_cuda_memory_fraction,
    _probe_devices,
    _reset_cuda_peak_stats,
    _rollout_candidates,
    _run_dummy_probe,
    _train_candidates,
    auto_tune_config,
    enumerate_candidates,
)
from areno.engine.api import ArenoEngine
from areno.engine.modeling import _zero_model_parameters
from areno.engine.protocol import Op


def test_enumerate_candidates_grows_rollout_and_train_knobs() -> None:
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        batch_size=4,
        n_samples=4,
        mini_bs=4,
    )

    candidates = enumerate_candidates(config)

    assert candidates[0] == AutoTuneCandidate(
        tp_size=4,
        batch_size=1,
        n_samples=4,
        mini_bs=1,
        max_running_prompts=1,
        adam_8bit=False,
        keep_rollout_state=False,
    )
    assert any(candidate.max_running_prompts == 16 for candidate in candidates)
    assert len(_rollout_candidates(config)) <= 16
    assert candidates == sorted(
        candidates,
        key=lambda item: (
            item.max_running_prompts,
            item.train_rows,
            item.mini_bs,
            item.batch_size,
            item.n_samples,
            -item.tp_size,
            not item.adam_8bit,
            item.keep_rollout_state,
        ),
    )


def test_enumerate_candidates_limits_policy_samples_and_mini_bs_values() -> None:
    config = PolicyTrainerConfig(
        algo="grpo",
        ckpt="actor",
        dataset_path="dataset",
        batch_size=4,
        n_samples=8,
        mini_bs=16,
    )

    candidates = enumerate_candidates(config)

    assert {candidate.n_samples for candidate in candidates} == {8}
    assert {candidate.mini_bs for candidate in candidates} <= {1, 2, 4, 8, 16}
    assert all(candidate.mini_bs <= candidate.batch_size * candidate.n_samples for candidate in candidates)
    assert all(not candidate.keep_rollout_state for candidate in candidates)
    assert {candidate.adam_8bit for candidate in candidates} == {False}
    assert len(_rollout_candidates(config)) <= 16
    assert all(len(_train_candidates(config, rollout)) <= 32 for rollout in _rollout_candidates(config))


def test_enumerate_candidates_uses_user_tp_and_power_of_two_numeric_values() -> None:
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        world_size=12,
        tp_size=3,
        batch_size=6,
        n_samples=8,
        mini_bs=12,
    )

    candidates = enumerate_candidates(config)

    assert {candidate.tp_size for candidate in candidates} == {3}
    assert {candidate.batch_size for candidate in candidates} <= {1, 2, 4}
    assert {candidate.n_samples for candidate in candidates} == {8}
    assert {candidate.mini_bs for candidate in candidates} <= {1, 2, 4, 8}
    assert all(candidate.max_running_prompts & (candidate.max_running_prompts - 1) == 0 for candidate in candidates)


def test_train_candidates_use_rollout_tp_and_vary_batch_and_mini_batch() -> None:
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        world_size=8,
        batch_size=4,
        n_samples=7,
        mini_bs=16,
    )
    rollout = AutoTuneCandidate(
        tp_size=4,
        batch_size=4,
        n_samples=7,
        mini_bs=1,
        max_running_prompts=8,
        adam_8bit=False,
        keep_rollout_state=False,
    )

    candidates = _train_candidates(config, rollout)

    assert {candidate.max_running_prompts for candidate in candidates} == {8}
    assert {candidate.n_samples for candidate in candidates} == {7}
    assert {candidate.tp_size for candidate in candidates} == {4}
    assert {candidate.batch_size for candidate in candidates} <= {1, 2, 4}
    assert {candidate.mini_bs for candidate in candidates} <= {1, 2, 4, 8, 16}


def test_auto_max_samples_caps_rollout_prompts_and_train_batch_size() -> None:
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        world_size=8,
        tp_size=4,
        batch_size=64,
        n_samples=8,
        mini_bs=16,
    )

    rollout_candidates = _rollout_candidates(config, auto_max_samples=32)
    train_candidates = [
        candidate
        for rollout in rollout_candidates
        for candidate in _train_candidates(config, rollout, auto_max_samples=32)
    ]

    assert max(candidate.max_running_prompts for candidate in rollout_candidates) == 32
    assert max(candidate.batch_size for candidate in train_candidates) == 4
    assert all(candidate.batch_size * candidate.n_samples <= 32 for candidate in train_candidates)


def test_rollout_candidates_vary_running_prompts_with_fixed_tp() -> None:
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        world_size=8,
        tp_size=4,
        batch_size=8,
        n_samples=8,
        mini_bs=4,
    )

    candidates = _rollout_candidates(config)

    assert {candidate.tp_size for candidate in candidates} == {4}
    assert candidates == sorted(
        candidates,
        key=lambda item: (
            item.max_running_prompts,
            item.train_rows,
            item.batch_size,
            item.n_samples,
            item.mini_bs,
            item.keep_rollout_state,
        ),
        reverse=True,
    )


def test_auto_tune_selects_largest_candidate_under_memory_fraction() -> None:
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        batch_size=4,
        n_samples=4,
        mini_bs=4,
    )

    def fake_probe(_config, candidate, stage):
        peak = 0.1 + candidate.max_running_prompts * 0.05
        return AutoTuneMeasurement(candidate=candidate, peak_mem_frac=peak, ok=peak < 0.95)

    result = auto_tune_config(config, mem_frac=0.5, probe_fn=fake_probe)

    assert isinstance(result.config, PolicyTrainerConfig)
    assert result.config.resolved_max_running_prompts() == 8
    assert result.config.n_samples == 4
    assert result.config.batch_size in {1, 2, 4}
    assert result.config.keep_rollout_state is False
    assert result.measurement is not None
    assert result.measurement.peak_mem_frac <= 0.5


def test_auto_tune_probes_sparse_rollout_and_train_candidates() -> None:
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        batch_size=8,
        n_samples=8,
        mini_bs=16,
    )
    probed = []

    def fake_probe(_config, candidate, stage):
        probed.append(candidate)
        peak = 0.2 if candidate.max_running_prompts <= 32 else 0.95
        return AutoTuneMeasurement(candidate=candidate, peak_mem_frac=peak, ok=True)

    result = auto_tune_config(config, mem_frac=0.9, probe_fn=fake_probe)

    assert len(_rollout_candidates(config)) <= 16
    assert all(len(_train_candidates(config, rollout)) <= 32 for rollout in _rollout_candidates(config))
    assert len(probed) <= 11
    assert result.config.max_running_prompts == 32
    assert result.config.batch_size == 4


def test_auto_tune_train_tunes_mini_bs_after_fixed_rollout_batch() -> None:
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        batch_size=8,
        n_samples=8,
        mini_bs=16,
    )
    train_probes = []

    def fake_probe(_config, candidate, stage):
        if stage == "rollout":
            return AutoTuneMeasurement(candidate=candidate, peak_mem_frac=0.1, ok=True)
        train_probes.append(candidate)
        if candidate.mini_bs > 2:
            return AutoTuneMeasurement(candidate=candidate, peak_mem_frac=1.0, ok=False, error="oom")
        return AutoTuneMeasurement(candidate=candidate, peak_mem_frac=0.2, ok=True)

    result = auto_tune_config(config, mem_frac=0.9, auto_max_samples=64, probe_fn=fake_probe)

    assert result.config.batch_size == 8
    assert result.config.mini_bs == 2
    assert [candidate.mini_bs for candidate in train_probes] == [16, 8, 4, 2]
    assert {candidate.batch_size for candidate in train_probes} == {8}


def test_auto_tune_preserves_user_adam_8bit_choice() -> None:
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        batch_size=4,
        n_samples=8,
        mini_bs=4,
        adam_8bit=True,
    )

    def fake_probe(_config, candidate, stage):
        del stage
        return AutoTuneMeasurement(candidate=candidate, peak_mem_frac=0.1, ok=True)

    result = auto_tune_config(config, mem_frac=0.9, probe_fn=fake_probe)

    assert result.config.adam_8bit is True
    assert {measurement.candidate.adam_8bit for measurement in result.measurements} == {True}


def test_auto_tune_logs_progress(monkeypatch) -> None:
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        batch_size=2,
        n_samples=8,
        mini_bs=2,
    )

    def fake_probe(_config, candidate, stage):
        return AutoTuneMeasurement(candidate=candidate, peak_mem_frac=0.1, ok=True)

    messages = []
    monkeypatch.setattr(auto_tune.logger, "info", lambda message, *args: messages.append(message % args))

    auto_tune_config(config, mem_frac=0.5, probe_fn=fake_probe)

    text = "\n".join(messages)
    assert "auto tune start" in text
    assert "auto tune rollout stage start" in text
    assert "auto tune train stage start" in text
    assert text.index("auto tune rollout stage start") < text.index("auto tune train stage start")
    assert "probe start" in text
    assert "auto tune probe result" in text
    assert "auto tune selected" in text


def test_auto_tune_does_not_change_user_tp_size() -> None:
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        world_size=8,
        tp_size=4,
        batch_size=1,
        n_samples=1,
        mini_bs=1,
    )

    def fake_probe(_config, candidate, stage):
        return AutoTuneMeasurement(candidate=candidate, peak_mem_frac=0.4, ok=True)

    result = auto_tune_config(config, mem_frac=0.9, probe_fn=fake_probe)

    assert result.config.tp_size == 4
    assert result.measurement is not None
    assert {measurement.candidate.tp_size for measurement in result.measurements} == {4}


def test_auto_tune_surfaces_user_tp_compatibility_errors() -> None:
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        world_size=8,
        tp_size=8,
        batch_size=4,
        n_samples=8,
        mini_bs=4,
    )

    def fake_probe(_config, candidate, stage):
        del stage
        return AutoTuneMeasurement(
            candidate=candidate,
            peak_mem_frac=1.0,
            ok=False,
            error="num_key_value_heads must be divisible by tp_size",
        )

    with pytest.raises(RuntimeError, match="num_key_value_heads must be divisible by tp_size"):
        auto_tune_config(config, mem_frac=0.9, probe_fn=fake_probe)


def test_auto_tune_treats_probe_worker_sigkill_as_oom() -> None:
    message = "worker exited without reporting result during Op.TRAIN: rank 0 pid 13907 exitcode -9"

    assert _is_oom_error(message)


def test_auto_tune_rejects_non_rollout_configs() -> None:
    config = TrainerConfig(algo="sft", ckpt="actor", dataset_path="dataset")

    try:
        auto_tune_config(config, probe_fn=lambda _config, candidate, stage: AutoTuneMeasurement(candidate, 0.1, True))
    except ValueError as exc:
        assert "--auto currently tunes rollout-based trainers only" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_zero_model_parameters_makes_dummy_load_deterministic() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    for param in model.parameters():
        param.data.normal_()

    _zero_model_parameters(model)

    for param in model.parameters():
        assert torch.count_nonzero(param).item() == 0


def test_probe_devices_uses_world_size_visible_cuda_devices(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assert _probe_devices(2) == [0, 1]


def test_probe_devices_fails_when_world_size_exceeds_visible_cuda_devices(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    try:
        _probe_devices(8)
    except RuntimeError as exc:
        assert "world_size=8" in str(exc)
        assert "visible_cuda_devices=4" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_reset_cuda_peak_stats_initializes_each_device_by_index(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(("set", device)))
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: calls.append(("empty", args, kwargs)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(("sync", device)))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: calls.append(("reset", device)))

    _reset_cuda_peak_stats([0, 1])

    assert calls == [
        ("set", 0),
        ("empty", ((),), {"device": "cuda:0"}),
        ("sync", 0),
        ("reset", 0),
        ("set", 1),
        ("empty", ((),), {"device": "cuda:1"}),
        ("sync", 1),
        ("reset", 1),
    ]


def test_peak_cuda_memory_fraction_uses_device_indices(monkeypatch) -> None:
    class Props:
        total_memory = 100

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: Props())
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: {0: 40, 1: 80}[device])

    assert _peak_cuda_memory_fraction([0, 1]) == 0.8


class _TinyTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [11, 12, 13]


def test_dummy_probe_rows_respect_prompt_response_and_context_lengths() -> None:
    tokenizer = _TinyTokenizer()
    prompt_len, response_len = _dummy_token_budgets(max_prompt_tokens=7, max_new_tokens=9, max_context_len=12)
    prompt = _dummy_prompt_tokens(tokenizer, max_prompt_tokens=prompt_len)
    response = _dummy_response_tokens(tokenizer, response_len=response_len)
    rows = _dummy_train_rows(prompt_tokens=prompt, response_tokens=response, eos_token_id=0, target_rows=2)

    assert len(prompt) == 7
    assert len(response) == 5
    assert len(rows) == 2
    assert all(len(row.tokens) == 12 for row in rows)
    assert all(row.prompt_mask == [True] * 7 + [False] * 5 for row in rows)


def test_dummy_probe_uses_context_len_when_context_is_set() -> None:
    tokenizer = _TinyTokenizer()

    prompt_len, response_len = _dummy_token_budgets(max_prompt_tokens=7, max_new_tokens=6, max_context_len=100)
    prompt = _dummy_prompt_tokens(tokenizer, max_prompt_tokens=prompt_len)
    response = _dummy_response_tokens(tokenizer, response_len=response_len)

    assert len(prompt) == 7
    assert len(response) == 93
    assert len(prompt) + len(response) == 100


def test_dummy_probe_uses_prompt_plus_new_tokens_without_context() -> None:
    prompt_len, response_len = _dummy_token_budgets(max_prompt_tokens=7, max_new_tokens=6, max_context_len=None)

    assert prompt_len == 7
    assert response_len == 6


def test_rollout_dummy_probe_skips_train(monkeypatch) -> None:
    import areno.api

    calls = []

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def init(self):
            calls.append("init")

        def get_tokenizer(self):
            return _TinyTokenizer()

        def probe_rollout_cache(self, **kwargs):
            calls.append(("probe_rollout_cache", kwargs))
            return 0.25

        def train(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("rollout auto tune probe must not run train")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(areno.api, "Trainer", FakeTrainer)
    monkeypatch.setattr(auto_tune, "_probe_devices", lambda world_size: [0])
    monkeypatch.setattr(auto_tune, "_reset_cuda_peak_stats", lambda devices: calls.append(("reset", devices)))
    monkeypatch.setattr(auto_tune, "_peak_cuda_memory_fraction", lambda devices: 0.0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(("sync", device)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    config = PolicyTrainerConfig(algo="gspo", ckpt="actor", dataset_path="dataset")
    candidate = AutoTuneCandidate(
        tp_size=1,
        batch_size=1,
        n_samples=4,
        mini_bs=1,
        max_running_prompts=4,
        adam_8bit=True,
        keep_rollout_state=False,
    )

    peak = _run_dummy_probe(config, candidate, stage="rollout")

    assert peak == 0.25
    assert any(call[0] == "probe_rollout_cache" for call in calls if isinstance(call, tuple))
    assert "close" in calls


def test_train_dummy_probe_drops_rollout_state_before_train(monkeypatch) -> None:
    import areno.api

    calls = []

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def init(self):
            calls.append("init")

        def get_tokenizer(self):
            return _TinyTokenizer()

        def begin_rollout_session(self):
            calls.append("begin_rollout_session")

        def probe_rollout_cache(self, **kwargs):
            calls.append(("probe_rollout_cache", kwargs))
            return 0.5

        def end_rollout_session(self):
            calls.append("end_rollout_session")

        def train(self, *args, **kwargs):
            calls.append("train")
            return {"auto_tune_worker_peak_mem_frac": 0.42}

        def close(self):
            calls.append("close")

    monkeypatch.setattr(areno.api, "Trainer", FakeTrainer)
    monkeypatch.setattr(auto_tune, "_probe_devices", lambda world_size: [0])
    monkeypatch.setattr(auto_tune, "_reset_cuda_peak_stats", lambda devices: calls.append(("reset", devices)))
    monkeypatch.setattr(auto_tune, "_peak_cuda_memory_fraction", lambda devices: 0.0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(("sync", device)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    config = PolicyTrainerConfig(algo="gspo", ckpt="actor", dataset_path="dataset")
    candidate = AutoTuneCandidate(
        tp_size=1,
        batch_size=1,
        n_samples=4,
        mini_bs=1,
        max_running_prompts=4,
        adam_8bit=False,
        keep_rollout_state=False,
    )

    peak = _run_dummy_probe(config, candidate, stage="train")

    assert peak == 0.42
    assert calls.index("begin_rollout_session") < calls.index("end_rollout_session") < calls.index("train")
    assert "close" in calls


def test_train_dummy_probe_preserves_original_oom_when_cleanup_fails(monkeypatch) -> None:
    import areno.api

    calls = []

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def init(self):
            calls.append("init")

        def get_tokenizer(self):
            return _TinyTokenizer()

        def begin_rollout_session(self):
            calls.append("begin_rollout_session")

        def probe_rollout_cache(self, **kwargs):
            calls.append(("probe_rollout_cache", kwargs))
            return 0.5

        def end_rollout_session(self):
            calls.append("end_rollout_session")
            raise RuntimeError("worker exited without reporting result during Op.ROLLOUT_SESSION_END")

        def train(self, *args, **kwargs):
            del args, kwargs
            calls.append("train")
            raise RuntimeError("CUDA out of memory while probing train")

        def close(self):
            calls.append("close")
            raise RuntimeError("worker exited during close")

    monkeypatch.setattr(areno.api, "Trainer", FakeTrainer)
    monkeypatch.setattr(auto_tune, "_probe_devices", lambda world_size: [0])
    monkeypatch.setattr(auto_tune, "_reset_cuda_peak_stats", lambda devices: calls.append(("reset", devices)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    config = PolicyTrainerConfig(algo="gspo", ckpt="actor", dataset_path="dataset")
    candidate = AutoTuneCandidate(
        tp_size=1,
        batch_size=1,
        n_samples=4,
        mini_bs=1,
        max_running_prompts=4,
        adam_8bit=False,
        keep_rollout_state=False,
    )

    measurement = auto_tune.probe_candidate_with_dummy_run(config, candidate, "train")

    assert measurement.ok is False
    assert measurement.error is not None
    assert "CUDA out of memory" in measurement.error
    assert "ROLLOUT_SESSION_END" not in measurement.error
    assert calls == [
        ("reset", [0]),
        "init",
        "begin_rollout_session",
        ("probe_rollout_cache", {"max_new_tokens": 3071, "max_running_prompts": 4, "max_prompt_len": 1024}),
        "end_rollout_session",
        "train",
        "close",
        "empty_cache",
    ]


def test_dummy_policy_loss_aligns_next_token_logprobs_with_response_advantages() -> None:
    pack = {
        "prompt_mask": torch.tensor([[True, True, False, False]]),
        "advantages": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
    }
    logprobs = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)

    loss, stats = _dummy_policy_loss(pack, logprobs)

    assert torch.isclose(loss, torch.tensor(-0.25))
    assert "auto_tune_dummy_loss" in stats
    loss.backward()
    assert logprobs.grad is not None


def test_engine_rollout_cache_probe_sizes_payload_without_decoding() -> None:
    calls = []

    class FakeCluster:
        def call(self, op, payload):
            calls.append((op, payload))
            return [0.2, 0.4]

    engine = object.__new__(ArenoEngine)
    engine.config = SimpleNamespace(dp_size=2, runtime=SimpleNamespace(kv_block_size=16))
    engine.cluster = FakeCluster()

    peak = engine.probe_rollout_cache(max_new_tokens=20, max_running_prompts=5, max_prompt_len=33)

    assert len(calls) == 1
    op, payload = calls[0]
    assert op is Op.PROBE_ROLLOUT_CACHE
    assert payload.max_running_seqs == 3
    assert payload.max_cache_len == 53
    assert payload.max_blocks_per_seq == 4
    assert payload.num_blocks == 12
    assert peak == 0.4
