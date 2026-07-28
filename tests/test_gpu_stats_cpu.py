"""CPU coverage for bounded GPU run telemetry (Issue #257)."""

from __future__ import annotations

import json
import time
from dataclasses import replace

import pytest

from areno.api.trainer_config import TrainerConfig
from areno.cli import gpu_stats as gpu_stats_mod
from areno.cli import train as train_cli
from areno.cli.gpu_stats import (
    GPUSample,
    GPUSampler,
    map_visible_devices,
    parse_nvidia_smi_csv,
    visible_device_selectors,
)


def _sample(
    timestamp: float,
    physical_index: int,
    *,
    uuid: str | None = None,
    mem: int = 1000,
    util: int = 50,
    temp: int = 60,
) -> GPUSample:
    return GPUSample(
        timestamp_s=timestamp,
        index=physical_index,
        physical_index=physical_index,
        uuid=uuid or f"GPU-{physical_index}",
        name="NVIDIA H100",
        mem_used_mb=mem,
        mem_total_mb=81920,
        util_pct=util,
        temp_c=temp,
    )


def _wait_for(predicate, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for sampler")
        time.sleep(0.005)


def _config(tmp_path, **overrides) -> TrainerConfig:
    values = {
        "algo": "sft",
        "ckpt": "unused",
        "dataset_path": "unused",
        "world_size": 2,
        "tp_size": 1,
        "metrics_log_dir": str(tmp_path),
        "gpu_stats": True,
        "gpu_stats_interval_s": 0.01,
        "gpu_stats_history": 4,
    }
    values.update(overrides)
    return TrainerConfig(**values)


def test_parse_multi_device_rows_with_stable_identity():
    samples = parse_nvidia_smi_csv(
        "0, GPU-aaaa, NVIDIA H100, 71234, 81920, 63, 71\n1, GPU-bbbb, NVIDIA H100, 70988, 81920, 61, 70\n"
    )

    assert [sample.index for sample in samples] == [0, 1]
    assert samples[0].physical_index == 0
    assert samples[0].uuid == "GPU-aaaa"
    assert samples[0].name == "NVIDIA H100"
    assert samples[0].mem_used_mb == 71234
    assert samples[0].temp_c == 71


def test_parse_missing_trailing_fields_and_malformed_rows():
    samples = parse_nvidia_smi_csv("garbage, row\n0, GPU-aaaa, NVIDIA H100, 71234, 81920, 63\n1, GPU-bbbb\n")

    assert [sample.index for sample in samples] == [0, 1]
    assert samples[0].util_pct == 63
    assert samples[0].temp_c is None
    assert samples[1].name is None
    assert samples[1].mem_used_mb is None


def test_visible_device_selectors_use_cuda_order_and_world_size():
    assert visible_device_selectors(2, {}) == ["0", "1"]
    assert visible_device_selectors(2, {"CUDA_VISIBLE_DEVICES": "3,1,7"}) == ["3", "1"]
    assert visible_device_selectors(1, {"CUDA_VISIBLE_DEVICES": "GPU-bbbb,GPU-aaaa"}) == ["GPU-bbbb"]
    assert visible_device_selectors(1, {"CUDA_VISIBLE_DEVICES": ""}) == []


def test_map_visible_devices_reorders_physical_indices_to_logical_indices():
    physical = [_sample(1.0, 0), _sample(1.0, 1), _sample(1.0, 2)]

    mapped = map_visible_devices(physical, ["2", "0"])

    assert [(sample.index, sample.physical_index) for sample in mapped] == [(0, 2), (1, 0)]


def test_map_visible_devices_accepts_gpu_uuid_prefixes():
    physical = [_sample(1.0, 0, uuid="GPU-aaaaaaaa"), _sample(1.0, 1, uuid="GPU-bbbbbbbb")]

    mapped = map_visible_devices(physical, ["GPU-bbbb"])

    assert len(mapped) == 1
    assert mapped[0].index == 0
    assert mapped[0].physical_index == 1


def test_empty_visible_device_list_maps_no_host_gpus():
    assert map_visible_devices([_sample(1.0, 0)], []) == []


def test_sampler_applies_mapping_in_worker_flow():
    sampler = GPUSampler(
        interval_s=0.01,
        max_history=10,
        device_selectors=["2", "0"],
        sample_fn=lambda: [_sample(time.time(), 0), _sample(time.time(), 1), _sample(time.time(), 2)],
    )

    sampler.start()
    _wait_for(lambda: len(sampler.history()) >= 2)
    sampler.stop()

    assert sampler.devices == [0, 1]
    assert {(sample.index, sample.physical_index) for sample in sampler.history()} == {(0, 2), (1, 0)}


def test_history_and_jsonl_snapshot_are_bounded(tmp_path):
    path = tmp_path / "gpu_stats.jsonl"
    tick = 0

    def sample_fn():
        nonlocal tick
        tick += 1
        return [_sample(time.time(), 0, mem=tick)]

    sampler = GPUSampler(
        interval_s=0.002,
        max_history=3,
        device_selectors=["0"],
        sample_fn=sample_fn,
        jsonl_path=str(path),
    )
    sampler.start()
    _wait_for(lambda: tick >= 8)
    sampler.stop()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(sampler.history()) == 3
    assert len(rows) == 3
    assert [row["mem_used_mb"] for row in rows] == [tick - 2, tick - 1, tick]


def test_stop_is_idempotent_and_history_stabilizes():
    sampler = GPUSampler(
        interval_s=0.05,
        max_history=10,
        sample_fn=lambda: [_sample(time.time(), 0)],
    )
    sampler.start()
    _wait_for(lambda: bool(sampler.history()))
    sampler.stop()
    snapshot = sampler.history()
    sampler.stop()

    assert not sampler.is_active()
    assert sampler.history() == snapshot


def test_missing_nvidia_smi_degrades_with_discovery_reason(monkeypatch):
    monkeypatch.setattr(gpu_stats_mod.shutil, "which", lambda _name: None)
    sampler = GPUSampler(interval_s=0.1, max_history=10)

    sampler.start()

    assert not sampler.is_active()
    assert sampler.summary()["failure"]["stage"] == "discovery"
    assert "nvidia-smi not found" in sampler.summary_text()


def test_sampler_failure_identifies_stage_and_error():
    def fail():
        raise RuntimeError("query fixture failed")

    sampler = GPUSampler(interval_s=0.002, max_history=5, sample_fn=fail)
    sampler.start()
    _wait_for(lambda: sampler.reason is not None)
    sampler.stop()

    assert sampler.summary()["failure"] == {
        "stage": "sampling",
        "message": "RuntimeError: query fixture failed",
    }
    assert "GPU stats sampling failed" in sampler.summary_text()


def test_artifact_failure_identifies_write_stage(tmp_path):
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("blocked", encoding="utf-8")
    sampler = GPUSampler(
        interval_s=0.002,
        max_history=5,
        sample_fn=lambda: [_sample(time.time(), 0)],
        jsonl_path=str(blocked_parent / "gpu.jsonl"),
    )
    sampler.start()
    _wait_for(lambda: sampler.summary()["failure"] is not None)
    sampler.stop()

    assert sampler.summary()["failure"]["stage"] == "artifact"


def test_summary_reports_logical_and_physical_devices():
    sampler = GPUSampler(interval_s=5.0, max_history=10, sample_fn=lambda: [])
    with sampler._lock:
        sampler._history.extend(
            [
                replace(_sample(10.0, 2, mem=1000, util=40, temp=60), index=0),
                replace(_sample(12.0, 2, mem=2000, util=80, temp=70), index=0),
            ]
        )

    summary = sampler.summary()

    assert summary["duration_s"] == 2.0
    assert summary["per_device"]["0"]["physical_index"] == 2
    assert summary["per_device"]["0"]["peak_mem_used_mb"] == 2000
    assert summary["per_device"]["0"]["mean_util_pct"] == 60
    assert "device 0 (physical 2)" in sampler.summary_text()


@pytest.mark.parametrize(
    ("interval", "history", "message"),
    [
        (0, 10, "interval_s must be positive"),
        (-1, 10, "interval_s must be positive"),
        (1, 0, "max_history must be positive"),
        (1, -1, "max_history must be positive"),
    ],
)
def test_constructor_rejects_non_positive_bounds(interval, history, message):
    with pytest.raises(ValueError, match=message):
        GPUSampler(interval_s=interval, max_history=history, sample_fn=lambda: [])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gpu_stats_interval_s": 0}, "gpu_stats_interval_s must be positive"),
        ({"gpu_stats_history": 0}, "gpu_stats_history must be positive"),
    ],
)
def test_public_trainer_config_rejects_non_positive_gpu_bounds(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, **overrides)


def test_disabled_config_does_not_start_sampler(tmp_path):
    config = _config(tmp_path, gpu_stats=False)

    assert train_cli._maybe_start_gpu_sampler(config) is None
    assert not list(tmp_path.glob("gpu_stats*"))


def test_cli_lifecycle_writes_bounded_artifacts_with_fake_sampler(tmp_path, monkeypatch):
    real_sampler = GPUSampler

    def fake_factory(**kwargs):
        return real_sampler(
            **kwargs,
            sample_fn=lambda: [_sample(time.time(), 0), _sample(time.time(), 1)],
        )

    monkeypatch.setattr(gpu_stats_mod, "GPUSampler", fake_factory)
    config = _config(tmp_path)

    sampler = train_cli._maybe_start_gpu_sampler(config)
    assert sampler is not None
    _wait_for(lambda: len(sampler.history()) >= 4)
    train_cli._flush_gpu_stats(sampler, config)

    jsonl_path = next(tmp_path.glob("gpu_stats.*.jsonl"))
    summary_path = next(tmp_path.glob("gpu_stats_summary.*.json"))
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert summary["devices"] == [0, 1]
    assert summary["n_samples"] == 4


def test_gpu_startup_failure_is_warning_not_training_failure(tmp_path, monkeypatch, capsys):
    class BrokenSampler:
        def __init__(self, **_kwargs):
            raise OSError("read-only artifact path")

    monkeypatch.setattr(gpu_stats_mod, "GPUSampler", BrokenSampler)

    assert train_cli._maybe_start_gpu_sampler(_config(tmp_path)) is None
    assert "WARNING: GPU stats startup failed" in capsys.readouterr().err


def test_gpu_flush_failure_does_not_escape_or_hide_training(tmp_path, capsys):
    class BrokenSampler:
        def stop(self):
            raise RuntimeError("stop failed")

        def write_summary(self, _path):
            raise OSError("summary failed")

    train_cli._flush_gpu_stats(BrokenSampler(), _config(tmp_path))

    errors = capsys.readouterr().err
    assert "GPU stats shutdown failed" in errors
    assert "GPU stats summary write failed" in errors
