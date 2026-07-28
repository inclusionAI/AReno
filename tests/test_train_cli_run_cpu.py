"""CPU tests for `areno.cli.train.run()` exit-path guarantees.

These tests verify that `api_trainer.close()` is always called via the
`try/finally` wrapper in `run()`, even when `fit()` raises an exception or
the user sends `KeyboardInterrupt`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import areno.api
from areno.cli import train as train_cli


class _TrackingTrainer:
    """Substitute Trainer that records close() calls.

    Installed as ``areno.api.Trainer`` so that ``run()``'s lazy
    ``import areno.api; areno.api.Trainer(...)`` picks it up.
    """

    _last_instance: "_TrackingTrainer | None" = None

    def __init__(self, *args, **kwargs):
        self.close_count = 0
        _TrackingTrainer._last_instance = self

    def close(self):
        self.close_count += 1


class _FakeAlgorithmTrainer:
    """Simulates the trainer returned by build_trainer."""

    def __init__(self, *, fit_side_effect=None):
        self._fit_side_effect = fit_side_effect

    def fit(self):
        if self._fit_side_effect is not None:
            raise self._fit_side_effect


def _patch_run_dependencies(monkeypatch, *, fit_side_effect=None):
    """Patch run()'s external dependencies so it can run without a real backend.

    Returns nothing; use ``_TrackingTrainer._last_instance`` to inspect.
    """
    _TrackingTrainer._last_instance = None

    # Patch module-level helpers called before the try block.
    monkeypatch.setattr(train_cli, "resolve_model_refs_for_config", lambda config: config)
    monkeypatch.setattr(train_cli, "_write_dashboard_run_config", lambda config: None)
    monkeypatch.setattr(train_cli, "_loss_fn_for_config", lambda config: lambda *a, **kw: None)
    monkeypatch.setattr(train_cli, "_reward_fn_path_for_config", lambda config: None)
    monkeypatch.setattr(train_cli, "_load_dataset_for_training", lambda *a, **kw: [])

    # Replace areno.api.Trainer so run() creates our tracking instance.
    monkeypatch.setattr(areno.api, "Trainer", _TrackingTrainer)

    # Stub out load_reward_fn (called via ``from areno.api.rewards import``).
    import areno.api.rewards as rewards_mod

    monkeypatch.setattr(rewards_mod, "load_reward_fn", lambda path: None)

    # Stub out build_trainer (called via ``from areno.api.trainer_factory import``).
    import areno.api.trainer_factory as factory_mod

    fake_trainer = _FakeAlgorithmTrainer(fit_side_effect=fit_side_effect)
    monkeypatch.setattr(
        factory_mod,
        "build_trainer",
        lambda config, **kwargs: fake_trainer,
    )


def _make_config():
    return SimpleNamespace(
        world_size=1,
        ckpt="unused",
        metrics_log_dir=None,
        dataset_path=None,
        dataset_loader_fn=None,
        model_hub="modelscope",
        areno_config=lambda: None,
        algo="sft",
    )


def test_run_calls_close_on_fit_exception(monkeypatch):
    """run() should call api_trainer.close() when fit() raises."""

    _patch_run_dependencies(monkeypatch, fit_side_effect=RuntimeError("training crashed"))

    with pytest.raises(RuntimeError, match="training crashed"):
        train_cli.run(_make_config())

    assert _TrackingTrainer._last_instance is not None
    assert _TrackingTrainer._last_instance.close_count == 1


def test_run_handles_keyboard_interrupt(monkeypatch):
    """run() should call api_trainer.close() on KeyboardInterrupt and re-raise."""

    _patch_run_dependencies(monkeypatch, fit_side_effect=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        train_cli.run(_make_config())

    assert _TrackingTrainer._last_instance is not None
    assert _TrackingTrainer._last_instance.close_count == 1