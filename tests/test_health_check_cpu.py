"""CPU tests for the startup-window health check (Issue #249).

Covers the pure-function core (`areno.api.health_check`), config validation,
default-off backward compatibility, and the `Trainer.train()` wiring using a
stub backend (no GPU / no real model). Mirrors the `*_cpu.py` convention used
by the rest of the suite.
"""

from __future__ import annotations

import json
import unittest

from areno.api.health_check import (
    EffectiveTokensCheckConfig,
    HealthCheckConfig,
    HealthCheckConfigError,
    LossChangeCheckConfig,
    RewardVarianceCheckConfig,
    SkippedBatchesCheckConfig,
    WindowSignals,
    check_effective_tokens,
    check_loss_change,
    check_reward_variance,
    check_skipped_batches,
    run_health_check,
)
from areno.api.metrics import collect_train_batch_stats
from areno.api.models import TrainSequence


def _signals(**overrides) -> WindowSignals:
    """A healthy 3-step window by default; callers override degenerate fields."""

    base = dict(
        effective_tokens_per_batch=[512, 512, 512],
        rewards=[0.1, 0.3, 0.2],
        losses=[2.3, 2.1, 2.0],
        skipped_long=0,
        total_batches=3,
        grad_zero_ratios=[0.0, 0.0, 0.0],
    )
    base.update(overrides)
    return WindowSignals(**base)


def _seq(prompt_len: int, resp_len: int, reward: float = 0.1) -> TrainSequence:
    """Build a minimal TrainSequence for collect_train_batch_stats."""

    tokens = [1] * (prompt_len + resp_len)
    prompt_mask = [True] * prompt_len + [False] * resp_len
    return TrainSequence(
        prompt_mask=prompt_mask,
        tokens=tokens,
        logprobs=[0.0] * len(tokens),
        advantages=[0.0] * prompt_len + [0.5] * resp_len,
        reward=reward,
        eos_token_id=0,
    )


class CheckLogicTest(unittest.TestCase):
    """Per-check healthy / degenerate / boundary paths."""

    def test_effective_tokens_passes(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        r = check_effective_tokens(cfg, _signals())
        self.assertEqual(r.status, "pass")
        self.assertEqual(r.stage, "trainer")
        self.assertTrue(r.metric_ref.startswith("metrics/"))

    def test_effective_tokens_zero_fails(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        r = check_effective_tokens(cfg, _signals(effective_tokens_per_batch=[0, 0, 0]))
        self.assertEqual(r.status, "fail")
        self.assertEqual(r.input, "effective_tokens.fail_if_zero")

    def test_effective_tokens_empty_fails_pointing_at_input(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        r = check_effective_tokens(cfg, WindowSignals())
        self.assertEqual(r.status, "fail")
        self.assertEqual(r.input, "effective_tokens")

    def test_effective_tokens_low_warns(self):
        cfg = HealthCheckConfig(
            enabled=True,
            startup_window_updates=3,
            effective_tokens=EffectiveTokensCheckConfig(min_per_batch=1024),
        )
        r = check_effective_tokens(cfg, _signals(effective_tokens_per_batch=[512, 512, 512]))
        self.assertEqual(r.status, "warn")

    def test_reward_variance_passes_with_variation(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        r = check_reward_variance(cfg, _signals())
        self.assertEqual(r.status, "pass")

    def test_reward_variance_constant_fails_when_required(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        r = check_reward_variance(cfg, _signals(rewards=[1.0, 1.0, 1.0]))
        self.assertEqual(r.status, "fail")
        self.assertEqual(r.input, "reward_variance.require_variation=true")

    def test_reward_variance_constant_passes_when_allowed(self):
        cfg = HealthCheckConfig(
            enabled=True,
            startup_window_updates=3,
            reward_variance=RewardVarianceCheckConfig(require_variation=False),
        )
        r = check_reward_variance(cfg, _signals(rewards=[1.0, 1.0, 1.0]))
        self.assertEqual(r.status, "pass")

    def test_reward_variance_nan_fails_with_original_error(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        sig = _signals(rewards=[0.1, float("nan"), 0.2])
        r = check_reward_variance(cfg, sig)
        self.assertEqual(r.status, "fail")
        self.assertTrue(any("non-finite" in e for e in r.errors))
        # Pure-function contract: signals must not be mutated.
        self.assertEqual(len(sig.original_errors), 0)

    def test_loss_change_passes(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        r = check_loss_change(cfg, _signals())
        self.assertEqual(r.status, "pass")

    def test_loss_change_unchanged_fails(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        r = check_loss_change(cfg, _signals(losses=[1.5, 1.5, 1.5]))
        self.assertEqual(r.status, "fail")

    def test_loss_change_single_step_warns(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=1)
        r = check_loss_change(cfg, WindowSignals(losses=[1.5]))
        self.assertEqual(r.status, "warn")
        self.assertEqual(r.input, "startup_window_updates")

    def test_loss_change_nan_fails(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        sig = _signals(losses=[2.0, float("nan")])
        r = check_loss_change(cfg, sig)
        self.assertEqual(r.status, "fail")
        self.assertTrue(any("non-finite" in e for e in r.errors))
        # Pure-function contract: signals must not be mutated.
        self.assertEqual(len(sig.original_errors), 0)

    def test_skipped_passes(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        r = check_skipped_batches(cfg, _signals())
        self.assertEqual(r.status, "pass")

    def test_skipped_zero_denominator_fails_at_input(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        r = check_skipped_batches(cfg, WindowSignals(total_batches=0))
        self.assertEqual(r.status, "fail")
        self.assertEqual(r.input, "total_batches")

    def test_skipped_high_ratio_fails(self):
        cfg = HealthCheckConfig(
            enabled=True,
            startup_window_updates=2,
            skipped_batches=SkippedBatchesCheckConfig(max_ratio_fail=0.1),
        )
        sig = _signals(skipped_long=5, total_batches=10, grad_zero_ratios=[0.0, 0.0])
        r = check_skipped_batches(cfg, sig)
        self.assertEqual(r.status, "fail")
        self.assertEqual(r.input, "skipped_batches.max_ratio_fail")

    def test_skipped_grad_zero_spike_fails(self):
        cfg = HealthCheckConfig(
            enabled=True,
            startup_window_updates=2,
            skipped_batches=SkippedBatchesCheckConfig(max_grad_zero_ratio_fail=0.5),
        )
        sig = _signals(grad_zero_ratios=[0.9, 0.0])
        r = check_skipped_batches(cfg, sig)
        self.assertEqual(r.status, "fail")


class AggregationTest(unittest.TestCase):
    """Aggregate takes the most severe status; on_fail governs raise behavior."""

    def test_healthy_window_pass(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        report = run_health_check(cfg, _signals(), completed_at_step=3)
        self.assertIsNotNone(report)
        self.assertEqual(report.summary, "pass")
        self.assertTrue(all(c.status == "pass" for c in report.checks))

    def test_degenerate_window_fail(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        sig = _signals(
            effective_tokens_per_batch=[0, 0, 0],
            rewards=[0.0, 0.0, 0.0],
            losses=[1.5, 1.5, 1.5],
        )
        report = run_health_check(cfg, sig, completed_at_step=3)
        self.assertEqual(report.summary, "fail")
        names = {c.name: c for c in report.checks}
        self.assertEqual(names["effective_tokens"].status, "fail")
        self.assertEqual(names["reward_variance"].status, "fail")
        self.assertEqual(names["loss_change"].status, "fail")

    def test_report_json_shape_and_no_sample_text(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        report = run_health_check(cfg, _signals(), completed_at_step=3)
        payload = report.to_json()
        self.assertIn("summary", payload)
        self.assertIn("checks", payload)
        self.assertIn("window", payload)
        self.assertEqual(payload["window"]["completed_at_step"], 3)
        # No training-sample / token text leaks into the artifact.
        blob = json.dumps(payload)
        self.assertNotIn("sample", blob)
        self.assertNotIn("token_text", blob)

    def test_each_check_carries_stage_and_metric_ref(self):
        cfg = HealthCheckConfig(enabled=True, startup_window_updates=3)
        report = run_health_check(cfg, _signals(), completed_at_step=3)
        for c in report.checks:
            self.assertIn(c.stage, ("trainer", "rollout"))
            self.assertTrue(c.metric_ref.startswith("metrics/"))


class ConfigValidationTest(unittest.TestCase):
    """Invalid config raises early with a message naming the offending field."""

    def test_window_must_be_positive(self):
        with self.assertRaises(HealthCheckConfigError) as ctx:
            HealthCheckConfig(enabled=True, startup_window_updates=0)
        self.assertIn("startup_window_updates", str(ctx.exception))

    def test_on_fail_enum(self):
        with self.assertRaises(HealthCheckConfigError) as ctx:
            HealthCheckConfig(enabled=True, on_fail="bogus")
        self.assertIn("on_fail", str(ctx.exception))

    def test_ratio_must_be_in_unit_interval(self):
        with self.assertRaises(HealthCheckConfigError) as ctx:
            HealthCheckConfig(
                enabled=True,
                skipped_batches=SkippedBatchesCheckConfig(max_ratio_warn=1.5),
            )
        self.assertIn("max_ratio_warn", str(ctx.exception))

    def test_warn_threshold_must_be_at_least_fail(self):
        with self.assertRaises(HealthCheckConfigError):
            HealthCheckConfig(
                enabled=True,
                loss_change=LossChangeCheckConfig(min_abs_delta_warn=0.0, min_abs_delta_fail=1.0),
            )

    def test_loss_mode_enum(self):
        with self.assertRaises(HealthCheckConfigError) as ctx:
            HealthCheckConfig(
                enabled=True,
                loss_change=LossChangeCheckConfig(mode="bogus"),
            )
        self.assertIn("mode", str(ctx.exception))


class DisabledDefaultTest(unittest.TestCase):
    """Default `enabled=False` produces no report (backward compatibility)."""

    def test_disabled_returns_none(self):
        cfg = HealthCheckConfig()  # enabled defaults to False
        self.assertFalse(cfg.enabled)
        self.assertIsNone(run_health_check(cfg, _signals(), completed_at_step=1))


class TrainerHookTest(unittest.TestCase):
    """`Trainer.train()` feeds the checker via a stub backend (no GPU)."""

    def _make_trainer_with_checker(self, cfg, tmp_path):
        import areno.api.trainer as trainer_mod
        from areno.api.context import Context

        trainer = trainer_mod.Trainer(world_size=1, model_path="unused")
        # Stub MetricsRecorder so the test never builds a real TensorBoard
        # writer (which depends on torch and has version-specific behavior).
        # The health-checker only needs `log_dir` + `add_scalar` from it.
        tmp_path = str(tmp_path)

        class FakeMetrics:
            log_dir = tmp_path
            scalars = []

            def add_scalar(self, tag, value, step):
                self.scalars.append((tag, value, step))

            def record_train_step(self, *, step, train_result, train_batch, timings=None):
                pass

            def record_dashboard_state(self, **kwargs):
                pass

            def close(self):
                pass

        trainer._metrics = FakeMetrics()
        # Real Context (no torch dependency) so global_step advances normally.
        trainer._ctx = Context(world_size=1, model_path="unused", tokenizer=None)
        trainer.configure_health_check(cfg)
        return trainer

    def test_observe_evaluates_at_window_end_and_writes_artifact(self):
        import pathlib
        import tempfile

        cfg = HealthCheckConfig(enabled=True, startup_window_updates=2)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer_with_checker(cfg, tmp)

            class BackendStub:
                def train(self, ctx, batch, loss_fn, mini_bs, grad_accum):
                    # Mirror ArenoBackend.train's flat result shape: loss and
                    # each metric are top-level scalars (no nested "metrics" key).
                    return {"loss": 2.0, "grad_zero_ratio": 0.0}

            trainer._backend = BackendStub()
            batch = [_seq(prompt_len=2, resp_len=4, reward=0.3)]
            trainer.train(batch, lambda data, lp: 0.0, mini_bs=1)
            trainer.train(batch, lambda data, lp: 0.0, mini_bs=1)
            # Window filled → artifact written under health_check/.
            artifacts = list(pathlib.Path(tmp, "health_check").glob("*.json"))
            self.assertEqual(len(artifacts), 1)
            payload = json.loads(artifacts[0].read_text())
            self.assertIn(payload["summary"], ("pass", "warn", "fail"))

    def test_disabled_checker_produces_nothing(self):
        import pathlib
        import tempfile

        cfg = HealthCheckConfig(enabled=False)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer_with_checker(cfg, tmp)

            class BackendStub:
                def train(self, ctx, batch, loss_fn, mini_bs, grad_accum):
                    return {"loss": 2.0}

            trainer._backend = BackendStub()
            batch = [_seq(prompt_len=2, resp_len=4)]
            trainer.train(batch, lambda data, lp: 0.0, mini_bs=1)
            self.assertFalse(pathlib.Path(tmp, "health_check").exists())


class CollectStatsReuseTest(unittest.TestCase):
    """Sanity: the reused `collect_train_batch_stats` yields the expected fields."""

    def test_response_len_and_rewards_extracted(self):
        batch = [
            _seq(prompt_len=3, resp_len=5, reward=0.7),
            _seq(prompt_len=2, resp_len=4, reward=0.1),
        ]
        stats = collect_train_batch_stats(batch)
        self.assertEqual(stats["response_len"], [5, 4])
        self.assertEqual(stats["rewards"], [0.7, 0.1])
        self.assertEqual(stats["skipped_long"], 0)


if __name__ == "__main__":
    unittest.main()
