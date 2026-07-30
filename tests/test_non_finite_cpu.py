"""CPU-only tests for non-finite value detection and reporting (Issue #238).

Covers: loss/param/grad/optimizer detection, NonFiniteReport output,
NonFiniteTrainingError, cross-rank all_reduce_non_finite_flag (no dist),
emit_non_finite_report skip/terminate paths, RuntimeConfig defaults,
trainer-level _check_non_finite_values, and CLI flag wiring.
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from areno.engine.config import RuntimeConfig
from areno.engine.runtime.non_finite import (
    NonFiniteEvent,
    NonFiniteReport,
    NonFiniteTrainingError,
    all_reduce_non_finite_flag,
    check_loss_non_finite,
    detect_non_finite,
    emit_non_finite_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_opt():
    """Create a tiny linear model + Adam optimizer for testing."""
    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, opt


# ---------------------------------------------------------------------------
# check_loss_non_finite
# ---------------------------------------------------------------------------

class CheckLossNonFiniteTest(unittest.TestCase):
    """Fast per-step loss NaN/Inf check."""

    def test_normal_loss_returns_false(self):
        """A finite scalar loss must pass the check."""
        loss = torch.tensor(1.5, requires_grad=True)
        self.assertFalse(check_loss_non_finite(loss))

    def test_nan_loss_returns_true(self):
        """NaN loss should be detected immediately."""
        loss = torch.tensor(float("nan"), requires_grad=True)
        self.assertTrue(check_loss_non_finite(loss))

    def test_inf_loss_returns_true(self):
        """Inf loss should be detected immediately."""
        loss = torch.tensor(float("inf"), requires_grad=True)
        self.assertTrue(check_loss_non_finite(loss))

    def test_neg_inf_loss_returns_true(self):
        """Negative Inf loss should be detected."""
        loss = torch.tensor(float("-inf"), requires_grad=True)
        self.assertTrue(check_loss_non_finite(loss))

    def test_multi_element_tensor(self):
        """Check works on multi-element tensors (takes .item() of bool)."""
        loss = torch.tensor([1.0, float("nan"), 3.0])
        self.assertTrue(check_loss_non_finite(loss))


# ---------------------------------------------------------------------------
# detect_non_finite
# ---------------------------------------------------------------------------

class DetectNonFiniteTest(unittest.TestCase):
    """Deep detection of NaN/Inf in model params, grads, optimizer state."""

    def test_normal_training_no_report(self):
        """Normal forward+backward should produce no report."""
        model, opt = _make_model_opt()
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        report = detect_non_finite(model, opt, loss, grad_norm=1.0, step=10, lr=1e-3)
        self.assertIsNone(report)

    def test_nan_loss_produces_report(self):
        """NaN loss must trigger a report with loss event."""
        model, opt = _make_model_opt()
        loss = torch.tensor(float("nan"), requires_grad=True)
        report = detect_non_finite(model, opt, loss, grad_norm=0.0, step=1, lr=1e-3)
        self.assertIsNotNone(report)
        self.assertTrue(math.isnan(report.loss_value))
        self.assertEqual(report.phase, "actor")

    def test_inf_param_detected(self):
        """Inf in a parameter must appear as a param event."""
        model, opt = _make_model_opt()
        with torch.no_grad():
            model.weight[0, 0] = float("inf")
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        report = detect_non_finite(model, opt, loss, grad_norm=1.0, step=50, lr=1e-4)
        self.assertIsNotNone(report)
        param_events = [e for e in report.events if not e.is_gradient]
        self.assertTrue(any(e.inf_count > 0 for e in param_events))

    def test_nan_grad_detected(self):
        """NaN in a gradient must appear as a grad event."""
        model, opt = _make_model_opt()
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        # Inject NaN into gradient
        for p in model.parameters():
            if p.grad is not None:
                p.grad[0, 0] = float("nan")
                break
        report = detect_non_finite(model, opt, loss, grad_norm=0.0, step=5, lr=1e-3)
        self.assertIsNotNone(report)
        grad_events = [e for e in report.events if e.is_gradient]
        self.assertTrue(any(e.nan_count > 0 for e in grad_events))

    def test_grad_explosion_detected(self):
        """An extremely large grad norm should be flagged above threshold."""
        model, opt = _make_model_opt()
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.fill_(1e8)
        report = detect_non_finite(
            model, opt, loss, grad_norm=1e8, step=200, lr=1e-3,
            grad_norm_threshold=1e6,
        )
        self.assertIsNotNone(report)

    def test_report_has_causes_and_suggestions(self):
        """Report should populate causes and suggestions after analyse()."""
        model, opt = _make_model_opt()
        loss = torch.tensor(float("nan"), requires_grad=True)
        report = detect_non_finite(model, opt, loss, grad_norm=0.0, step=1, lr=1e-3)
        self.assertIsNotNone(report)
        self.assertTrue(len(report.causes) > 0)
        self.assertTrue(len(report.suggestions) > 0)

    def test_phase_passed_through(self):
        """The phase label should appear in the report."""
        model, opt = _make_model_opt()
        loss = torch.tensor(float("inf"), requires_grad=True)
        report = detect_non_finite(model, opt, loss, grad_norm=0.0, step=1, lr=1e-3, phase="critic")
        self.assertIsNotNone(report)
        self.assertEqual(report.phase, "critic")


# ---------------------------------------------------------------------------
# NonFiniteReport output
# ---------------------------------------------------------------------------

class NonFiniteReportOutputTest(unittest.TestCase):
    """Report serialization: to_dict, to_json_dict, to_json_file, format_terminal."""

    def _make_report(self):
        """Build a minimal report for output tests."""
        report = NonFiniteReport(
            step=42,
            loss_value=float("nan"),
            phase="actor",
            events=[
                NonFiniteEvent(name="weight", layer="0", nan_count=3, inf_count=1, total_elements=8),
            ],
            learning_rate=1e-4,
            global_grad_norm=0.0,
        )
        report.analyse()
        return report

    def test_to_dict_returns_numeric_metrics(self):
        """to_dict should return float-safe metrics for _merge_metrics."""
        report = self._make_report()
        d = report.to_dict()
        self.assertIsInstance(d["non_finite_step"], float)
        self.assertEqual(d["non_finite_step"], 42.0)
        self.assertEqual(d["non_finite_total_nan"], 3.0)
        self.assertEqual(d["non_finite_total_inf"], 1.0)

    def test_to_json_dict_has_full_structure(self):
        """to_json_dict should include events, causes, suggestions."""
        report = self._make_report()
        d = report.to_json_dict()
        self.assertEqual(d["step"], 42)
        self.assertEqual(d["phase"], "actor")
        self.assertEqual(len(d["events"]), 1)
        self.assertEqual(d["events"][0]["name"], "weight")
        self.assertIn("causes", d)
        self.assertIn("suggestions", d)

    def test_to_json_file_writes_valid_json(self):
        """to_json_file should write a JSON file and return its path."""
        import json

        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = report.to_json_file(output_dir=tmpdir)
            self.assertTrue(os.path.exists(fpath))
            with open(fpath) as f:
                data = json.load(f)
            self.assertEqual(data["step"], 42)

    def test_format_terminal_includes_key_sections(self):
        """format_terminal should include LOCATION, ANOMALIES, CONTEXT, etc."""
        report = self._make_report()
        text = report.format_terminal()
        self.assertIn("Non-Finite Value Training Report", text)
        self.assertIn("Step: 42", text)
        self.assertIn("ANOMALIES DETECTED", text)
        self.assertIn("LIKELY CAUSES", text)
        self.assertIn("SUGGESTED FIXES", text)


# ---------------------------------------------------------------------------
# all_reduce_non_finite_flag (single-process, no dist)
# ---------------------------------------------------------------------------

class AllReduceNonFiniteFlagTest(unittest.TestCase):
    """Cross-rank flag aggregation when dist is not initialized."""

    def test_local_true_propagates(self):
        """When dist is unavailable, local flag passes through."""
        result = all_reduce_non_finite_flag(True, tp_group=None, dp_group=None)
        self.assertTrue(result)

    def test_local_false_propagates(self):
        """When dist is unavailable, local False passes through."""
        result = all_reduce_non_finite_flag(False, tp_group=None, dp_group=None)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# emit_non_finite_report
# ---------------------------------------------------------------------------

class EmitNonFiniteReportTest(unittest.TestCase):
    """emit_non_finite_report: log + optional NonFiniteTrainingError."""

    def test_none_report_does_nothing(self):
        """A None report should be a no-op."""
        # Should not raise.
        emit_non_finite_report(None, skip_update=False, terminate=False)

    def test_report_without_terminate_logs_warning(self):
        """A report without terminate should not raise."""
        report = NonFiniteReport(
            step=1, loss_value=float("nan"), phase="actor",
            events=[NonFiniteEvent(name="loss", layer="loss", nan_count=1, total_elements=1)],
        )
        report.analyse()
        # Should not raise.
        emit_non_finite_report(report, skip_update=False, terminate=False)

    def test_report_with_terminate_raises(self):
        """When terminate=True, should raise NonFiniteTrainingError."""
        report = NonFiniteReport(
            step=1, loss_value=float("nan"), phase="actor",
            events=[NonFiniteEvent(name="loss", layer="loss", nan_count=1, total_elements=1)],
        )
        report.analyse()
        with self.assertRaises(NonFiniteTrainingError):
            emit_non_finite_report(report, skip_update=True, terminate=True)


# ---------------------------------------------------------------------------
# RuntimeConfig defaults
# ---------------------------------------------------------------------------

class RuntimeConfigNonFiniteDefaultsTest(unittest.TestCase):
    """New config fields default to False (preserve existing behavior)."""

    def test_skip_update_defaults_false(self):
        """non_finite_skip_update must default to False."""
        cfg = RuntimeConfig()
        self.assertFalse(cfg.non_finite_skip_update)

    def test_terminate_defaults_false(self):
        """non_finite_terminate must default to False."""
        cfg = RuntimeConfig()
        self.assertFalse(cfg.non_finite_terminate)

    def test_can_enable_flags(self):
        """Flags should be settable to True."""
        cfg = RuntimeConfig(non_finite_skip_update=True, non_finite_terminate=True)
        self.assertTrue(cfg.non_finite_skip_update)
        self.assertTrue(cfg.non_finite_terminate)


# ---------------------------------------------------------------------------
# Trainer-level _check_non_finite_values
# ---------------------------------------------------------------------------

class TrainerLevelNonFiniteTest(unittest.TestCase):
    """Trainer-layer detection of NaN/Inf in rewards and advantages."""

    def _make_trainer_stub(self, *, terminate=False):
        """Build a minimal trainer stub with _check_non_finite_values."""
        from areno.api.trainers.policy_only import PolicyOnlyTrainer

        trainer = object.__new__(PolicyOnlyTrainer)
        trainer.config = SimpleNamespace(
            non_finite_skip_update=False,
            non_finite_terminate=terminate,
        )
        trainer.logger = __import__("logging").getLogger("test")
        return trainer

    def test_normal_values_no_warning(self):
        """Finite rewards/advantages should not trigger any warning."""
        trainer = self._make_trainer_stub()
        # Should not raise.
        trainer._check_non_finite_values([1.0, 2.0, -0.5], stage="rewards", step=0)

    def test_nan_in_values_logs_warning(self):
        """NaN in values should log a warning (captured via logger)."""
        trainer = self._make_trainer_stub()
        with patch.object(trainer.logger, "warning") as mock_warn:
            trainer._check_non_finite_values([1.0, float("nan"), 3.0], stage="rewards", step=5)
            self.assertTrue(mock_warn.called)

    def test_terminate_raises_error(self):
        """When terminate=True, NaN should raise NonFiniteTrainingError."""
        trainer = self._make_trainer_stub(terminate=True)
        with self.assertRaises(NonFiniteTrainingError):
            trainer._check_non_finite_values(
                [1.0, float("inf"), 3.0], stage="advantages", step=10
            )

    def test_inf_in_values_detected(self):
        """Inf in values should trigger the warning path."""
        trainer = self._make_trainer_stub()
        with patch.object(trainer.logger, "warning") as mock_warn:
            trainer._check_non_finite_values(
                [0.0, float("inf"), float("-inf")], stage="rewards", step=2
            )
            self.assertTrue(mock_warn.called)


# ---------------------------------------------------------------------------
# CLI flag wiring
# ---------------------------------------------------------------------------

class CLINonFiniteFlagTest(unittest.TestCase):
    """Verify --non-finite-skip-update and --non-finite-terminate reach config."""

    def test_flags_in_train_option_groups(self):
        """New flag names should be in TRAIN_OPTION_GROUPS Train group."""
        from areno.cli.train import TRAIN_OPTION_GROUPS

        all_opts: list[str] = []
        for _label, opts in TRAIN_OPTION_GROUPS:
            all_opts.extend(opts)
        self.assertIn("non_finite_skip_update", all_opts)
        self.assertIn("non_finite_terminate", all_opts)

    def test_trainer_config_has_fields(self):
        """TrainerConfig should have the two new fields with False defaults."""
        from areno.api.trainer_config import TrainerConfig

        cfg = TrainerConfig(algo="gspo", ckpt="x", dataset_path="x")
        self.assertFalse(cfg.non_finite_skip_update)
        self.assertFalse(cfg.non_finite_terminate)


if __name__ == "__main__":
    unittest.main()
