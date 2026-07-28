"""Integration-style tests for OOM diagnostics (issue #244).

These tests verify the OOM diagnostics module's integration with the training
pipeline without requiring a GPU, torch, or real model loading.  We use fakes
and stubs to isolate GPU-only behaviour.

The ``areno.cli.train`` module imports torch transitively, so tests that need
to exercise CLI integration functions are guarded by a torch availability
check.  The pure-Python OOM diagnostics module itself has no torch dependency
and is always tested.

Covers the acceptance criteria from issue #244:

* Test the three stages with synthetic errors.
* Omit irrelevant advice per stage.
* Do not mutate configuration or retry automatically.
* Verify existing behaviour is unchanged when the feature is not enabled
  (non-OOM errors pass through without guidance).
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr

from areno.engine.oom_diagnostics import (
    OOMStage,
    build_oom_guidance,
    detect_stage,
    diagnose_oom_from_exception,
    format_oom_guidance,
    is_oom_error,
)

# Check if torch is available (it won't be in CPU-only CI without torch installed).
try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Fake trainer config that mirrors TrainerConfig's interface
# ---------------------------------------------------------------------------


class FakeTrainerConfig:
    """Minimal stand-in for TrainerConfig used by _build_oom_config_snapshot."""

    tp_size = 4
    world_size = 8
    batch_size = 32
    mini_bs = 16
    max_new_tokens = 3071
    max_prompt_tokens = 1024
    attn_backend = "flash"
    activation_checkpointing = True
    keep_rollout_state = True
    eager_decode = False
    adam_8bit = False
    ckpt = "Qwen/Qwen3-0.6B"
    gradient_accumulation_steps = None
    n_samples = 8
    max_running_prompts = None

    def resolved_max_running_prompts(self):
        return self.batch_size * self.n_samples


def _build_snapshot_from_fake(config: FakeTrainerConfig) -> dict:
    """Standalone version of _build_oom_config_snapshot for torch-free testing."""

    snapshot = {
        "tp_size": config.tp_size,
        "world_size": config.world_size,
        "dp_size": config.world_size // config.tp_size if config.tp_size else None,
        "batch_size": config.batch_size,
        "mini_bs": config.mini_bs,
        "max_new_tokens": config.max_new_tokens,
        "max_prompt_tokens": config.max_prompt_tokens,
        "attn_backend": config.attn_backend,
        "activation_checkpointing": config.activation_checkpointing,
        "keep_rollout_state": config.keep_rollout_state,
        "drop_rollout_state": not config.keep_rollout_state,
        "eager_decode": config.eager_decode,
        "adam_8bit": config.adam_8bit,
        "model_path": config.ckpt,
        "compile_model": True,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "dummy_load": False,
        "n_samples": config.n_samples,
        "max_running_prompts": config.resolved_max_running_prompts(),
    }
    return snapshot


# Synthetic tracebacks mimicking what the worker sends back to the coordinator.

OOM_MODEL_LOADING = (
    "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB.\n"
    "  File \"areno/engine/modeling.py\", line 30, in build_model_on_device\n"
    "    model = build_model(config.model)"
)

OOM_ROLLOUT = (
    "RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB.\n"
    "  File \"areno/engine/inference.py\", line 500, in infer_rollout\n"
    "    self._init_infer_cache(spec)"
)

OOM_TRAINING = (
    "RuntimeError: CUDA out of memory. Tried to allocate 1.00 GiB.\n"
    "  File \"areno/engine/training.py\", line 94, in _train_step\n"
    "    (loss / max(grad_scale, 1)).backward()"
)

NON_OOM_ERROR = (
    "ValueError: invalid argument: tp_size must divide num_attention_heads"
)


# ---------------------------------------------------------------------------
# Integration: config snapshot + format_oom_guidance
# ---------------------------------------------------------------------------


class TestOOMConfigSnapshotIntegration(unittest.TestCase):
    """The snapshot dict should produce correct guidance for each stage."""

    def test_snapshot_produces_model_loading_guidance(self):
        snapshot = _build_snapshot_from_fake(FakeTrainerConfig())
        text = format_oom_guidance(OOMStage.MODEL_LOADING, snapshot)
        self.assertIn("model loading", text)
        self.assertIn("--tp-size", text)

    def test_snapshot_produces_rollout_guidance(self):
        snapshot = _build_snapshot_from_fake(FakeTrainerConfig())
        text = format_oom_guidance(OOMStage.ROLLOUT, snapshot)
        self.assertIn("rollout", text)
        self.assertIn("--max-running-prompts", text)

    def test_snapshot_produces_training_guidance(self):
        snapshot = _build_snapshot_from_fake(FakeTrainerConfig())
        text = format_oom_guidance(OOMStage.TRAINING, snapshot)
        self.assertIn("training", text)
        self.assertIn("--mini-bs", text)


# ---------------------------------------------------------------------------
# Integration: diagnose_oom_from_exception with synthetic tracebacks
# ---------------------------------------------------------------------------


class TestDiagnoseWithSyntheticTracebacks(unittest.TestCase):
    """diagnose_oom_from_exception should detect stage from realistic tracebacks."""

    def setUp(self):
        self.snapshot = _build_snapshot_from_fake(FakeTrainerConfig())

    def test_model_loading_traceback(self):
        try:
            raise RuntimeError(OOM_MODEL_LOADING)
        except RuntimeError as exc:
            text = diagnose_oom_from_exception(exc, self.snapshot)
            self.assertIn("model loading", text)
            self.assertIn("--tp-size", text)
            self.assertIn("troubleshooting", text.lower())

    def test_rollout_traceback(self):
        try:
            raise RuntimeError(OOM_ROLLOUT)
        except RuntimeError as exc:
            text = diagnose_oom_from_exception(exc, self.snapshot)
            self.assertIn("rollout", text)
            self.assertIn("--max-running-prompts", text)

    def test_training_traceback(self):
        try:
            raise RuntimeError(OOM_TRAINING)
        except RuntimeError as exc:
            text = diagnose_oom_from_exception(exc, self.snapshot)
            self.assertIn("training", text)
            self.assertIn("--mini-bs", text)

    def test_non_oom_error_produces_no_guidance(self):
        """Non-OOM errors must not produce guidance (backward compatible)."""
        try:
            raise ValueError(NON_OOM_ERROR)
        except ValueError as exc:
            text = diagnose_oom_from_exception(exc, self.snapshot)
            self.assertEqual(text, "")

    def test_unknown_stage_oom_produces_no_guidance(self):
        try:
            raise RuntimeError("CUDA out of memory. Something mysterious happened.")
        except RuntimeError as exc:
            text = diagnose_oom_from_exception(exc, self.snapshot)
            self.assertEqual(text, "")


# ---------------------------------------------------------------------------
# Integration: stage-specific omission of irrelevant advice
# ---------------------------------------------------------------------------


class TestStageSpecificOmission(unittest.TestCase):
    """Each stage must omit advice for the other two stages (acceptance criterion)."""

    def setUp(self):
        self.snapshot = _build_snapshot_from_fake(FakeTrainerConfig())

    def test_model_loading_omits_rollout_and_train_advice(self):
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, self.snapshot)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--max-running-prompts", options)
        self.assertNotIn("--mini-bs", options)
        self.assertNotIn("--eager-decode", options)

    def test_rollout_omits_model_and_train_advice(self):
        guidance = build_oom_guidance(OOMStage.ROLLOUT, self.snapshot)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--adam-8bit", options)
        self.assertNotIn("--mini-bs", options)
        self.assertNotIn("--activation-checkpointing", options)

    def test_training_omits_rollout_advice(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, self.snapshot)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--max-running-prompts", options)
        self.assertNotIn("--eager-decode", options)


# ---------------------------------------------------------------------------
# Integration: does not mutate configuration or retry
# ---------------------------------------------------------------------------


class TestNoMutationOrRetry(unittest.TestCase):
    """The OOM diagnostics must not mutate config or cause automatic retries."""

    def test_config_snapshot_is_not_mutated(self):
        config = FakeTrainerConfig()
        original_tp = config.tp_size
        snapshot = _build_snapshot_from_fake(config)
        self.assertEqual(config.tp_size, original_tp)
        self.assertIsInstance(snapshot, dict)

    def test_format_oom_guidance_does_not_modify_config(self):
        snapshot = {
            "tp_size": 4,
            "attn_backend": "flash",
            "adam_8bit": False,
            "dp_size": 2,
            "world_size": 8,
            "compile_model": True,
        }
        original = dict(snapshot)
        format_oom_guidance(OOMStage.MODEL_LOADING, snapshot)
        self.assertEqual(snapshot, original)

    def test_build_oom_guidance_does_not_modify_config(self):
        snapshot = _build_snapshot_from_fake(FakeTrainerConfig())
        original = dict(snapshot)
        build_oom_guidance(OOMStage.TRAINING, snapshot)
        self.assertEqual(snapshot, original)


# ---------------------------------------------------------------------------
# Integration: backward compatibility (feature not enabled / default)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility(unittest.TestCase):
    """When OOM diagnostics cannot identify a stage, output must be empty."""

    def setUp(self):
        self.snapshot = _build_snapshot_from_fake(FakeTrainerConfig())

    def test_unknown_stage_returns_empty(self):
        self.assertEqual(format_oom_guidance(OOMStage.UNKNOWN, self.snapshot), "")

    def test_non_oom_exception_returns_empty(self):
        try:
            raise RuntimeError("some other error")
        except RuntimeError as exc:
            text = diagnose_oom_from_exception(exc, self.snapshot)
            self.assertEqual(text, "")

    def test_is_oom_error_returns_false_for_non_oom(self):
        self.assertFalse(is_oom_error(RuntimeError("not oom")))
        self.assertFalse(is_oom_error(ValueError("not oom")))

    def test_is_oom_error_returns_true_for_oom(self):
        self.assertTrue(is_oom_error(RuntimeError("CUDA out of memory")))
        self.assertTrue(is_oom_error(RuntimeError("cuda error: out of memory")))


# ---------------------------------------------------------------------------
# Integration: CLI functions (only if torch is available)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_TORCH, "torch not available; CLI integration requires torch")
class TestCLIOOMGuidance(unittest.TestCase):
    """Test CLI-level OOM guidance functions when torch is importable."""

    def test_cli_snapshot_matches_standalone(self):
        from areno.cli.train import _build_oom_config_snapshot

        cli_snapshot = _build_oom_config_snapshot(FakeTrainerConfig())
        standalone = _build_snapshot_from_fake(FakeTrainerConfig())
        # Both should have the same keys and values.
        for key in standalone:
            self.assertEqual(cli_snapshot[key], standalone[key])

    def test_cli_print_oom_guidance_for_training(self):
        from areno.cli.train import _print_oom_guidance

        exc = RuntimeError(OOM_TRAINING)
        buf = io.StringIO()
        with redirect_stderr(buf):
            _print_oom_guidance(exc, FakeTrainerConfig())
        output = buf.getvalue()
        self.assertIn("training", output)
        self.assertIn("--mini-bs", output)

    def test_cli_no_output_for_non_oom(self):
        from areno.cli.train import _print_oom_guidance

        exc = ValueError(NON_OOM_ERROR)
        buf = io.StringIO()
        with redirect_stderr(buf):
            _print_oom_guidance(exc, FakeTrainerConfig())
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()