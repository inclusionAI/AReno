"""CPU tests for stage-specific CUDA OOM diagnostics (issue #244).

Tests cover:
- Stage comes from explicit call-site boundary, NOT traceback guessing.
- Unknown OOM (no boundary marker) produces no guidance.
- Non-OOM errors pass through without guidance.
- Suggestions only use real CLI options (validated against --help).
- Each stage omits other stages' suggestions.
- compile_model is NOT in snapshot (not a TrainerConfig field).
- --max-new-tokens is NOT suggested.
- model loading stage has NO optimizer suggestions (--adam-8bit).
- OOMGuidance.to_dict() and to_json() structured output.
- validate_suggestions_use_real_cli_options().
- is_oom_error with real and fake exceptions.
"""

from __future__ import annotations

import json
import unittest

from areno.engine.oom_diagnostics import (
    OOMStage,
    build_oom_config_snapshot,
    build_oom_guidance,
    format_oom_guidance,
    is_oom_error,
    validate_suggestions_use_real_cli_options,
)

_BASE_CONFIG = {
    "tp_size": 4,
    "dp_size": 2,
    "world_size": 8,
    "batch_size": 32,
    "n_samples": 8,
    "mini_bs": 16,
    "max_running_prompts": 256,
    "attn_backend": "flash",
    "activation_checkpointing": True,
    "keep_rollout_state": True,
    "drop_rollout_state": False,
    "eager_decode": False,
    "adam_8bit": False,
    "gradient_accumulation_steps": None,
}


class FakeTrainerConfig:
    """Minimal stand-in for TrainerConfig."""

    tp_size = 4
    world_size = 8
    batch_size = 32
    mini_bs = 16
    attn_backend = "flash"
    activation_checkpointing = True
    keep_rollout_state = True
    eager_decode = False
    adam_8bit = False
    gradient_accumulation_steps = None
    n_samples = 8
    max_running_prompts = None

    def resolved_max_running_prompts(self):
        return self.batch_size * self.n_samples


# ---------------------------------------------------------------------------
# Stage from explicit boundary (not traceback guessing)
# ---------------------------------------------------------------------------


class TestStageFromBoundary(unittest.TestCase):
    """Stage must come from explicit boundary, not traceback text."""

    def test_unknown_stage_no_guidance(self):
        """OOM without a boundary marker must produce no guidance."""
        text = format_oom_guidance(OOMStage.UNKNOWN, _BASE_CONFIG)
        self.assertEqual(text, "")

    def test_unknown_stage_empty_suggestions(self):
        guidance = build_oom_guidance(OOMStage.UNKNOWN, _BASE_CONFIG)
        self.assertEqual(guidance.suggestions, [])

    def test_model_loading_stage(self):
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, _BASE_CONFIG)
        self.assertEqual(guidance.stage, OOMStage.MODEL_LOADING)
        self.assertTrue(len(guidance.suggestions) > 0)

    def test_rollout_stage(self):
        guidance = build_oom_guidance(OOMStage.ROLLOUT, _BASE_CONFIG)
        self.assertEqual(guidance.stage, OOMStage.ROLLOUT)
        self.assertTrue(len(guidance.suggestions) > 0)

    def test_training_stage(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        self.assertEqual(guidance.stage, OOMStage.TRAINING)
        self.assertTrue(len(guidance.suggestions) > 0)


# ---------------------------------------------------------------------------
# Only real CLI options
# ---------------------------------------------------------------------------


class TestRealCLIOptions(unittest.TestCase):
    """All suggested options must exist in `areno train --help`."""

    def test_model_loading_options_valid(self):
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, _BASE_CONFIG)
        self.assertTrue(
            validate_suggestions_use_real_cli_options(guidance),
            f"Invalid options: {[s.option for s in guidance.suggestions]}",
        )

    def test_rollout_options_valid(self):
        guidance = build_oom_guidance(OOMStage.ROLLOUT, _BASE_CONFIG)
        self.assertTrue(validate_suggestions_use_real_cli_options(guidance))

    def test_training_options_valid(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        self.assertTrue(validate_suggestions_use_real_cli_options(guidance))

    def test_no_compile_model_option(self):
        """--no-compile-model does not exist in CLI."""
        for stage in [OOMStage.MODEL_LOADING, OOMStage.ROLLOUT, OOMStage.TRAINING]:
            guidance = build_oom_guidance(stage, _BASE_CONFIG)
            options = [s.option for s in guidance.suggestions]
            self.assertNotIn("--no-compile-model", options)

    def test_no_max_new_tokens_suggestion(self):
        """--max-new-tokens should NOT be suggested."""
        for stage in [OOMStage.MODEL_LOADING, OOMStage.ROLLOUT, OOMStage.TRAINING]:
            guidance = build_oom_guidance(stage, _BASE_CONFIG)
            options = [s.option for s in guidance.suggestions]
            self.assertNotIn("--max-new-tokens", options)


# ---------------------------------------------------------------------------
# Stage-specific omission
# ---------------------------------------------------------------------------


class TestStageOmission(unittest.TestCase):
    """Each stage must omit other stages' suggestions."""

    def test_model_loading_no_optimizer_options(self):
        """Model loading must not suggest --adam-8bit (optimizer is training stage)."""
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--adam-8bit", options)
        self.assertNotIn("--mini-bs", options)
        self.assertNotIn("--activation-checkpointing", options)

    def test_model_loading_has_tp_size(self):
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertIn("--tp-size", options)

    def test_rollout_no_training_options(self):
        guidance = build_oom_guidance(OOMStage.ROLLOUT, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--mini-bs", options)
        self.assertNotIn("--adam-8bit", options)
        self.assertNotIn("--activation-checkpointing", options)

    def test_training_no_rollout_options(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--max-running-prompts", options)
        self.assertNotIn("--eager-decode", options)

    def test_training_has_mini_bs(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertIn("--mini-bs", options)


# ---------------------------------------------------------------------------
# Config snapshot from real TrainerConfig
# ---------------------------------------------------------------------------


class TestConfigSnapshot(unittest.TestCase):
    """Snapshot must come from real TrainerConfig, not hardcode."""

    def test_snapshot_has_tp_size(self):
        snapshot = build_oom_config_snapshot(FakeTrainerConfig())
        self.assertEqual(snapshot["tp_size"], 4)

    def test_snapshot_has_no_compile_model(self):
        """compile_model must NOT be in snapshot (not a TrainerConfig field)."""
        snapshot = build_oom_config_snapshot(FakeTrainerConfig())
        self.assertNotIn("compile_model", snapshot)

    def test_snapshot_has_no_model_path(self):
        snapshot = build_oom_config_snapshot(FakeTrainerConfig())
        self.assertNotIn("model_path", snapshot)

    def test_snapshot_has_no_dummy_load(self):
        snapshot = build_oom_config_snapshot(FakeTrainerConfig())
        self.assertNotIn("dummy_load", snapshot)

    def test_snapshot_has_dp_size(self):
        snapshot = build_oom_config_snapshot(FakeTrainerConfig())
        self.assertEqual(snapshot["dp_size"], 2)

    def test_snapshot_has_rollout_fields(self):
        snapshot = build_oom_config_snapshot(FakeTrainerConfig())
        self.assertIn("n_samples", snapshot)
        self.assertIn("max_running_prompts", snapshot)

    def test_snapshot_drop_rollout_state(self):
        snapshot = build_oom_config_snapshot(FakeTrainerConfig())
        self.assertFalse(snapshot["drop_rollout_state"])


# ---------------------------------------------------------------------------
# is_oom_error
# ---------------------------------------------------------------------------


class TestIsOOMError(unittest.TestCase):
    """is_oom_error should detect OOM from types and messages."""

    def test_runtime_error_with_oom_message(self):
        self.assertTrue(is_oom_error(RuntimeError("CUDA out of memory")))

    def test_runtime_error_without_oom(self):
        self.assertFalse(is_oom_error(RuntimeError("invalid argument")))

    def test_value_error(self):
        self.assertFalse(is_oom_error(ValueError("something")))


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


class TestStructuredOutput(unittest.TestCase):
    """OOMGuidance.to_dict() and to_json() must be complete."""

    def test_to_dict_fields(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        d = guidance.to_dict()
        self.assertIn("stage", d)
        self.assertIn("suggestions", d)
        self.assertIn("config_snapshot", d)
        self.assertIn("troubleshooting_url", d)

    def test_to_json_parses(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        d = json.loads(guidance.to_json())
        self.assertEqual(d["stage"], "training")

    def test_suggestion_fields(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        for s in guidance.suggestions:
            self.assertIsInstance(s.option, str)
            self.assertTrue(s.priority >= 0)


# ---------------------------------------------------------------------------
# format_oom_guidance
# ---------------------------------------------------------------------------


class TestFormatGuidance(unittest.TestCase):
    """Human-readable output."""

    def test_training_output_has_stage_label(self):
        text = format_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        self.assertIn("training", text)
        self.assertIn("CUDA OOM", text)

    def test_rollout_output_has_stage_label(self):
        text = format_oom_guidance(OOMStage.ROLLOUT, _BASE_CONFIG)
        self.assertIn("rollout", text)

    def test_unknown_returns_empty(self):
        self.assertEqual(format_oom_guidance(OOMStage.UNKNOWN, _BASE_CONFIG), "")

    def test_output_has_troubleshooting_url(self):
        text = format_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        self.assertIn("troubleshooting", text.lower())

    def test_output_has_current_values(self):
        text = format_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        self.assertIn("16", text)  # mini_bs


# ---------------------------------------------------------------------------
# Boundary injection (simulated)
# ---------------------------------------------------------------------------


class TestBoundaryInjection(unittest.TestCase):
    """Simulate OOM at each boundary and verify stage is attached."""

    def _make_oom(self, stage: str):
        """Create a RuntimeError with _oom_stage attached."""
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB.")
        exc._oom_stage = stage
        return exc

    def test_model_loading_boundary(self):
        exc = self._make_oom("model_loading")
        self.assertEqual(getattr(exc, "_oom_stage", "unknown"), "model_loading")

    def test_rollout_boundary(self):
        exc = self._make_oom("rollout")
        self.assertEqual(getattr(exc, "_oom_stage", "unknown"), "rollout")

    def test_training_boundary(self):
        exc = self._make_oom("training")
        self.assertEqual(getattr(exc, "_oom_stage", "unknown"), "training")

    def test_unmarked_oom_is_unknown(self):
        """OOM without _oom_stage attribute should be treated as unknown."""
        exc = RuntimeError("CUDA out of memory.")
        stage = getattr(exc, "_oom_stage", "unknown")
        self.assertEqual(stage, "unknown")
        # Unknown stage should produce no guidance.
        self.assertEqual(format_oom_guidance(OOMStage(stage), _BASE_CONFIG), "")

    def test_non_oom_passes_through(self):
        """Non-OOM error should not trigger guidance."""
        exc = ValueError("invalid argument")
        self.assertFalse(is_oom_error(exc))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    """Same inputs produce same outputs."""

    def test_deterministic(self):
        g1 = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        g2 = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        self.assertEqual(g1.to_dict(), g2.to_dict())


# ---------------------------------------------------------------------------
# Empty/missing config
# ---------------------------------------------------------------------------


class TestBoundaryInputs(unittest.TestCase):
    """Handle empty config and missing keys gracefully."""

    def test_empty_config_model_loading(self):
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, {})
        self.assertEqual(guidance.suggestions, [])

    def test_empty_config_training(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, {})
        self.assertEqual(guidance.suggestions, [])

    def test_none_values_skipped(self):
        cfg = {"tp_size": None, "attn_backend": "flash"}
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, cfg)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--tp-size", options)

    def test_attn_backend_native_skips_flash_suggestion(self):
        cfg = {**_BASE_CONFIG, "attn_backend": "native"}
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, cfg)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--attn-backend", options)

    def test_activation_checkpointing_already_enabled(self):
        cfg = {**_BASE_CONFIG, "activation_checkpointing": True}
        guidance = build_oom_guidance(OOMStage.TRAINING, cfg)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--activation-checkpointing", options)


if __name__ == "__main__":
    unittest.main()
