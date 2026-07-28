"""CPU tests for stage-specific CUDA OOM diagnostics (issue #244).

These tests exercise the pure-Python OOM diagnostics module without any GPU
or CUDA dependency.  They cover:

* Stage detection from synthetic tracebacks for all three stages.
* Suggestion generation for model_loading, rollout, and training stages.
* Boundary / invalid inputs (empty config, missing keys, None values).
* Backward-compatible default (UNKNOWN stage produces empty guidance).
* Deterministic output for the same inputs.
* Structured (``OOMGuidance.to_dict``) and human-readable (``format_oom_guidance``) output.
* ``is_oom_error`` with real and fake exception types.
* ``diagnose_oom_from_exception`` end-to-end with a synthetic traceback.
"""

from __future__ import annotations

import unittest

from areno.engine.oom_diagnostics import (
    OOMGuidance,
    OOMStage,
    OOMSuggestion,
    build_oom_guidance,
    detect_stage,
    diagnose_oom_from_exception,
    format_oom_guidance,
    is_oom_error,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "tp_size": 4,
    "dp_size": 2,
    "world_size": 8,
    "batch_size": 32,
    "n_samples": 8,
    "mini_bs": 16,
    "max_new_tokens": 3071,
    "max_prompt_tokens": 1024,
    "max_running_prompts": 256,
    "attn_backend": "flash",
    "activation_checkpointing": True,
    "keep_rollout_state": True,
    "drop_rollout_state": False,
    "eager_decode": False,
    "adam_8bit": False,
    "compile_model": True,
    "dummy_load": False,
    "model_path": "Qwen/Qwen3-0.6B",
    "gradient_accumulation_steps": None,
}

_TRACEBACK_MODEL_LOADING = """\
Traceback (most recent call last):
  File "areno/engine/worker.py", line 60, in __init__
    self.model = build_model_on_device(config, self.device)
  File "areno/engine/modeling.py", line 30, in build_model_on_device
    model = build_model(config.model)
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB.
"""

_TRACEBACK_ROLLOUT = """\
Traceback (most recent call last):
  File "areno/engine/worker.py", line 132, in infer_rollout
    return self.inference.infer_rollout(payload)
  File "areno/engine/inference.py", line 500, in infer_rollout
    self._init_infer_cache(spec)
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB.
"""

_TRACEBACK_TRAINING = """\
Traceback (most recent call last):
  File "areno/engine/training.py", line 86, in _train_step
    loss_out = worker.loss_fn(data_pack, logprobs)
  File "areno/engine/training.py", line 94, in _train_step
    (loss / max(grad_scale, 1)).backward()
RuntimeError: CUDA out of memory. Tried to allocate 1.00 GiB.
"""

_TRACEBACK_AMBIGUOUS = """\
Traceback (most recent call last):
  File "<unknown>", line 1, in <module>
RuntimeError: CUDA out of memory. Tried to allocate 256.00 MiB.
"""

_TRACEBACK_NON_OOM = """\
Traceback (most recent call last):
  File "<unknown>", line 1, in <module>
ValueError: invalid argument
"""


# ---------------------------------------------------------------------------
# Stage detection
# ---------------------------------------------------------------------------


class TestStageDetection(unittest.TestCase):
    """detect_stage should correctly classify tracebacks."""

    def test_detect_model_loading(self):
        stage = detect_stage(_TRACEBACK_MODEL_LOADING)
        self.assertEqual(stage, OOMStage.MODEL_LOADING)

    def test_detect_rollout(self):
        stage = detect_stage(_TRACEBACK_ROLLOUT)
        self.assertEqual(stage, OOMStage.ROLLOUT)

    def test_detect_training(self):
        stage = detect_stage(_TRACEBACK_TRAINING)
        self.assertEqual(stage, OOMStage.TRAINING)

    def test_detect_unknown_for_ambiguous_oom(self):
        stage = detect_stage(_TRACEBACK_AMBIGUOUS)
        self.assertEqual(stage, OOMStage.UNKNOWN)

    def test_detect_unknown_for_non_oom(self):
        stage = detect_stage(_TRACEBACK_NON_OOM)
        self.assertEqual(stage, OOMStage.UNKNOWN)

    def test_detect_empty_string(self):
        self.assertEqual(detect_stage(""), OOMStage.UNKNOWN)

    def test_detect_case_insensitive(self):
        text = "BUILD_MODEL_ON_DEVICE raised CUDA out of memory"
        self.assertEqual(detect_stage(text), OOMStage.MODEL_LOADING)

    def test_detect_rollout_prefill_keyword(self):
        text = "prefill stage: CUDA out of memory"
        self.assertEqual(detect_stage(text), OOMStage.ROLLOUT)

    def test_detect_training_backward_keyword(self):
        text = "backward pass: CUDA out of memory"
        self.assertEqual(detect_stage(text), OOMStage.TRAINING)


# ---------------------------------------------------------------------------
# build_oom_guidance
# ---------------------------------------------------------------------------


class TestBuildOOMGuidance(unittest.TestCase):
    """build_oom_guidance should produce correct suggestions per stage."""

    def test_model_loading_suggestions_include_tp_size(self):
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertIn("--tp-size", options)
        tp_suggestion = next(s for s in guidance.suggestions if s.option == "--tp-size")
        self.assertEqual(tp_suggestion.current_value, 4)

    def test_model_loading_suggestions_include_attn_backend(self):
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertIn("--attn-backend", options)

    def test_model_loading_suggestions_include_adam_8bit(self):
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertIn("--adam-8bit", options)

    def test_model_loading_suggestions_omit_rollout_options(self):
        """Model loading guidance must not mention rollout-specific options."""
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--max-running-prompts", options)
        self.assertNotIn("--mini-bs", options)
        self.assertNotIn("--eager-decode", options)

    def test_rollout_suggestions_include_max_running_prompts(self):
        guidance = build_oom_guidance(OOMStage.ROLLOUT, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertIn("--max-running-prompts", options)
        mrp = next(s for s in guidance.suggestions if s.option == "--max-running-prompts")
        self.assertEqual(mrp.current_value, 256)

    def test_rollout_suggestions_include_eager_decode(self):
        guidance = build_oom_guidance(OOMStage.ROLLOUT, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertIn("--eager-decode", options)

    def test_rollout_suggestions_omit_training_options(self):
        """Rollout guidance must not mention training-specific options."""
        guidance = build_oom_guidance(OOMStage.ROLLOUT, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--mini-bs", options)
        self.assertNotIn("--adam-8bit", options)
        self.assertNotIn("--activation-checkpointing", options)

    def test_training_suggestions_include_mini_bs(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertIn("--mini-bs", options)
        mb = next(s for s in guidance.suggestions if s.option == "--mini-bs")
        self.assertEqual(mb.current_value, 16)

    def test_training_suggestions_include_drop_rollout_state(self):
        cfg = {**_BASE_CONFIG, "drop_rollout_state": False}
        guidance = build_oom_guidance(OOMStage.TRAINING, cfg)
        options = [s.option for s in guidance.suggestions]
        self.assertIn("--drop-rollout-state", options)

    def test_training_suggestions_omit_rollout_options(self):
        """Training guidance must not mention rollout-specific options."""
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--max-running-prompts", options)
        self.assertNotIn("--eager-decode", options)

    def test_suggestions_are_ordered_by_priority(self):
        for stage in (OOMStage.MODEL_LOADING, OOMStage.ROLLOUT, OOMStage.TRAINING):
            with self.subTest(stage=stage):
                guidance = build_oom_guidance(stage, _BASE_CONFIG)
                priorities = [s.priority for s in guidance.suggestions]
                self.assertEqual(priorities, sorted(priorities))

    def test_unknown_stage_produces_empty_suggestions(self):
        guidance = build_oom_guidance(OOMStage.UNKNOWN, _BASE_CONFIG)
        self.assertEqual(guidance.suggestions, [])

    def test_config_snapshot_filters_irrelevant_keys(self):
        """Only stage-relevant keys should appear in config_snapshot."""
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, _BASE_CONFIG)
        self.assertNotIn("max_running_prompts", guidance.config_snapshot)
        self.assertNotIn("mini_bs", guidance.config_snapshot)
        self.assertIn("tp_size", guidance.config_snapshot)

        guidance = build_oom_guidance(OOMStage.ROLLOUT, _BASE_CONFIG)
        self.assertNotIn("mini_bs", guidance.config_snapshot)
        self.assertIn("max_running_prompts", guidance.config_snapshot)

        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        self.assertNotIn("max_running_prompts", guidance.config_snapshot)
        self.assertIn("mini_bs", guidance.config_snapshot)


# ---------------------------------------------------------------------------
# Boundary / invalid inputs
# ---------------------------------------------------------------------------


class TestBoundaryInputs(unittest.TestCase):
    """The module must handle missing keys, empty config, and None values gracefully."""

    def test_empty_config_model_loading(self):
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, {})
        self.assertEqual(guidance.suggestions, [])

    def test_empty_config_rollout(self):
        guidance = build_oom_guidance(OOMStage.ROLLOUT, {})
        self.assertEqual(guidance.suggestions, [])

    def test_empty_config_training(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, {})
        self.assertEqual(guidance.suggestions, [])

    def test_missing_keys_produce_fewer_suggestions(self):
        cfg = {"tp_size": 4}
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, cfg)
        options = [s.option for s in guidance.suggestions]
        self.assertIn("--tp-size", options)
        # Should not crash; just fewer suggestions.
        self.assertTrue(len(guidance.suggestions) >= 1)

    def test_none_values_are_skipped(self):
        cfg = {"tp_size": None, "attn_backend": "flash"}
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, cfg)
        # tp_size is None so the suggestion should be skipped.
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

    def test_activation_checkpointing_disabled(self):
        cfg = {**_BASE_CONFIG, "activation_checkpointing": False}
        guidance = build_oom_guidance(OOMStage.TRAINING, cfg)
        options = [s.option for s in guidance.suggestions]
        self.assertIn("--activation-checkpointing", options)

    def test_adam_8bit_already_enabled(self):
        cfg = {**_BASE_CONFIG, "adam_8bit": True}
        guidance = build_oom_guidance(OOMStage.TRAINING, cfg)
        options = [s.option for s in guidance.suggestions]
        self.assertNotIn("--adam-8bit", options)


# ---------------------------------------------------------------------------
# format_oom_guidance (human-readable output)
# ---------------------------------------------------------------------------


class TestFormatOOMGuidance(unittest.TestCase):
    """format_oom_guidance should produce readable, informative output."""

    def test_model_loading_output_contains_stage_label(self):
        text = format_oom_guidance(OOMStage.MODEL_LOADING, _BASE_CONFIG)
        self.assertIn("model loading", text)
        self.assertIn("CUDA OOM", text)

    def test_rollout_output_contains_stage_label(self):
        text = format_oom_guidance(OOMStage.ROLLOUT, _BASE_CONFIG)
        self.assertIn("rollout generation", text)

    def test_training_output_contains_stage_label(self):
        text = format_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        self.assertIn("training", text)

    def test_output_contains_troubleshooting_url(self):
        for stage in (OOMStage.MODEL_LOADING, OOMStage.ROLLOUT, OOMStage.TRAINING):
            with self.subTest(stage=stage):
                text = format_oom_guidance(stage, _BASE_CONFIG)
                self.assertIn("troubleshooting", text.lower())

    def test_output_contains_current_values(self):
        text = format_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        self.assertIn("16", text)  # mini_bs value
        self.assertIn("--mini-bs", text)

    def test_unknown_stage_returns_empty_string(self):
        """Backward-compatible default: UNKNOWN stage produces no output."""
        self.assertEqual(format_oom_guidance(OOMStage.UNKNOWN, _BASE_CONFIG), "")

    def test_empty_config_returns_empty_string(self):
        """No suggestions means empty output, even for a known stage."""
        self.assertEqual(format_oom_guidance(OOMStage.MODEL_LOADING, {}), "")

    def test_output_is_deterministic(self):
        text1 = format_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        text2 = format_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        self.assertEqual(text1, text2)

    def test_output_contains_numbered_suggestions(self):
        text = format_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        self.assertIn("1.", text)
        self.assertIn("2.", text)


# ---------------------------------------------------------------------------
# OOMGuidance.to_dict (structured output)
# ---------------------------------------------------------------------------


class TestOOMGuidanceToDict(unittest.TestCase):
    """to_dict should produce a JSON-serialisable structure with all fields."""

    def test_to_dict_has_required_fields(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        d = guidance.to_dict()
        self.assertIn("stage", d)
        self.assertIn("suggestions", d)
        self.assertIn("config_snapshot", d)
        self.assertIn("troubleshooting_url", d)

    def test_to_dict_stage_is_string(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        d = guidance.to_dict()
        self.assertEqual(d["stage"], "training")

    def test_to_dict_suggestions_have_fields(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        d = guidance.to_dict()
        self.assertTrue(len(d["suggestions"]) > 0)
        for s in d["suggestions"]:
            self.assertIn("option", s)
            self.assertIn("current_value", s)
            self.assertIn("recommended_action", s)
            self.assertIn("priority", s)


# ---------------------------------------------------------------------------
# is_oom_error
# ---------------------------------------------------------------------------


class TestIsOOMError(unittest.TestCase):
    """is_oom_error should detect OOM from exception types and messages."""

    def test_runtime_error_with_oom_message(self):
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB.")
        self.assertTrue(is_oom_error(exc))

    def test_runtime_error_without_oom_message(self):
        exc = RuntimeError("invalid argument")
        self.assertFalse(is_oom_error(exc))

    def test_value_error_is_not_oom(self):
        exc = ValueError("something else")
        self.assertFalse(is_oom_error(exc))

    def test_none_input(self):
        # A non-exception with no "out of memory" in str(None).
        self.assertFalse(is_oom_error(None))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# diagnose_oom_from_exception (end-to-end)
# ---------------------------------------------------------------------------


class TestDiagnoseFromException(unittest.TestCase):
    """diagnose_oom_from_exception should detect stage and produce guidance."""

    def test_diagnose_training_oom(self):
        try:
            raise RuntimeError("CUDA out of memory in _train_step during backward")
        except RuntimeError as exc:
            text = diagnose_oom_from_exception(exc, _BASE_CONFIG)
            self.assertIn("training", text)
            self.assertIn("--mini-bs", text)

    def test_diagnose_rollout_oom(self):
        try:
            raise RuntimeError("CUDA out of memory in infer_rollout during prefill")
        except RuntimeError as exc:
            text = diagnose_oom_from_exception(exc, _BASE_CONFIG)
            self.assertIn("rollout", text)
            self.assertIn("--max-running-prompts", text)

    def test_diagnose_model_loading_oom(self):
        try:
            raise RuntimeError("CUDA out of memory in build_model_on_device")
        except RuntimeError as exc:
            text = diagnose_oom_from_exception(exc, _BASE_CONFIG)
            self.assertIn("model loading", text)
            self.assertIn("--tp-size", text)

    def test_diagnose_unknown_oom_returns_empty(self):
        try:
            raise RuntimeError("CUDA out of memory. Something went wrong.")
        except RuntimeError as exc:
            text = diagnose_oom_from_exception(exc, _BASE_CONFIG)
            self.assertEqual(text, "")

    def test_diagnose_non_oom_returns_empty(self):
        try:
            raise ValueError("not an OOM")
        except ValueError as exc:
            text = diagnose_oom_from_exception(exc, _BASE_CONFIG)
            self.assertEqual(text, "")


# ---------------------------------------------------------------------------
# Omit irrelevant advice (acceptance criterion)
# ---------------------------------------------------------------------------


class TestOmitIrrelevantAdvice(unittest.TestCase):
    """Each stage must omit advice for the other two stages."""

    def test_model_loading_omits_rollout_and_train_advice(self):
        guidance = build_oom_guidance(OOMStage.MODEL_LOADING, _BASE_CONFIG)
        all_text = " ".join(s.recommended_action for s in guidance.suggestions)
        self.assertNotIn("max-running-prompts", all_text.lower())
        self.assertNotIn("mini-bs", all_text.lower())
        self.assertNotIn("eager-decode", all_text.lower())

    def test_rollout_omits_model_and_train_advice(self):
        guidance = build_oom_guidance(OOMStage.ROLLOUT, _BASE_CONFIG)
        all_text = " ".join(s.recommended_action for s in guidance.suggestions)
        # Should not mention adam-8bit (model/train) or mini-bs (train).
        self.assertNotIn("adam", all_text.lower())
        self.assertNotIn("mini-bs", all_text.lower())
        self.assertNotIn("activation-checkpointing", all_text.lower())

    def test_training_omits_rollout_advice(self):
        guidance = build_oom_guidance(OOMStage.TRAINING, _BASE_CONFIG)
        all_text = " ".join(s.recommended_action for s in guidance.suggestions)
        self.assertNotIn("max-running-prompts", all_text.lower())
        self.assertNotIn("eager-decode", all_text.lower())


if __name__ == "__main__":
    unittest.main()