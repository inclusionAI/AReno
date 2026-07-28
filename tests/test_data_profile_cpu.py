from __future__ import annotations

import json
import unittest

import areno.api.trainer as trainer_mod
from areno.api.data_profile import DataProfileReport, StageProfiler, StageStats
from areno.api.trainer import Trainer
from tests.helpers import PatchedContext


def _encode_from_record_prompt(_tokenizer, prompt: str) -> list[int]:
    """Deterministic tokenizer stub matching the one in test_trainer_api_cpu."""

    tokens_by_prompt = {
        "short": [1, 2],
        "long": [1, 2, 3, 4, 5],
        "next": [3],
        "a": [10],
        "b": [11],
        "c": [12],
    }
    return tokens_by_prompt[prompt]


class StageProfilerTest(unittest.TestCase):
    """Unit tests for the StageProfiler / DataProfileReport primitives."""

    def test_disabled_profiler_is_noop(self):
        """When enabled=False, stage() should not accumulate any stats."""
        profiler = StageProfiler(enabled=False)
        with profiler.stage("tokenize", index=0):
            pass
        self.assertEqual(profiler.stages, {})

    def test_enabled_profiler_records_calls_and_time(self):
        profiler = StageProfiler(enabled=True)
        with profiler.stage("tokenize", index=0):
            pass
        self.assertIn("tokenize", profiler.stages)
        self.assertEqual(profiler.stages["tokenize"].calls, 1)
        self.assertGreaterEqual(profiler.stages["tokenize"].total_seconds, 0.0)

    def test_inject_delay_attribution(self):
        """Injecting delay into one stage must only affect that stage."""
        profiler = StageProfiler(enabled=True)
        with profiler.stage("tokenize", inject_delay_s=0.05):
            pass
        with profiler.stage("filter"):
            pass
        # Only the injected stage should show measurable time.
        self.assertGreaterEqual(profiler.stages["tokenize"].total_seconds, 0.045)
        self.assertLess(profiler.stages["filter"].total_seconds, 0.045)

    def test_slow_record_stores_bounded_identifiers_only(self):
        profiler = StageProfiler(enabled=True, slow_threshold_s=0.0)
        with profiler.stage("tokenize", index=42, tokens=7):
            pass
        slow = profiler.stages["tokenize"].slow_records
        self.assertEqual(len(slow), 1)
        self.assertEqual(slow[0], {"index": 42, "tokens": 7})
        # No prompt text or training content leaked.
        self.assertNotIn("prompt", slow[0])
        self.assertNotIn("text", slow[0])

    def test_report_to_dict_is_deterministic(self):
        profiler = StageProfiler(enabled=True)
        with profiler.stage("tokenize", index=0):
            pass
        report = profiler.build_report(records_scanned=3, records_skipped_long=1, wall_seconds=0.1)
        d1 = report.to_dict()
        d2 = report.to_dict()
        self.assertEqual(d1, d2)
        self.assertEqual(d1["records_scanned"], 3)
        self.assertEqual(d1["records_skipped_long"], 1)

    def test_report_render_human_contains_stage_names(self):
        profiler = StageProfiler(enabled=True)
        with profiler.stage("record_access"):
            pass
        with profiler.stage("tokenize"):
            pass
        report = profiler.build_report(records_scanned=1, records_skipped_long=0, wall_seconds=0.01)
        text = report.render_human()
        self.assertIn("record_access", text)
        self.assertIn("tokenize", text)
        self.assertIn("records_scanned=1", text)


class LoadPromptBatchesProfiledTest(unittest.TestCase):
    """Integration tests for Trainer.load_prompt_batches_profiled."""

    def _make_trainer(self):
        trainer = Trainer(world_size=1, model_path="unused")
        trainer._tokenizer = object()
        return trainer

    def test_profiled_yields_batches_with_reports(self):
        trainer = self._make_trainer()
        dataset = [{"prompt": "a"}, {"prompt": "b"}]
        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_record_prompt):
            results = list(trainer.load_prompt_batches_profiled(dataset, batch_size=2, max_prompt_tokens=4))
        self.assertEqual(len(results), 1)
        batch, report = results[0]
        self.assertEqual(batch.prompts, ["a", "b"])
        self.assertIsInstance(report, DataProfileReport)
        self.assertEqual(report.records_scanned, 2)
        self.assertEqual(report.records_skipped_long, 0)

    def test_profiled_stages_are_present(self):
        trainer = self._make_trainer()
        dataset = [{"prompt": "a"}, {"prompt": "b"}]
        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_record_prompt):
            results = list(trainer.load_prompt_batches_profiled(dataset, batch_size=2, max_prompt_tokens=4))
        _, report = results[0]
        for stage_name in ("record_access", "contract_conversion", "tokenize", "filter", "batch"):
            self.assertIn(stage_name, report.stages, f"stage '{stage_name}' missing from report")

    def test_profiled_skips_long_prompts(self):
        trainer = self._make_trainer()
        dataset = [{"prompt": "long"}, {"prompt": "short"}]
        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_record_prompt):
            results = list(trainer.load_prompt_batches_profiled(dataset, batch_size=1, max_prompt_tokens=3))
        self.assertEqual(len(results), 1)
        batch, report = results[0]
        self.assertEqual(batch.prompts, ["short"])
        self.assertEqual(report.records_skipped_long, 1)

    def test_profiled_preserves_value_error_for_missing_prompt_key(self):
        trainer = self._make_trainer()
        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_record_prompt):
            with self.assertRaisesRegex(ValueError, "`prompt`"):
                list(trainer.load_prompt_batches_profiled(
                    [{"question": "raw"}], batch_size=1, max_prompt_tokens=3))

    def test_profiled_default_behavior_unchanged(self):
        """load_prompt_batches (unprofiled) must still work exactly as before."""
        trainer = self._make_trainer()
        dataset = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_record_prompt):
            batches = list(trainer.load_prompt_batches(dataset, batch_size=2, max_prompt_tokens=4))
        self.assertEqual([b.prompts for b in batches], [["a", "b"], ["c"]])

    def test_profiled_report_serializable_json(self):
        """to_dict() output must be JSON-serializable for MetricsRecorder."""
        trainer = self._make_trainer()
        dataset = [{"prompt": "a"}]
        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_record_prompt):
            results = list(trainer.load_prompt_batches_profiled(dataset, batch_size=1, max_prompt_tokens=4))
        _, report = results[0]
        s = json.dumps(report.to_dict())
        d = json.loads(s)
        self.assertEqual(d["records_scanned"], 1)
        self.assertIn("stages", d)


class TrainerConfigProfileTest(unittest.TestCase):
    """Config validation tests for the new profile fields."""

    def test_config_defaults_are_off(self):
        from areno.api.trainer_config import TrainerConfig
        cfg = TrainerConfig(algo="sft", ckpt="x", dataset_path="y")
        self.assertFalse(cfg.profile_dataset_stages)
        self.assertEqual(cfg.profile_slow_threshold_s, 1.0)

    def test_config_rejects_negative_threshold(self):
        from areno.api.trainer_config import TrainerConfig
        with self.assertRaisesRegex(ValueError, "profile_slow_threshold_s must be non-negative"):
            TrainerConfig(algo="sft", ckpt="x", dataset_path="y", profile_slow_threshold_s=-1.0)


class CliProfileOptionTest(unittest.TestCase):
    """CLI preflight validation for --profile-slow-threshold-s."""

    def test_cli_rejects_negative_threshold(self):
        from click import UsageError
        from click.testing import CliRunner
        from areno.cli.train import train_command
        runner = CliRunner()
        result = runner.invoke(
            train_command,
            ["--algo", "sft", "--ckpt", "x", "--dataset-path", "y",
             "--dataset-loader-fn", "examples/sft/alpaca/dataset_loader.py",
             "--profile-slow-threshold-s", "-1"],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("must be non-negative", result.output)


class ProfileTimeOutputTest(unittest.TestCase):
    """Print per-stage timing so the test output shows real numbers."""

    def test_print_stage_timing_summary(self):
        profiler = StageProfiler(enabled=True, slow_threshold_s=0.01)

        # Simulate 5 stages with a known delay injected into tokenize.
        with profiler.stage("record_access", index=0):
            pass
        with profiler.stage("contract_conversion", index=0):
            pass
        with profiler.stage("tokenize", index=0, inject_delay_s=0.05):
            pass
        with profiler.stage("filter", index=0):
            pass
        with profiler.stage("batch"):
            pass

        report = profiler.build_report(records_scanned=1, records_skipped_long=0, wall_seconds=0.06)

        print("\n" + "=" * 55)
        print("Stage timing summary (inject_delay_s=0.05 on tokenize):")
        print("=" * 55)
        for name in ["record_access", "contract_conversion", "tokenize", "filter", "batch"]:
            s = report.stages[name]
            print(f"  {name:25s}  calls={s.calls}  total={s.total_seconds:.6f}s")
        print(f"\n{report.render_human()}")

        # Verify the injected delay is attributed to the correct stage.
        self.assertGreaterEqual(report.stages["tokenize"].total_seconds, 0.045)
        self.assertLess(report.stages["filter"].total_seconds, 0.045)

    def test_print_profiled_batch_timing(self):
        trainer = Trainer(world_size=1, model_path="unused")
        trainer._tokenizer = object()
        dataset = [
            {"prompt": "a", "answer": "1"},
            {"prompt": "long", "answer": "skip"},
            {"prompt": "b", "answer": "2"},
            {"prompt": "c", "answer": "3"},
        ]

        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_record_prompt):
            results = list(trainer.load_prompt_batches_profiled(
                dataset, batch_size=2, max_prompt_tokens=3, profile_slow_threshold_s=0.001
            ))

        print("\n" + "=" * 55)
        print(f"Profiled batches: {len(results)} (from {len(dataset)} records)")
        print("=" * 55)
        for i, (batch, report) in enumerate(results):
            print(f"\n--- Batch {i}: prompts={batch.prompts} scanned={batch.scanned} skipped={batch.skipped_long} ---")
            for name in sorted(report.stages):
                s = report.stages[name]
                slow = f"  slow={len(s.slow_records)}" if s.slow_records else ""
                print(f"  {name:25s}  calls={s.calls:3d}  total={s.total_seconds:.6f}s{slow}")
            print(f"  wall_seconds={report.wall_seconds:.6f}s")

        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()