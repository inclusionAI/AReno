"""CPU tests for the completion-quality summary feature.

Covers:
- Data model changes (RolloutSequence.finish_reason, AgentTrainBatch.filtered_count).
- Trainer-side stat computation (_compute_single_turn / _compute_agentic).
- CLI loading, aggregation, fallback, and output formatting.
- MetricsRecorder.record_completion_summary file I/O.
"""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from pathlib import Path

from click.testing import CliRunner

from areno.api import metrics as metrics_mod
from areno.api.metrics import MetricsRecorder, compute_percentile
from areno.api.models import RolloutSequence, RolloutResult
from areno.cli import completion_summary as cs_mod
from areno.cli.completion_summary import completion_summary_command
from areno.cli.main import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rollout_result(num_seqs: int, lengths: list[int], finish_reasons: list[str]) -> RolloutResult:
    seqs = []
    for i in range(num_seqs):
        seqs.append(
            RolloutSequence(
                resp_tokens=list(range(lengths[i])),
                resp_logprobs=[0.0] * lengths[i],
                finish_reason=finish_reasons[i],
            )
        )
    return RolloutResult(sequences=seqs)


class _AgentBatchStub:
    """Minimal stand-in for AgentTrainBatch."""

    def __init__(self, loss_masks, token_rows=None, filtered_count=0):
        self.loss_masks = loss_masks
        self.token_rows = token_rows or ([[]] * len(loss_masks))
        self.filtered_count = filtered_count
        self.reward_records = []


class _TrainerStub:
    """Minimal object exposing _percentile_value and _agent_model_context_len."""

    def _percentile_value(self, sorted_values, fraction):
        if not sorted_values:
            return 0
        index = min(int(round((len(sorted_values) - 1) * fraction)), len(sorted_values) - 1)
        return int(sorted_values[index])

    def _agent_model_context_len(self):
        return None


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------

class FinishReasonModelTest(unittest.TestCase):

    def test_rollout_sequence_has_finish_reason_field(self):
        seq = RolloutSequence(resp_tokens=[1, 2], resp_logprobs=[0.0, 0.0], finish_reason="length")
        self.assertEqual(seq.finish_reason, "length")

    def test_rollout_sequence_finish_reason_defaults_empty(self):
        seq = RolloutSequence(resp_tokens=[1])
        self.assertEqual(seq.finish_reason, "")


class AgentTrainBatchFilteredCountTest(unittest.TestCase):

    def test_default_filtered_count_is_zero(self):
        from areno.api.agentic import AgentTrainBatch

        batch = AgentTrainBatch(
            token_rows=[],
            response_masks=[],
            loss_masks=[],
            rollout_logprobs=[],
            rewards=[],
            records=[],
            reward_records=[],
        )
        self.assertEqual(batch.filtered_count, 0)

    def test_filtered_count_set(self):
        from areno.api.agentic import AgentTrainBatch

        batch = AgentTrainBatch(
            token_rows=[],
            response_masks=[],
            loss_masks=[],
            rollout_logprobs=[],
            rewards=[],
            records=[],
            reward_records=[],
            filtered_count=5,
        )
        self.assertEqual(batch.filtered_count, 5)


# ---------------------------------------------------------------------------
# Trainer-side computation tests
# ---------------------------------------------------------------------------

class SingleTurnSummaryTest(unittest.TestCase):

    def setUp(self):
        self.stub = _TrainerStub()
        # Bind the real method from PolicyOnlyTrainer.
        from areno.api.trainers.policy_only import PolicyOnlyTrainer

        self._compute = PolicyOnlyTrainer._compute_single_turn_completion_summary.__get__(self.stub)

    def test_counts_finish_reasons(self):
        result = _make_rollout_result(
            6,
            lengths=[0, 10, 20, 30, 0, 40],
            finish_reasons=["stop", "stop", "length", "stop", "length", "tool_calls"],
        )
        summary = self._compute(epoch=0, step=0, rollout_results=[result], rollout_time_s=10.0)
        self.assertEqual(summary["total_completions"], 6)
        self.assertEqual(summary["empty_count"], 2)
        self.assertEqual(summary["length_limit_count"], 2)
        self.assertEqual(summary["stop_count"], 3)
        self.assertEqual(summary["tool_calls_count"], 1)
        self.assertEqual(summary["filtered_count"], 0)

    def test_all_empty_completions(self):
        result = _make_rollout_result(
            4, lengths=[0, 0, 0, 0], finish_reasons=["length", "length", "length", "length"]
        )
        summary = self._compute(epoch=0, step=0, rollout_results=[result], rollout_time_s=5.0)
        self.assertEqual(summary["empty_count"], 4)
        self.assertEqual(summary["total_generated_tokens"], 0)

    def test_tokens_per_second(self):
        result = _make_rollout_result(2, lengths=[100, 200], finish_reasons=["stop", "stop"])
        summary = self._compute(epoch=0, step=0, rollout_results=[result], rollout_time_s=10.0)
        self.assertAlmostEqual(summary["tokens_per_second"], 30.0, places=1)

    def test_tokens_per_second_zero_rollout_time(self):
        result = _make_rollout_result(2, lengths=[100, 200], finish_reasons=["stop", "stop"])
        summary = self._compute(epoch=0, step=0, rollout_results=[result], rollout_time_s=0.0)
        self.assertEqual(summary["tokens_per_second"], 0.0)

    def test_length_distribution(self):
        result = _make_rollout_result(3, lengths=[10, 50, 100], finish_reasons=["stop", "stop", "stop"])
        summary = self._compute(epoch=0, step=0, rollout_results=[result], rollout_time_s=1.0)
        self.assertEqual(summary["completion_length_min"], 10)
        self.assertEqual(summary["completion_length_max"], 100)
        self.assertAlmostEqual(summary["completion_length_mean"], 53.333, places=1)
        self.assertEqual(summary["completion_lengths"], [10, 50, 100])


class AgenticSummaryTest(unittest.TestCase):

    def setUp(self):
        self.stub = _TrainerStub()
        from areno.api.trainers.policy_only import PolicyOnlyTrainer

        self._compute = PolicyOnlyTrainer._compute_agentic_completion_summary.__get__(self.stub)

    def test_counts_loss_mask_tokens(self):
        batch = _AgentBatchStub(
            loss_masks=[
                [False, True, True, True, False, True],  # 4 trainable
                [False, False, True, True],  # 2 trainable
            ],
        )
        summary = self._compute(epoch=0, step=0, agent_batch=batch, rollout_time_s=8.0)
        self.assertEqual(summary["total_completions"], 2)
        self.assertEqual(summary["total_generated_tokens"], 6)
        self.assertEqual(summary["completion_lengths"], [4, 2])

    def test_includes_filtered_count(self):
        batch = _AgentBatchStub(loss_masks=[[True, True]], filtered_count=3)
        summary = self._compute(epoch=0, step=0, agent_batch=batch, rollout_time_s=1.0)
        self.assertEqual(summary["filtered_count"], 3)

    def test_empty_completion_detected(self):
        batch = _AgentBatchStub(loss_masks=[[False, False, False], [True, True]])
        summary = self._compute(epoch=0, step=0, agent_batch=batch, rollout_time_s=2.0)
        self.assertEqual(summary["empty_count"], 1)

    def test_stop_and_tool_calls_are_negative_one(self):
        batch = _AgentBatchStub(loss_masks=[[True]])
        summary = self._compute(epoch=0, step=0, agent_batch=batch, rollout_time_s=1.0)
        self.assertEqual(summary["stop_count"], -1)
        self.assertEqual(summary["tool_calls_count"], -1)


# ---------------------------------------------------------------------------
# compute_percentile helper tests
# ---------------------------------------------------------------------------

class ComputePercentileTest(unittest.TestCase):

    def test_median(self):
        self.assertAlmostEqual(compute_percentile([1, 2, 3, 4, 5], 0.5), 3.0)

    def test_p90(self):
        result = compute_percentile(list(range(1, 11)), 0.9)
        self.assertAlmostEqual(result, 9.1, places=1)

    def test_empty_list(self):
        self.assertEqual(compute_percentile([], 0.5), 0.0)

    def test_single_element(self):
        self.assertAlmostEqual(compute_percentile([42], 0.5), 42.0)


# ---------------------------------------------------------------------------
# CLI loading & aggregation tests
# ---------------------------------------------------------------------------

class CompletionSummaryLoadingTest(unittest.TestCase):

    def test_load_completion_summary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completion_summary.12345.jsonl"
            records = [
                {"step": 0, "kind": "rollout", "total_completions": 8},
                {"step": 1, "kind": "rollout", "total_completions": 8},
            ]
            with path.open("w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            loaded, source = cs_mod._load_completion_summary_files(Path(tmp), None)
            self.assertEqual(source, "completion_summary")
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["total_completions"], 8)

    def test_load_with_pid_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            for pid in (111, 222):
                path = Path(tmp) / f"completion_summary.{pid}.jsonl"
                with path.open("w") as f:
                    f.write(json.dumps({"step": 0, "kind": "rollout", "pid": pid}) + "\n")
            loaded, _ = cs_mod._load_completion_summary_files(Path(tmp), 111)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["pid"], 111)

    def test_fallback_to_rollout_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout_samples.99999.jsonl"
            samples = [
                {"kind": "rollout", "epoch": 0, "step": 0, "response_tokens": [1, 2, 3], "finish_reason": "stop"},
                {"kind": "rollout", "epoch": 0, "step": 0, "response_tokens": [], "finish_reason": "length"},
                {"kind": "rollout", "epoch": 0, "step": 1, "response_tokens": [1], "finish_reason": "stop"},
            ]
            with path.open("w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            loaded, source = cs_mod._load_completion_summary_files(Path(tmp), None)
            self.assertEqual(source, "rollout_samples")
            self.assertEqual(len(loaded), 2)  # Two steps
            self.assertEqual(loaded[0]["total_completions"], 2)
            self.assertEqual(loaded[0]["empty_count"], 1)
            self.assertEqual(loaded[0]["length_limit_count"], 1)

    def test_no_files_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, source = cs_mod._load_completion_summary_files(Path(tmp), None)
            self.assertEqual(loaded, [])
            self.assertEqual(source, "none")

    def test_missing_fields_filled_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completion_summary.1.jsonl"
            with path.open("w") as f:
                # Minimal record—no stop_count, completion_lengths, etc.
                f.write(json.dumps({"step": 0, "kind": "rollout", "total_completions": 4}) + "\n")
            loaded, _ = cs_mod._load_completion_summary_files(Path(tmp), None)
            self.assertEqual(loaded[0]["stop_count"], -1)  # default
            self.assertEqual(loaded[0]["filtered_count"], 0)  # default
            self.assertEqual(loaded[0]["completion_lengths"], [])  # default


class AggregationTest(unittest.TestCase):

    def test_aggregate_overall_merges_completion_lengths(self):
        records = [
            {"kind": "rollout", "completion_lengths": [10, 20], "total_generated_tokens": 30, "rollout_time_s": 1.0,
             "total_completions": 2, "empty_count": 0, "length_limit_count": 0, "stop_count": 2,
             "tool_calls_count": 0, "filtered_count": 0},
            {"kind": "rollout", "completion_lengths": [30, 40], "total_generated_tokens": 70, "rollout_time_s": 2.0,
             "total_completions": 2, "empty_count": 0, "length_limit_count": 1, "stop_count": 1,
             "tool_calls_count": 0, "filtered_count": 0},
        ]
        overall = cs_mod._aggregate_overall(records)
        rollout = overall["rollout"]
        self.assertEqual(rollout["total_completions"], 4)
        self.assertEqual(rollout["completion_length"]["min"], 10)
        self.assertEqual(rollout["completion_length"]["max"], 40)

    def test_aggregate_tokens_per_second_uses_total(self):
        records = [
            {"kind": "rollout", "total_generated_tokens": 100, "rollout_time_s": 10.0, "completion_lengths": [50, 50],
             "total_completions": 2, "empty_count": 0, "length_limit_count": 0, "stop_count": 2,
             "tool_calls_count": 0, "filtered_count": 0},
            {"kind": "rollout", "total_generated_tokens": 200, "rollout_time_s": 10.0, "completion_lengths": [100, 100],
             "total_completions": 2, "empty_count": 0, "length_limit_count": 0, "stop_count": 2,
             "tool_calls_count": 0, "filtered_count": 0},
        ]
        overall = cs_mod._aggregate_overall(records)
        # 300 total tokens / 20 total seconds = 15.0
        self.assertAlmostEqual(overall["rollout"]["tokens_per_second"], 15.0, places=1)

    def test_aggregate_empty_records_returns_none(self):
        overall = cs_mod._aggregate_overall([])
        self.assertIsNone(overall["rollout"])
        self.assertIsNone(overall["agentic"])

    def test_aggregate_mixed_kinds(self):
        records = [
            {"kind": "rollout", "total_generated_tokens": 10, "rollout_time_s": 1.0,
             "completion_lengths": [10], "total_completions": 1, "empty_count": 0,
             "length_limit_count": 0, "stop_count": 1, "tool_calls_count": 0, "filtered_count": 0},
            {"kind": "agentic", "total_generated_tokens": 20, "rollout_time_s": 2.0,
             "completion_lengths": [20], "total_completions": 1, "empty_count": 0,
             "length_limit_count": 0, "stop_count": -1, "tool_calls_count": -1, "filtered_count": 1},
        ]
        overall = cs_mod._aggregate_overall(records)
        self.assertIsNotNone(overall["rollout"])
        self.assertIsNotNone(overall["agentic"])
        self.assertEqual(overall["agentic"]["filtered_count"], 1)


# ---------------------------------------------------------------------------
# CLI command tests (via Click CliRunner)
# ---------------------------------------------------------------------------

class CompletionSummaryCliTest(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()

    def _write_summary_file(self, tmp: str, records: list[dict], pid: int = 12345):
        path = Path(tmp) / f"completion_summary.{pid}.jsonl"
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def test_json_output_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {"step": 0, "kind": "rollout", "total_completions": 8, "total_generated_tokens": 64,
                 "empty_count": 0, "length_limit_count": 1, "stop_count": 7, "tool_calls_count": 0,
                 "filtered_count": 0, "completion_lengths": [8] * 8, "rollout_time_s": 5.0,
                 "tokens_per_second": 12.8, "completion_length_min": 8, "completion_length_max": 8,
                 "completion_length_mean": 8.0, "completion_length_p50": 8, "completion_length_p90": 8},
                {"step": 1, "kind": "rollout", "total_completions": 8, "total_generated_tokens": 80,
                 "empty_count": 1, "length_limit_count": 0, "stop_count": 7, "tool_calls_count": 0,
                 "filtered_count": 0, "completion_lengths": [0, 10, 10, 10, 10, 10, 10, 20],
                 "rollout_time_s": 6.0, "tokens_per_second": 13.3, "completion_length_min": 0,
                 "completion_length_max": 20, "completion_length_mean": 10.0,
                 "completion_length_p50": 10, "completion_length_p90": 20},
            ]
            self._write_summary_file(tmp, records)
            result = self.runner.invoke(completion_summary_command, ["--metrics-log-dir", tmp, "--json"])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            data = json.loads(result.output)
            self.assertEqual(data["source"], "completion_summary")
            self.assertEqual(data["overall"]["rollout"]["num_steps"], 2)
            self.assertEqual(data["overall"]["rollout"]["total_completions"], 16)
            self.assertEqual(data["overall"]["rollout"]["empty_count"], 1)
            self.assertEqual(len(data["per_step"]), 2)

    def test_human_readable_output_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {"step": 0, "kind": "rollout", "total_completions": 4, "total_generated_tokens": 40,
                 "empty_count": 0, "length_limit_count": 0, "stop_count": 4, "tool_calls_count": 0,
                 "filtered_count": 0, "completion_lengths": [10] * 4, "rollout_time_s": 2.0,
                 "tokens_per_second": 20.0, "completion_length_min": 10, "completion_length_max": 10,
                 "completion_length_mean": 10.0, "completion_length_p50": 10, "completion_length_p90": 10},
            ]
            self._write_summary_file(tmp, records)
            result = self.runner.invoke(completion_summary_command, ["--metrics-log-dir", tmp])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("Completion Quality Summary", result.output)
            self.assertIn("rollout", result.output)
            self.assertIn("Per-step detail", result.output)

    def test_pid_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_summary_file(tmp, [{"step": 0, "kind": "rollout", "total_completions": 1,
                "completion_lengths": [1]}], pid=111)
            self._write_summary_file(tmp, [{"step": 0, "kind": "rollout", "total_completions": 1,
                "completion_lengths": [1]}], pid=222)
            result = self.runner.invoke(
                completion_summary_command, ["--metrics-log-dir", tmp, "--json", "--pid", "222"]
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            data = json.loads(result.output)
            self.assertEqual(len(data["per_step"]), 1)

    def test_no_files_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(completion_summary_command, ["--metrics-log-dir", tmp])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("no completion_summary", result.output.lower())

    def test_handles_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completion_summary.999.jsonl"
            # Intentionally sparse record.
            with path.open("w") as f:
                f.write(json.dumps({"step": 0, "kind": "rollout", "total_completions": 2}) + "\n")
            result = self.runner.invoke(
                completion_summary_command, ["--metrics-log-dir", tmp, "--json"]
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            data = json.loads(result.output)
            self.assertEqual(data["overall"]["rollout"]["total_completions"], 2)

    def test_mixed_kinds_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {"step": 0, "kind": "rollout", "total_completions": 4, "total_generated_tokens": 40,
                 "empty_count": 0, "length_limit_count": 0, "stop_count": 4, "tool_calls_count": 0,
                 "filtered_count": 0, "completion_lengths": [10] * 4, "rollout_time_s": 2.0,
                 "tokens_per_second": 20.0, "completion_length_min": 10, "completion_length_max": 10,
                 "completion_length_mean": 10.0, "completion_length_p50": 10, "completion_length_p90": 10},
                {"step": 1, "kind": "agentic", "total_completions": 2, "total_generated_tokens": 60,
                 "empty_count": 0, "length_limit_count": 0, "stop_count": -1, "tool_calls_count": -1,
                 "filtered_count": 1, "completion_lengths": [30, 30], "rollout_time_s": 3.0,
                 "tokens_per_second": 20.0, "completion_length_min": 30, "completion_length_max": 30,
                 "completion_length_mean": 30.0, "completion_length_p50": 30, "completion_length_p90": 30},
            ]
            self._write_summary_file(tmp, records)
            result = self.runner.invoke(
                completion_summary_command, ["--metrics-log-dir", tmp, "--json"]
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            data = json.loads(result.output)
            self.assertIsNotNone(data["overall"]["rollout"])
            self.assertIsNotNone(data["overall"]["agentic"])
            self.assertEqual(data["overall"]["agentic"]["filtered_count"], 1)

    def test_command_registered_in_main(self):
        result = self.runner.invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("completion-summary", result.output)

    def test_fallback_from_rollout_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout_samples.555.jsonl"
            with path.open("w") as f:
                f.write(json.dumps({"kind": "rollout", "epoch": 0, "step": 0,
                    "response_tokens": [1, 2, 3], "response_len": 3, "finish_reason": "stop"}) + "\n")
                f.write(json.dumps({"kind": "rollout", "epoch": 0, "step": 0,
                    "response_tokens": [], "response_len": 0, "finish_reason": "length"}) + "\n")
            result = self.runner.invoke(
                completion_summary_command, ["--metrics-log-dir", tmp, "--json"]
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            data = json.loads(result.output)
            self.assertEqual(data["source"], "rollout_samples")
            self.assertEqual(data["overall"]["rollout"]["total_completions"], 2)
            self.assertEqual(data["overall"]["rollout"]["empty_count"], 1)


# ---------------------------------------------------------------------------
# MetricsRecorder.record_completion_summary I/O test
# ---------------------------------------------------------------------------

class RecordCompletionSummaryTest(unittest.TestCase):

    def test_record_completion_summary_writes_jsonl(self):
        class FakeWriter:
            def close(self):
                pass

        old_factory = metrics_mod.create_tensorboard_writer
        metrics_mod.create_tensorboard_writer = lambda _log_dir: FakeWriter()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with MetricsRecorder(tmp) as recorder:
                    recorder.record_completion_summary({"step": 0, "kind": "rollout", "total_completions": 4})
                    recorder.record_completion_summary({"step": 1, "kind": "rollout", "total_completions": 4})
                # Verify the file exists and contains 2 lines.
                files = list(Path(tmp).glob("completion_summary.*.jsonl"))
                self.assertEqual(len(files), 1)
                lines = files[0].read_text().strip().split("\n")
                self.assertEqual(len(lines), 2)
                first = json.loads(lines[0])
                self.assertEqual(first["step"], 0)
                self.assertEqual(first["pid"], os.getpid())
        finally:
            metrics_mod.create_tensorboard_writer = old_factory


if __name__ == "__main__":
    unittest.main()