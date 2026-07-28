"""CPU tests for token-length distribution report.

These tests run without torch, CUDA, or transformers installed by loading
the data module directly via importlib, bypassing the heavy areno.api.__init__
import chain.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Load areno.api.data without triggering areno.api.__init__ (which imports torch).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_PY = _REPO_ROOT / "areno" / "api" / "data.py"

# Ensure numpy is importable (it's a dependency of data.py).
_spec = importlib.util.spec_from_file_location("areno.api.data", _DATA_PY)
_data_mod = importlib.util.module_from_spec(_spec)
sys.modules["areno.api.data"] = _data_mod
_spec.loader.exec_module(_data_mod)

# Re-export the symbols we need.
LengthStats = _data_mod.LengthStats
PromptItem = _data_mod.PromptItem
PromptBatch = _data_mod.PromptBatch
TokenLengthReport = _data_mod.TokenLengthReport
compute_token_length_report = _data_mod.compute_token_length_report

# Load areno.cli.diagnostics and areno.cli.main for CLI tests.
# These also avoid importing heavy modules at load time.
_DIAG_PY = _REPO_ROOT / "areno" / "cli" / "diagnostics.py"
_spec_diag = importlib.util.spec_from_file_location("areno.cli.diagnostics", _DIAG_PY)
diag_mod = importlib.util.module_from_spec(_spec_diag)
sys.modules["areno.cli.diagnostics"] = diag_mod
_spec_diag.loader.exec_module(diag_mod)

_MAIN_PY = _REPO_ROOT / "areno" / "cli" / "main.py"
_spec_main = importlib.util.spec_from_file_location("areno.cli.main", _MAIN_PY)
main_mod = importlib.util.module_from_spec(_spec_main)
sys.modules["areno.cli.main"] = main_mod
_spec_main.loader.exec_module(main_mod)


class _MockTokenizer:
    """Minimal tokenizer stub for CPU tests — encodes by splitting on words."""

    chat_template = None

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))


def _make_item(prompt_len: int, response: str | None = None) -> PromptItem:
    return PromptItem(
        prompt="word " * prompt_len,
        solutions=None,
        input_tokens=list(range(prompt_len)),
        record={"answer": response} if response is not None else {},
    )


class TokenLengthReportCoreTest(unittest.TestCase):
    """Core logic tests — no GPU, no real tokenizer."""

    def test_basic_stats_match_handcalc(self):
        items = [_make_item(n) for n in [10, 20, 30, 40, 50]]
        report = compute_token_length_report(items, max_context=100)

        self.assertEqual(report.prompt_stats.count, 5)
        self.assertEqual(report.prompt_stats.min, 10)
        self.assertEqual(report.prompt_stats.max, 50)
        self.assertEqual(report.prompt_stats.p50, 30)

    def test_empty_dataset_raises(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            compute_token_length_report([])

    def test_invalid_sample_ratio_zero_raises(self):
        items = [_make_item(10)]
        with self.assertRaisesRegex(ValueError, "sample_ratio"):
            compute_token_length_report(items, sample_ratio=0.0)

    def test_invalid_sample_ratio_above_one_raises(self):
        items = [_make_item(10)]
        with self.assertRaisesRegex(ValueError, "sample_ratio"):
            compute_token_length_report(items, sample_ratio=1.5)

    def test_sampling_determinism(self):
        items = [_make_item(n) for n in range(100)]
        r1 = compute_token_length_report(items, sample_ratio=0.1, sample_seed=42)
        r2 = compute_token_length_report(items, sample_ratio=0.1, sample_seed=42)
        self.assertEqual(r1.prompt_stats.as_dict(), r2.prompt_stats.as_dict())
        self.assertEqual(r1.sampled, r2.sampled)

    def test_full_scan_no_seed(self):
        items = [_make_item(n) for n in range(50)]
        report = compute_token_length_report(items, sample_ratio=1.0)
        self.assertEqual(report.sampled, 50)
        self.assertEqual(report.total_samples, 50)
        self.assertIsNone(report.sampling_seed)

    def test_over_context_pct(self):
        items = [_make_item(n) for n in [5, 10, 15, 20, 25]]
        report = compute_token_length_report(items, max_context=10)
        self.assertEqual(report.over_context_count, 3)
        self.assertAlmostEqual(report.over_context_pct, 60.0, places=1)

    def test_retained_under_drop_policy(self):
        items = [_make_item(n) for n in [5, 10, 15, 20, 25]]
        report = compute_token_length_report(items, max_context=10)
        self.assertEqual(report.retained_under_policy["drop"], 2)

    def test_retained_under_truncate_policy(self):
        items = [_make_item(n) for n in [5, 10, 15, 20, 25]]
        report = compute_token_length_report(items, max_context=10)
        self.assertEqual(report.retained_under_policy["truncate"], 5)

    def test_response_stats_with_tokenizer(self):
        items = [
            PromptItem(
                prompt="hello world",
                solutions=None,
                input_tokens=[0, 1],
                record={"answer": "foo bar baz"},
            )
        ]
        tok = _MockTokenizer()
        report = compute_token_length_report(items, response_field="answer", tokenizer=tok, max_context=100)
        self.assertIsNotNone(report.response_stats)
        self.assertEqual(report.response_stats.count, 1)
        self.assertEqual(report.response_stats.min, 3)
        self.assertIsNotNone(report.total_stats)
        self.assertEqual(report.total_stats.min, 5)

    def test_response_stats_none_without_field(self):
        items = [_make_item(10)]
        report = compute_token_length_report(items)
        self.assertIsNone(report.response_stats)
        self.assertIsNone(report.total_stats)

    def test_response_field_without_tokenizer_raises(self):
        items = [_make_item(10)]
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            compute_token_length_report(items, response_field="answer")

    def test_single_sample(self):
        items = [_make_item(42)]
        report = compute_token_length_report(items, max_context=100)
        self.assertEqual(report.prompt_stats.min, 42)
        self.assertEqual(report.prompt_stats.p50, 42)
        self.assertEqual(report.prompt_stats.max, 42)

    def test_all_same_length(self):
        items = [_make_item(30) for _ in range(10)]
        report = compute_token_length_report(items, max_context=100)
        self.assertEqual(report.prompt_stats.min, 30)
        self.assertEqual(report.prompt_stats.p50, 30)
        self.assertEqual(report.prompt_stats.p99, 30)
        self.assertEqual(report.prompt_stats.max, 30)

    def test_sampling_seed_recorded(self):
        items = [_make_item(n) for n in range(100)]
        report = compute_token_length_report(items, sample_ratio=0.5, sample_seed=123)
        self.assertEqual(report.sampling_seed, 123)

    def test_as_dict_serializable(self):
        items = [_make_item(10), _make_item(20)]
        report = compute_token_length_report(items, max_context=100)
        d = report.as_dict()
        json.dumps(d)
        self.assertIn("prompt_stats", d)
        self.assertIn("over_context_pct", d)


class TokenLengthReportCLITest(unittest.TestCase):
    """CLI tests using CliRunner with mocked internals."""

    def _make_jsonl(self, tmp: str) -> str:
        path = Path(tmp) / "data.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for lengths in [2, 4, 6, 8, 10]:
                f.write(json.dumps({"prompt": "word " * lengths, "answer": "a " * lengths}) + "\n")
        return str(path)

    def test_cli_json_output(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_jsonl(tmp)
            with patch.object(diag_mod, "_run_token_report") as mock_run:
                mock_run.return_value = TokenLengthReport(
                    total_samples=5,
                    sampled=5,
                    sampling_seed=None,
                    max_context=4096,
                    prompt_stats=LengthStats(5, 2, 6, 10, 10, 10, 10, 6.0),
                    response_stats=LengthStats(5, 2, 6, 10, 10, 10, 10, 6.0),
                    total_stats=LengthStats(5, 4, 12, 20, 20, 20, 20, 12.0),
                    over_context_count=0,
                    over_context_pct=0.0,
                    retained_under_policy={"drop": 5, "truncate": 5},
                )
                result = runner.invoke(
                    diag_mod.token_report_command,
                    ["--dataset-path", dataset, "--tokenizer", "fake", "--json"],
                )

        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        self.assertEqual(parsed["total_samples"], 5)
        self.assertEqual(parsed["prompt_stats"]["p50"], 6)
        self.assertEqual(parsed["over_context_pct"], 0.0)

    def test_cli_human_output(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_jsonl(tmp)
            with patch.object(diag_mod, "_run_token_report") as mock_run:
                mock_run.return_value = TokenLengthReport(
                    total_samples=5,
                    sampled=5,
                    sampling_seed=None,
                    max_context=4096,
                    prompt_stats=LengthStats(5, 2, 6, 10, 10, 10, 10, 6.0),
                    response_stats=None,
                    total_stats=None,
                    over_context_count=0,
                    over_context_pct=0.0,
                    retained_under_policy={"drop": 5, "truncate": 5},
                )
                result = runner.invoke(
                    diag_mod.token_report_command,
                    ["--dataset-path", dataset, "--tokenizer", "fake"],
                )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Token Length Distribution Report", result.output)
        self.assertIn("Prompt Length", result.output)
        self.assertIn("Over-context", result.output)
        self.assertIn("Retained under policy", result.output)

    def test_cli_registered_in_main(self):
        result = CliRunner().invoke(main_mod.main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("token-report", result.output)

    def test_cli_invalid_sample_ratio(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_jsonl(tmp)
            result = runner.invoke(
                diag_mod.token_report_command,
                ["--dataset-path", dataset, "--tokenizer", "fake", "--sample-ratio", "0"],
            )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("sample_ratio", result.output)

    def test_cli_missing_dataset_file(self):
        runner = CliRunner()
        result = runner.invoke(
            diag_mod.token_report_command,
            ["--dataset-path", "/nonexistent/path.jsonl", "--tokenizer", "fake"],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("not found", result.output)


class BackwardCompatibilityTest(unittest.TestCase):
    """Existing PromptItem and PromptBatch behavior must not change."""

    def test_prompt_item_fields_unchanged(self):
        item = PromptItem(
            prompt="test",
            solutions=["a"],
            input_tokens=[1, 2, 3],
            record={"key": "value"},
        )
        self.assertEqual(item.prompt, "test")
        self.assertEqual(item.input_tokens, [1, 2, 3])
        self.assertEqual(item.record["key"], "value")

    def test_prompt_batch_prompts_property_unchanged(self):
        items = [_make_item(5), _make_item(10)]
        batch = PromptBatch(items=items, scanned=2, skipped_long=0, total_skipped_long=0)
        self.assertEqual(batch.prompts, [item.prompt for item in items])


class TokenLengthReportIntegrationTest(unittest.TestCase):
    """Integration test: load JSONL fixture → tokenize with mock → compute → assert.

    Exercises the full compute pipeline (PromptItem construction →
    compute_token_length_report → as_dict → JSON serialize) with a tiny
    deterministic fixture and a mock tokenizer. No real HuggingFace model
    or GPU required.
    """

    def _load_fixture(self) -> list[PromptItem]:
        """Build PromptItems from hand-crafted records with known token lengths.

        Using _MockTokenizer (word-count encoding), the prompt and response
        token lengths are deterministic:

            prompt tokens:   [2, 4, 6, 8, 3]
            response tokens: [2, 3, 4, 2, 1]
            total tokens:    [4, 7, 10, 10, 4]
        """
        records = [
            ("hello world", "yes no"),
            ("the quick brown fox", "a b c"),
            ("a b c d e f", "foo bar baz qux"),
            ("one two three four five six seven eight", "alpha beta"),
            ("x y z", "p"),
        ]
        items = []
        for prompt_text, answer_text in records:
            prompt_ids = list(range(len(prompt_text.split())))
            items.append(
                PromptItem(
                    prompt=prompt_text,
                    solutions=None,
                    input_tokens=prompt_ids,
                    record={"answer": answer_text},
                )
            )
        return items

    def test_integration_prompt_and_response_stats(self):
        """Verify prompt, response, and total length stats with known values."""
        items = self._load_fixture()
        tok = _MockTokenizer()
        report = compute_token_length_report(items, max_context=5, response_field="answer", tokenizer=tok)
        d = report.as_dict()

        # Prompt lengths: [2, 4, 6, 8, 3]
        self.assertEqual(d["prompt_stats"]["count"], 5)
        self.assertEqual(d["prompt_stats"]["min"], 2)
        self.assertEqual(d["prompt_stats"]["max"], 8)

        # Response lengths: [2, 3, 4, 2, 1]
        self.assertEqual(d["response_stats"]["min"], 1)
        self.assertEqual(d["response_stats"]["max"], 4)

        # Total lengths: [4, 7, 10, 10, 4]
        self.assertEqual(d["total_stats"]["min"], 4)
        self.assertEqual(d["total_stats"]["max"], 10)

        # Over-context (total > 5): 7, 10, 10 → 3 out of 5 = 60%
        self.assertEqual(d["over_context_count"], 3)
        self.assertAlmostEqual(d["over_context_pct"], 60.0, places=1)
        self.assertEqual(d["retained_under_policy"]["drop"], 2)
        self.assertEqual(d["retained_under_policy"]["truncate"], 5)

        # Must be JSON-serializable end-to-end
        json.dumps(d)

    def test_integration_jsonl_file_to_report(self):
        """Read a real JSONL file, build PromptItems, compute, and verify."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.jsonl"
            records = [
                {"prompt": "hello world", "answer": "yes no"},
                {"prompt": "the quick brown fox", "answer": "a b c"},
                {"prompt": "x y z", "answer": "p"},
            ]
            with path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            # Load JSONL the same way _run_token_report does
            tok = _MockTokenizer()
            items = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    prompt_text = record.get("prompt", "")
                    items.append(
                        PromptItem(
                            prompt=prompt_text,
                            solutions=None,
                            input_tokens=list(range(len(prompt_text.split()))),
                            record=record,
                        )
                    )

            report = compute_token_length_report(items, max_context=100, response_field="answer", tokenizer=tok)
            d = report.as_dict()

            self.assertEqual(d["total_samples"], 3)
            self.assertEqual(d["prompt_stats"]["count"], 3)
            # prompt lengths: [2, 4, 3]
            self.assertEqual(d["prompt_stats"]["min"], 2)
            self.assertEqual(d["prompt_stats"]["max"], 4)
            # response lengths: [2, 3, 1]
            self.assertEqual(d["response_stats"]["min"], 1)
            self.assertEqual(d["response_stats"]["max"], 3)

    def test_integration_boundary_empty_file(self):
        """An empty JSONL file should produce zero items and raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("\n\n\n", encoding="utf-8")

            items = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))

            self.assertEqual(len(items), 0)
            with self.assertRaisesRegex(ValueError, "empty"):
                compute_token_length_report(items)

    def test_integration_disabled_by_default(self):
        """The feature is opt-in — compute_token_length_report is never called
        unless the user explicitly invokes it. Verify that PromptItem and
        PromptBatch work normally without the report."""
        items = self._load_fixture()
        # Normal usage without report — just build a PromptBatch
        batch = PromptBatch(items=items, scanned=5, skipped_long=0, total_skipped_long=0)
        self.assertEqual(len(batch.items), 5)
        self.assertEqual(batch.prompts[0], "hello world")
        self.assertEqual(batch.scanned, 5)


if __name__ == "__main__":
    unittest.main()
