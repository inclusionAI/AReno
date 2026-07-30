"""CPU tests for the ``areno data inspect`` CLI command.

These tests invoke the ``data inspect`` Click command with mocked dataset
loading to verify terminal output, JSON output, contract validation
success and failure, and backward-compatible default behaviour.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from areno.cli import data_inspect


def _mock_load_dataset(records):
    """Return a patcher that replaces _load_dataset_for_training in data_inspect."""

    return patch.object(
        data_inspect,
        "_load_dataset",
        return_value=records,
    )


class DataInspectCliTest(unittest.TestCase):
    """The data inspect CLI should load, display, and validate datasets."""

    def test_inspect_without_contract_shows_record_summary(self):
        """Without --contract the command should show a dataset summary."""
        records = [
            {"prompt": "Hello", "response": "Hi"},
            {"prompt": "World", "response": "Hey"},
        ]
        with _mock_load_dataset(records):
            result = CliRunner().invoke(
                data_inspect.data_command,
                ["inspect", "--dataset-path", "dummy", "--model-hub", "hf"],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("records: 2", result.output)
        self.assertIn("prompt: str", result.output)
        self.assertIn("response: str", result.output)

    def test_inspect_without_contract_json_output(self):
        """JSON output without --contract should include total_records."""
        records = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
        with _mock_load_dataset(records):
            result = CliRunner().invoke(
                data_inspect.data_command,
                ["inspect", "--dataset-path", "dummy", "--model-hub", "hf", "--json"],
            )
        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        self.assertEqual(parsed["total_records"], 3)
        self.assertIsNone(parsed["contract"])

    def test_inspect_contract_passes_for_valid_sft(self):
        """A valid SFT dataset should produce a passing contract report."""
        records = [{"prompt": "What is 2+2?", "response": "4"}]
        with _mock_load_dataset(records):
            result = CliRunner().invoke(
                data_inspect.data_command,
                [
                    "inspect",
                    "--dataset-path", "dummy",
                    "--model-hub", "hf",
                    "--contract",
                    "--mode", "sft",
                ],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("passed", result.output)

    def test_inspect_contract_fails_for_invalid_sft(self):
        """An invalid SFT dataset should exit with code 1 and show errors."""
        records = [{"prompt": "Missing response"}]
        with _mock_load_dataset(records):
            result = CliRunner().invoke(
                data_inspect.data_command,
                [
                    "inspect",
                    "--dataset-path", "dummy",
                    "--model-hub", "hf",
                    "--contract",
                    "--mode", "sft",
                ],
            )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("failed", result.output)
        self.assertIn("ERROR", result.output)
        self.assertIn("response", result.output)

    def test_inspect_contract_json_output_for_invalid(self):
        """JSON output with --contract should include structured errors."""
        records = [{"prompt": 42}]
        with _mock_load_dataset(records):
            result = CliRunner().invoke(
                data_inspect.data_command,
                [
                    "inspect",
                    "--dataset-path", "dummy",
                    "--model-hub", "hf",
                    "--contract",
                    "--mode", "sft",
                    "--json",
                ],
            )
        self.assertEqual(result.exit_code, 1)
        parsed = json.loads(result.output)
        self.assertFalse(parsed["ok"])
        self.assertGreater(len(parsed["errors"]), 0)
        self.assertEqual(parsed["mode"], "sft")

    def test_contract_without_mode_raises_usage_error(self):
        """--contract without --mode should produce a usage error."""
        records = [{"prompt": "a", "response": "b"}]
        with _mock_load_dataset(records):
            result = CliRunner().invoke(
                data_inspect.data_command,
                [
                    "inspect",
                    "--dataset-path", "dummy",
                    "--model-hub", "hf",
                    "--contract",
                ],
            )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--mode is required", result.output)

    def test_inspect_dpo_valid(self):
        """A valid DPO dataset should pass contract validation."""
        records = [
            {"prompt": "Q", "chosen": "Good A", "rejected": "Bad A"},
        ]
        with _mock_load_dataset(records):
            result = CliRunner().invoke(
                data_inspect.data_command,
                [
                    "inspect",
                    "--dataset-path", "dummy",
                    "--model-hub", "hf",
                    "--contract",
                    "--mode", "dpo",
                ],
            )
        self.assertEqual(result.exit_code, 0)

    def test_inspect_online_rl_valid(self):
        """A valid online RL dataset should pass contract validation."""
        records = [{"prompt": "Solve this.", "solutions": ["answer"]}]
        with _mock_load_dataset(records):
            result = CliRunner().invoke(
                data_inspect.data_command,
                [
                    "inspect",
                    "--dataset-path", "dummy",
                    "--model-hub", "hf",
                    "--contract",
                    "--mode", "online_rl",
                ],
            )
        self.assertEqual(result.exit_code, 0)

    def test_inspect_agentic_valid_prompt(self):
        """A valid agentic dataset with prompt-only format should pass."""
        records = [{"prompt": "Play a game."}]
        with _mock_load_dataset(records):
            result = CliRunner().invoke(
                data_inspect.data_command,
                [
                    "inspect",
                    "--dataset-path", "dummy",
                    "--model-hub", "hf",
                    "--contract",
                    "--mode", "agentic",
                ],
            )
        self.assertEqual(result.exit_code, 0)

    def test_inspect_agentic_valid_messages(self):
        """A valid agentic dataset with messages format should pass."""
        records = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ]
            }
        ]
        with _mock_load_dataset(records):
            result = CliRunner().invoke(
                data_inspect.data_command,
                [
                    "inspect",
                    "--dataset-path", "dummy",
                    "--model-hub", "hf",
                    "--contract",
                    "--mode", "agentic",
                ],
            )
        self.assertEqual(result.exit_code, 0)

    def test_max_samples_limit(self):
        """--max-samples should limit the number of records scanned."""
        records = [{"prompt": f"p{i}", "response": f"r{i}"} for i in range(20)]
        with _mock_load_dataset(records):
            result = CliRunner().invoke(
                data_inspect.data_command,
                [
                    "inspect",
                    "--dataset-path", "dummy",
                    "--model-hub", "hf",
                    "--contract",
                    "--mode", "sft",
                    "--max-samples", "5",
                ],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("scanned=5", result.output)

    def test_inspect_shows_hint_for_errors(self):
        """Error output should include the hint line."""
        records = [{"prompt": "no response here"}]
        with _mock_load_dataset(records):
            result = CliRunner().invoke(
                data_inspect.data_command,
                [
                    "inspect",
                    "--dataset-path", "dummy",
                    "--model-hub", "hf",
                    "--contract",
                    "--mode", "sft",
                ],
            )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("hint:", result.output)


if __name__ == "__main__":
    unittest.main()
