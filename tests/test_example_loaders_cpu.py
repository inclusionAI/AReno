"""Keep the OPD and SFT example loaders on the same prompts.

Both loaders are standalone files passed through ``--dataset-loader-fn`` and
imported by path, so they cannot import each other. The prompt text is
duplicated on purpose; these tests fail as soon as the two copies drift apart,
which would silently break the "same questions, comparable numbers" claim of
the SFT baseline.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

GSM8K_ROWS = [
    {
        "question": (
            "Natalia sold clips to 48 of her friends in April, and then she "
            "sold half as many clips in May. How many clips did Natalia sell "
            "altogether in April and May?"
        ),
        "answer": (
            "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
            "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April "
            "and May.\n#### 72"
        ),
    },
    {
        "question": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just did "
            "50 minutes of babysitting. How much did she earn?"
        ),
        "answer": (
            "Weng earns 12/60 = $<<12/60=0.2>>0.2 a minute.\n"
            "50 minutes of babysitting = 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"
        ),
    },
    {"question": "What is 2 + 2?", "answer": "4"},
]

EXPECTED_PREFIX = (
    "Solve the following grade-school math problem. Show your reasoning "
    "and put the final answer in \\boxed{}.\n\n"
)


class _Dataset(list):
    """Stand-in for a `datasets.Dataset`: a row list that can `map`."""

    def map(self, function):
        return type(self)(function(record) for record in self)


def _load(module_name: str, relative_path: str):
    path = EXAMPLES / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_rows(relative_path: str, module_name: str, rows):
    loader = _load(module_name, relative_path)
    return loader.load_training_dataset("unused", default_loader=lambda _: _Dataset(rows))


class PromptParityTest(unittest.TestCase):
    """The two example loaders must feed the trainer identical prompts."""

    def test_prompts_and_solutions_match(self):
        opd_rows = _load_rows("math/dataset_loader.py", "opd_loader_for_tests", GSM8K_ROWS)
        sft_rows = _load_rows(
            "opd/sft_baseline_loader.py", "opd_sft_baseline_for_tests", GSM8K_ROWS
        )

        self.assertEqual(len(sft_rows), len(GSM8K_ROWS))
        for opd_row, sft_row in zip(opd_rows, sft_rows, strict=True):
            self.assertEqual(opd_row["prompt"], sft_row["prompt"])
            self.assertEqual(opd_row["solutions"], sft_row["solutions"])

    def test_prompt_text_is_pinned(self):
        # Parity alone would also pass if both copies changed together, so the
        # rendered prompt is anchored to a literal as well.
        rows = _load_rows("opd/sft_baseline_loader.py", "opd_sft_pin_for_tests", GSM8K_ROWS)
        self.assertEqual(
            rows[0]["prompt"],
            EXPECTED_PREFIX + f"Problem: {GSM8K_ROWS[0]['question']}\nSolution:",
        )


class SftBaselineLoaderTest(unittest.TestCase):
    def setUp(self):
        self.loader = _load("opd_sft_baseline_unit_for_tests", "opd/sft_baseline_loader.py")

    def test_keeps_full_rationale_and_final_answer(self):
        rows = self.loader.load_training_dataset(
            "unused", default_loader=lambda _: _Dataset(GSM8K_ROWS)
        )

        self.assertIn("<<48/2=24>>24 clips in May", rows[0]["response"])
        self.assertEqual(rows[0]["solutions"], ["72"])
        # The SFT target is the rationale plus the final line, not the bare
        # answer, so the baseline is trained the same way the student is asked
        # to answer at evaluation time.
        self.assertTrue(rows[0]["response"].startswith(" Natalia sold"))

    def test_bare_answer_row_keeps_answer_as_solution(self):
        rows = self.loader.load_training_dataset(
            "unused", default_loader=lambda _: _Dataset(GSM8K_ROWS)
        )

        self.assertEqual(rows[2]["solutions"], ["4"])
        self.assertEqual(rows[2]["response"], " 4")

    def test_prompt_response_rows_pass_through(self):
        raw = _Dataset([{"prompt": "p", "response": "r"}])

        self.assertIs(
            self.loader.load_training_dataset("unused", default_loader=lambda _: raw), raw
        )

    def test_empty_dataset_short_circuits(self):
        raw = _Dataset()

        rows = self.loader.load_training_dataset("unused", default_loader=lambda _: raw)

        self.assertEqual(rows, [])

    def test_unknown_schema_raises(self):
        with self.assertRaisesRegex(KeyError, "prompt"):
            self.loader.load_training_dataset(
                "unused", default_loader=lambda _: _Dataset([{"problem": "p", "solution": "s"}])
            )
