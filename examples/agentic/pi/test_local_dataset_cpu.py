"""Offline contracts for the no-image-build BigCodeBench example."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_loader import load_training_dataset
from generate_local_dataset import REVISION, SPLIT, check_reference, convert_row, generate


def task(task_id="BigCodeBench/7"):
    return {
        "task_id": task_id,
        "complete_prompt": 'import json\nOFFSET = 2\ndef task_func(text):\n    """Read value from JSON and add OFFSET."""\n',
        "instruct_prompt": "Implement the JSON utility with its documented offset.",
        "entry_point": "task_func",
        "libs": "['json']",
        "canonical_solution": '    # REFERENCE_ONLY_MARKER\n    return json.loads(text)["value"] + OFFSET\n',
        "test": (
            "import unittest\nfrom unittest.mock import patch\nclass TestCases(unittest.TestCase):\n"
            "    def test_value(self):\n        self.assertEqual(task_func('{\"value\": 10}'), 12)\n"
            "    def test_module_namespace(self):\n"
            "        with patch('__main__.OFFSET', 5):\n"
            "            self.assertEqual(task_func('{\"value\": 10}'), 15)\n"
        ),
    }


def test_reference_passes_stub_fails_and_answers_stay_out_of_inputs():
    raw = task()
    row = convert_row(raw, source_sha256="digest")
    check_reference(row, raw["complete_prompt"] + raw["canonical_solution"])
    assert "REFERENCE_ONLY_MARKER" not in json.dumps(row)
    assert "TestCases" not in row["prompt"] + json.dumps(row["files"])
    assert row["source"]["revision"] == REVISION
    assert row["source"]["sha256"] == "digest"
    assert load_training_dataset("unused", default_loader=lambda _: [row]) == [row]


def test_non_stdlib_dependency_rejected_even_if_missing_from_metadata():
    raw = task()
    raw["test"] = "import pandas\n" + raw["test"]
    with pytest.raises(ValueError, match="non-stdlib.*pandas"):
        convert_row(raw, source_sha256="digest")


@pytest.mark.parametrize(
    "tests",
    [
        "import unittest\nclass TestCases(unittest.TestCase): pass\n",
        "import unittest\nclass TestCases(unittest.TestCase):\n"
        "    @unittest.skip('fixture')\n    def test_skipped(self): pass\n",
    ],
)
def test_empty_or_skipped_test_suite_cannot_get_positive_reward(tmp_path, tests):
    raw = task()
    raw["test"] = tests
    row = convert_row(raw, source_sha256="digest")
    (tmp_path / "solution.py").write_text(raw["complete_prompt"] + raw["canonical_solution"])
    result = subprocess.run([sys.executable, "-I", "-c", row["verify"]], cwd=tmp_path, capture_output=True, timeout=10)
    assert result.returncode != 0


def test_generation_selection_provenance_and_preserving_output(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data = tmp_path / "data"
    data.mkdir()
    pq.write_table(pa.Table.from_pylist([task("BigCodeBench/19"), task()]), data / f"{SPLIT}-00000.parquet")
    output = tmp_path / "out.jsonl"
    ids = ["BigCodeBench/7", "BigCodeBench/19"]
    assert generate(tmp_path, output, task_ids=ids, validate=True) == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["source"]["task_id"] for row in rows] == ids
    assert all(len(row["source"]["sha256"]) == 64 for row in rows)
    before = output.read_bytes()
    generate(tmp_path, output, task_ids=ids)
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match="missing"):
        generate(tmp_path, output, task_ids=["not-a-task"])
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match="positive"):
        generate(tmp_path, output, task_ids=ids, limit=0)
    assert generate(tmp_path, output, task_ids=ids, limit=1) == 1
