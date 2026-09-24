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


def snapshot(tmp_path, rows):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data = tmp_path / "data"
    data.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), data / f"{SPLIT}-00000.parquet")
    return tmp_path


def test_default_scans_full_dataset_and_limit_counts_accepted_tasks(tmp_path):
    unsupported = task("BigCodeBench/1")
    unsupported["libs"] = "['tensorflow']"
    snapshot(tmp_path, [task("BigCodeBench/100"), unsupported, task("BigCodeBench/20")])
    output = tmp_path / "all.jsonl"
    assert generate(tmp_path, output) == 2
    report = json.loads(Path(str(output) + ".report.json").read_text())
    assert report["selected_task_ids"] == ["BigCodeBench/20", "BigCodeBench/100"]
    assert report["source_rows"] == 3
    assert report["excluded"][0]["task_id"] == "BigCodeBench/1"
    assert not report["reference_checked"]
    assert report["not_considered"] == 0
    assert generate(tmp_path, output, limit=1) == 1
    report = json.loads(Path(str(output) + ".report.json").read_text())
    assert report["selected_task_ids"] == ["BigCodeBench/20"]
    assert report["not_considered"] == 1


def test_extended_profile_maps_imports_to_shared_packages():
    raw = task("BigCodeBench/20")
    raw["libs"] = "['PIL', 'sklearn', 'numpy', 'mpl_toolkits']"
    row = convert_row(raw, source_sha256="digest", profile="extended")
    assert row["source"]["required_packages"] == ["Pillow", "matplotlib", "numpy", "scikit-learn"]
    assert "Available third-party modules" in row["prompt"]
    with pytest.raises(ValueError, match="non-stdlib"):
        convert_row(raw, source_sha256="digest")


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess\n",
        "import socket\n",
        "DATA = '/shared/data'\n",
        "DATA = '../outside'\n",
        "import os\nos.system('echo hello')\n",
    ],
)
def test_bulk_selection_rejects_external_runtime_requirements(source):
    raw = task("BigCodeBench/20")
    raw["complete_prompt"] = source + raw["complete_prompt"]
    with pytest.raises(ValueError, match="external|path literal"):
        convert_row(raw, source_sha256="digest", profile="extended")


def test_bulk_validation_filters_failed_references_but_explicit_selection_is_strict(tmp_path):
    bad = task("BigCodeBench/1")
    bad["canonical_solution"] = "    return -1\n"
    snapshot(tmp_path, [bad, task("BigCodeBench/20")])
    output = tmp_path / "all.jsonl"
    assert generate(tmp_path, output, validate=True, limit=1) == 1
    report = json.loads(Path(str(output) + ".report.json").read_text())
    assert report["reference_checked"]
    assert report["excluded"][0]["stage"] == "reference_check"
    before = output.read_bytes()
    with pytest.raises(ValueError, match="requested tasks failed"):
        generate(tmp_path, output, task_ids=["BigCodeBench/1"], validate=True)
    assert output.read_bytes() == before


def test_no_eligible_tasks_preserves_output_and_writes_report(tmp_path):
    bad = task("BigCodeBench/1")
    bad["libs"] = "['tensorflow']"
    snapshot(tmp_path, [bad])
    output = tmp_path / "all.jsonl"
    output.write_text("existing")
    with pytest.raises(ValueError, match="no eligible tasks"):
        generate(tmp_path, output)
    assert output.read_text() == "existing"
    assert json.loads(Path(str(output) + ".report.json").read_text())["accepted"] == 0
    with pytest.raises(ValueError, match="must differ"):
        generate(tmp_path, output, report_path=output)


def test_reference_timeout_is_bounded():
    raw = task()
    row = convert_row(raw, source_sha256="digest")
    row["verify_timeout"] = 0.1
    with pytest.raises(subprocess.TimeoutExpired):
        check_reference(row, "while True: pass\n")
