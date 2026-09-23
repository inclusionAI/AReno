"""Offline SWE-bench conversion contracts, without model or Docker execution."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_dataset import DATASET, REVISION, convert_row, generate
from swe_dataset_loader import load_training_dataset


def task(instance_id="repo__project-1"):
    return {
        "instance_id": instance_id,
        "repo": "repo/project",
        "base_commit": "abc123",
        "version": "1.0",
        "problem_statement": "Fix the parser when its input is empty.",
        "patch": "SECRET_GOLD_PATCH",
        "hints_text": "SECRET_HINTS",
        "test_patch": "PRIVATE_TEST_PATCH",
        "FAIL_TO_PASS": '["test_empty"]',
        "PASS_TO_PASS": '["test_regular"]',
    }


def convert(raw):
    return convert_row(raw, dataset=DATASET, revision=REVISION, split="dev", source_sha256="abc")


def test_gold_and_tests_do_not_leak_into_agent_input():
    row = convert(task())
    assert "SECRET" not in json.dumps(row)
    assert "PRIVATE" not in row["prompt"]
    assert "test_empty" not in row["prompt"]
    assert row["swebench"]["FAIL_TO_PASS"] == ["test_empty"]
    assert row["swebench"]["test_patch"] == "PRIVATE_TEST_PATCH"
    assert "files" not in row
    assert load_training_dataset("unused", default_loader=lambda _: [row]) == [row]


def test_missing_test_targets_rejected():
    raw = task()
    raw["FAIL_TO_PASS"] = "[]"
    with pytest.raises(ValueError, match="FAIL_TO_PASS"):
        convert(raw)


def test_parquet_generation_selects_only_requested_split(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data = tmp_path / "data"
    data.mkdir()
    pq.write_table(pa.Table.from_pylist([task(), task("repo__project-2")]), data / "dev-00000.parquet")
    pq.write_table(pa.Table.from_pylist([task("TEST-ONLY")]), data / "test-00000.parquet")
    output = tmp_path / "out.jsonl"
    kwargs = dict(dataset=DATASET, revision=REVISION, split="dev")
    assert generate(tmp_path, output, **kwargs) == 2
    before = output.read_bytes()
    assert b"TEST-ONLY" not in before
    assert len(json.loads(before.splitlines()[0])["source"]["sha256"]) == 64
    generate(tmp_path, output, **kwargs)
    assert output.read_bytes() == before
    assert generate(tmp_path, output, limit=1, **kwargs) == 1
    with pytest.raises(ValueError, match="positive"):
        generate(tmp_path, output, limit=0, **kwargs)
    with pytest.raises(ValueError, match="no parquet"):
        generate(tmp_path, output, dataset=DATASET, revision=REVISION, split="train")


def test_invalid_input_preserves_existing_output(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    (tmp_path / "data").mkdir()
    raw = task()
    raw["problem_statement"] = ""
    pq.write_table(pa.Table.from_pylist([raw]), tmp_path / "data/dev-0.parquet")
    output = tmp_path / "out.jsonl"
    output.write_text("existing")
    with pytest.raises(ValueError, match="problem_statement"):
        generate(tmp_path, output, dataset=DATASET, revision=REVISION, split="dev")
    assert output.read_text() == "existing"
