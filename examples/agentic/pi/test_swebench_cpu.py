"""Real-issue packaging, archive safety and immutable regression grading."""

import base64
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_dataset import check_task
from generate_swebench import prepare
from swebench_verify import unpack


def test_real_repair_keeps_broken_source_and_immutable_tests(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "generate_swebench.archive_files",
        lambda *_: {
            "pydicom/__init__.py": b"def answer():\n    return 0\n",
            "tests/test_answer.py": b"from pydicom import answer\ndef test_stable():\n    assert isinstance(answer(), int)\n",
        },
    )
    raw = {
        "repo": "pydicom/pydicom",
        "instance_id": "pydicom__pydicom-123",
        "base_commit": "fixture",
        "split": "dev",
        "problem_statement": "The answer should be 42.",
        "FAIL_TO_PASS": json.dumps(["tests/test_answer.py::test_answer"]),
        "PASS_TO_PASS": json.dumps(["tests/test_answer.py::test_stable"]),
        "test_patch": "--- a/tests/test_answer.py\n+++ b/tests/test_answer.py\n@@ -1,3 +1,5 @@\n from pydicom import answer\n def test_stable():\n     assert isinstance(answer(), int)\n+def test_answer():\n+    assert answer() == 42\n",
        "patch": "--- a/pydicom/__init__.py\n+++ b/pydicom/__init__.py\n@@ -1,2 +1,2 @@\n def answer():\n-    return 0\n+    return 42\n",
    }
    row, evidence = prepare(raw, tmp_path, sys.executable)
    assert evidence["broken"]["returncode"] == 1
    assert evidence["reference"]["returncode"] == 0
    assert row["source"]["synthetic"] is False
    assert "patch" not in row["source"]
    unpack(row["workspace_archive"], tmp_path)
    assert "return 0" in (tmp_path / "pydicom/__init__.py").read_text()
    (tmp_path / "tests/test_answer.py").write_text("def test_answer(): pass\n")
    assert check_task(tmp_path, row["verify"], sys.executable)["returncode"] == 1
    (tmp_path / "pydicom/__init__.py").write_text("def answer(): return 42\n")
    assert check_task(tmp_path, row["verify"], sys.executable)["returncode"] == 0


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", tarfile.REGTYPE),
        ("/escape", tarfile.REGTYPE),
        (".pi/config", tarfile.REGTYPE),
        ("link", tarfile.SYMTYPE),
    ],
)
def test_archive_rejects_unsafe_entries(tmp_path, name, kind):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        entry = tarfile.TarInfo(name)
        entry.type = kind
        entry.linkname = "/tmp"
        archive.addfile(entry, io.BytesIO())
    with pytest.raises(ValueError, match="invalid snapshot"):
        unpack(base64.b64encode(data.getvalue()), tmp_path)


def test_frozen_manifest_has_100_distinct_real_issue_repairs():
    rows = json.loads(Path(__file__).with_name("swebench_100.json").read_text())
    assert len(rows) == len({r["issue_url"] for r in rows}) == len({r["task_id"] for r in rows}) == 100
    for row in rows:
        assert row["synthetic"] is False
        assert row["issue_url"].startswith(f"https://github.com/{row['repo']}/issues/")
        assert len(row["base_commit"]) == 40
        assert row["fail_to_pass"] and row["pass_to_pass"]
        assert not set(row["fail_to_pass"]) & set(row["pass_to_pass"])


def test_local_archive_records_do_not_use_arrow_json_loader(tmp_path):
    from dataset_loader import load_training_dataset

    row = {"prompt": "repair", "files": {}, "verify": "pass", "workspace_archive": "archive"}
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(row) + "\n\n")

    def forbidden_loader(_):
        raise AssertionError("local task records must not go through the Arrow JSON reader")

    assert load_training_dataset(str(path), default_loader=forbidden_loader) == [row]
