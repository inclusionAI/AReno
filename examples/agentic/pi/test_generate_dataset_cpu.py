"""Repository repair grading checks, without downloads or model inference."""

import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_dataset import REPO, check_task, convert_row, select_candidates, write_snapshot
from swesmith_verify import verify


def fixture_repo():
    return {
        "sqlparse/__init__.py": b"from .utils import answer\n",
        "sqlparse/utils.py": b"def answer():\n    return 42\n",
        "tests/test_answer.py": (
            b"from sqlparse import answer\ndef test_answer():\n    assert answer() == 42\n"
            b"def test_stable():\n    assert isinstance(answer(), int)\n"
        ),
        "tests/fixture.bin": b"\xff\x80",
    }


def raw_task():
    return {
        "repo": REPO,
        "instance_id": "fixture.func_basic__onefile",
        "problem_statement": "Fix the answer API.",
        "FAIL_TO_PASS": ["tests/test_answer.py::test_answer"],
        "PASS_TO_PASS": ["tests/test_answer.py::test_stable"],
        "patch": """diff --git a/sqlparse/utils.py b/sqlparse/utils.py
--- a/sqlparse/utils.py
+++ b/sqlparse/utils.py
@@ -1,2 +1,2 @@
 def answer():
-    return 42
+    return 0
""",
    }


def test_generated_repo_has_bug_without_gold_patch_and_checks_reference(tmp_path):
    row, evidence = convert_row(raw_task(), fixture_repo(), python=sys.executable, digest="fixture")
    assert evidence["reference"]["returncode"] == 0
    assert evidence["broken"]["tests"]["tests/test_answer.py::test_answer"] == "failed"
    assert "return 0" in row["files"]["sqlparse/utils.py"]
    assert "patch" not in row["source"]
    assert "return 42" not in row["verify"]
    assert base64.b64decode(row["binary_files"]["tests/fixture.bin"]) == b"\xff\x80"
    files = {name: value.encode() for name, value in row["files"].items()}
    files.update({name: base64.b64decode(value) for name, value in row["binary_files"].items()})
    write_snapshot(tmp_path, files)
    # Modifying visible tests cannot convert an incorrect implementation to a pass.
    (tmp_path / "tests/test_answer.py").write_text("def test_answer(): pass\n")
    assert check_task(tmp_path, row["verify"], sys.executable)["returncode"] == 1
    (tmp_path / "sqlparse/utils.py").write_text("def answer(): return 42\n")
    assert check_task(tmp_path, row["verify"], sys.executable)["returncode"] == 0


def test_grader_rejects_skipped_tests_and_symlink_package(tmp_path):
    files = fixture_repo()
    write_snapshot(tmp_path, files)
    protected = {
        "tests/test_skip.py": base64.b64encode(b"import pytest\n@pytest.mark.skip\ndef test_skip(): pass\n").decode()
    }
    assert not verify(tmp_path, protected, ["tests/test_skip.py::test_skip"])
    (tmp_path / "sqlparse/link.py").symlink_to(tmp_path / "tests/test_answer.py")
    with pytest.raises(ValueError, match="symlinks"):
        verify(tmp_path, protected, ["tests/test_skip.py::test_skip"])


def test_generator_rejects_bad_regression_metadata():
    raw = raw_task()
    raw["PASS_TO_PASS"] = []
    with pytest.raises(ValueError, match="FAIL_TO_PASS and PASS_TO_PASS"):
        convert_row(raw, fixture_repo(), python=sys.executable, digest="fixture")
    raw = raw_task()
    raw["patch"] = raw["patch"].replace("b/sqlparse/utils.py", "b/tests/test_answer.py")
    with pytest.raises(ValueError, match="package-only"):
        convert_row(raw, fixture_repo(), python=sys.executable, digest="fixture")


def test_focused_repairs_keep_real_regression_grading(tmp_path):
    raw = raw_task()
    raw["instance_id"] = "fixture.func_basic__onefile"
    combined = dict(raw, instance_id="fixture.combine_module__twofiles", patch=raw["patch"] * 2)
    assert select_candidates([combined, raw], 0) == [raw]
    row, evidence = convert_row(raw, fixture_repo(), python=sys.executable, digest="fixture")
    assert row["source"]["difficulty"] == "focused"
    assert "tests/test_answer.py::test_answer" in row["prompt"]
    assert evidence["reference"]["returncode"] == 0
    assert evidence["broken"]["returncode"] == 1
    assert evidence["changed_files"] == ["sqlparse/utils.py"]
    write_snapshot(tmp_path, {name: value.encode() for name, value in row["files"].items()})
    assert check_task(tmp_path, row["verify"], sys.executable)["returncode"] == 1
    (tmp_path / "sqlparse/utils.py").write_text("def answer(): return 42\n")
    assert check_task(tmp_path, row["verify"], sys.executable)["returncode"] == 0
