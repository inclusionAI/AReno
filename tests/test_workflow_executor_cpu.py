"""CPU tests for the composable local skill-workflow executor.

Covers success paths, invalid input, dependency cycles, missing outputs,
restart-from-failed-step, dry-run, and boundary conditions. No GPU, network,
or external database is required.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Import the executor as a module so we can unit-test internal functions.
_SCRIPT = Path(__file__).resolve().parents[1] / ".agents" / "scripts" / "run_workflow.py"
import importlib.util

_spec = importlib.util.spec_from_file_location("run_workflow", _SCRIPT)
run_workflow_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_workflow_mod)

parse_yaml = run_workflow_mod.parse_yaml
run_workflow = run_workflow_mod.run_workflow
YamlParseError = run_workflow_mod.YamlParseError
WorkflowError = run_workflow_mod.WorkflowError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wf_dir(tmp_path: Path) -> Path:
    """Create a temp directory with echo/echo scripts for workflow tests."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()

    # A script that echoes its args as JSON.
    (scripts / "echo.py").write_text(
        textwrap.dedent(
            """
            #!/usr/bin/env python3
            from __future__ import annotations
            import argparse, json, sys
            parser = argparse.ArgumentParser()
            parser.add_argument("--value", default="default")
            parser.add_argument("--flag", action="store_true")
            args = parser.parse_args()
            out = {"ok": True, "value": args.value, "flag_set": args.flag}
            print(json.dumps(out))
            """
        ),
        encoding="utf-8",
    )

    # A script that always fails (non-zero exit).
    (scripts / "fail.py").write_text(
        textwrap.dedent(
            """
            #!/usr/bin/env python3
            from __future__ import annotations
            import json
            print(json.dumps({"ok": False, "error": "intentional failure"}))
            raise SystemExit(1)
            """
        ),
        encoding="utf-8",
    )

    # A script that produces no JSON (empty stdout).
    (scripts / "empty.py").write_text(
        textwrap.dedent(
            """
            #!/usr/bin/env python3
            print("", end="")
            """
        ),
        encoding="utf-8",
    )

    return tmp_path


def _write_wf(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# YAML parser tests
# ---------------------------------------------------------------------------


class TestParseYaml:
    def test_simple_mapping(self):
        text = "name: foo\nvalue: 42\n"
        result = parse_yaml(text)
        assert result == {"name": "foo", "value": 42}

    def test_nested_mapping(self):
        text = "outer:\n  inner: val\n  num: 3\n"
        result = parse_yaml(text)
        assert result == {"outer": {"inner": "val", "num": 3}}

    def test_sequence_of_scalars(self):
        text = "items:\n  - a\n  - b\n  - c\n"
        result = parse_yaml(text)
        assert result == {"items": ["a", "b", "c"]}

    def test_sequence_of_mappings(self):
        text = textwrap.dedent(
            """
            steps:
              - id: step1
                script: a.py
              - id: step2
                script: b.py
            """
        )
        result = parse_yaml(text)
        assert result["steps"] == [
            {"id": "step1", "script": "a.py"},
            {"id": "step2", "script": "b.py"},
        ]

    def test_types(self):
        text = textwrap.dedent(
            """
            str_val: hello
            int_val: 42
            float_val: 3.14
            bool_true: true
            bool_false: false
            null_val: null
            empty_val:
            quoted: "with spaces"
            """
        )
        result = parse_yaml(text)
        assert result["str_val"] == "hello"
        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["bool_true"] is True
        assert result["bool_false"] is False
        assert result["null_val"] is None
        assert result["empty_val"] is None
        assert result["quoted"] == "with spaces"

    def test_inline_list(self):
        text = "items: [1, 2, 3]\n"
        result = parse_yaml(text)
        assert result["items"] == [1, 2, 3]

    def test_comments_stripped(self):
        text = "name: foo  # this is a comment\nvalue: 1\n# full line comment\n"
        result = parse_yaml(text)
        assert result == {"name": "foo", "value": 1}

    def test_tabs_rejected(self):
        text = "name: foo\n\tvalue: 1\n"
        with pytest.raises(YamlParseError, match="tabs are not allowed"):
            parse_yaml(text)

    def test_empty(self):
        assert parse_yaml("") == {}

    def test_interpolation_kept_as_string(self):
        text = "val: ${steps.foo.bar}\n"
        result = parse_yaml(text)
        assert result["val"] == "${steps.foo.bar}"


# ---------------------------------------------------------------------------
# Workflow validation tests
# ---------------------------------------------------------------------------


class TestWorkflowValidation:
    def test_missing_name(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            steps:
              - id: a
                script: scripts/echo.py
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is False
        assert "missing required key: 'name'" in result["error"]

    def test_missing_steps(self, wf_dir: Path):
        wf = _write_wf(wf_dir / "wf.yaml", "name: test\n")
        result = run_workflow(wf)
        assert result["ok"] is False
        assert "missing required key: 'steps'" in result["error"]

    def test_duplicate_step_id(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                script: scripts/echo.py
              - id: a
                script: scripts/echo.py
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is False
        assert "duplicate step id" in result["error"]

    def test_missing_script(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                depends_on: []
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is False
        assert "lacks a string 'script'" in result["error"]

    def test_dependency_on_unknown_step(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                script: scripts/echo.py
                depends_on: [nonexistent]
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is False
        assert "depends on unknown step" in result["error"]


# ---------------------------------------------------------------------------
# Cycle detection tests
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_simple_cycle(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                script: scripts/echo.py
                depends_on: [b]
              - id: b
                script: scripts/echo.py
                depends_on: [a]
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is False
        assert "cycle" in result["error"].lower()

    def test_self_cycle(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                script: scripts/echo.py
                depends_on: [a]
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is False
        assert "cycle" in result["error"].lower()


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------


class TestExecution:
    def test_success_basic(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            description: A basic success workflow
            steps:
              - id: step1
                script: scripts/echo.py
                inputs:
                  --value: hello
                depends_on: []
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is True
        assert result["steps_executed"] == 1
        assert result["results"][0]["ok"] is True
        assert result["results"][0]["output"]["value"] == "hello"

    def test_output_passing(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: first
                script: scripts/echo.py
                inputs:
                  --value: passed-value
                depends_on: []
              - id: second
                script: scripts/echo.py
                inputs:
                  --value: ${steps.first.value}
                depends_on: [first]
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is True
        assert result["results"][1]["output"]["value"] == "passed-value"

    def test_boolean_flag_passthrough(self, wf_dir: Path):
        """A True value should emit --flag without a value (store_true)."""
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: first
                script: scripts/echo.py
                inputs:
                  --value: x
                depends_on: []
              - id: second
                script: scripts/echo.py
                inputs:
                  --value: y
                  --flag: ${steps.first.ok}
                depends_on: [first]
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is True
        assert result["results"][1]["output"]["flag_set"] is True

    def test_step_failure_stops_pipeline(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: fail_step
                script: scripts/fail.py
                depends_on: []
              - id: after
                script: scripts/echo.py
                depends_on: [fail_step]
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is False
        assert result["results"][0]["ok"] is False
        assert result["results"][1]["ok"] is False
        assert "skipped (prior failure)" in result["results"][1]["error"]

    def test_missing_output_key(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: first
                script: scripts/echo.py
                inputs:
                  --value: x
                depends_on: []
              - id: second
                script: scripts/echo.py
                inputs:
                  --value: ${steps.first.nonexistent_key}
                depends_on: [first]
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is False
        assert result["results"][1]["ok"] is False
        assert "missing key" in result["results"][1]["error"]

    def test_script_not_found(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                script: scripts/nonexistent.py
                depends_on: []
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is False
        assert "script not found" in result["results"][0]["error"]

    def test_empty_stdout(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                script: scripts/empty.py
                depends_on: []
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is False
        assert "no stdout output" in result["results"][0]["error"]


# ---------------------------------------------------------------------------
# Restart-from-failed-step tests
# ---------------------------------------------------------------------------


class TestRestart:
    def test_start_from_skips_earlier(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                script: scripts/echo.py
                inputs:
                  --value: aaa
                depends_on: []
              - id: b
                script: scripts/echo.py
                inputs:
                  --value: bbb
                depends_on: [a]
              - id: c
                script: scripts/echo.py
                inputs:
                  --value: ccc
                depends_on: [b]
            """,
        )
        result = run_workflow(wf, start_from="b")
        assert result["ok"] is True
        # 'a' is re-executed (b depends on it), 'b' and 'c' are executed.
        assert result["results"][0]["ok"] is True
        assert result["results"][0]["status"] == "success"
        assert result["results"][1]["ok"] is True
        assert result["results"][1]["output"]["value"] == "bbb"
        assert result["results"][2]["output"]["value"] == "ccc"

    def test_start_from_resolves_deps(self, wf_dir: Path):
        """--start-from must re-execute dependencies so their outputs are available."""
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: recipe
                script: scripts/echo.py
                inputs:
                  --value: gspo
                depends_on: []
              - id: run
                script: scripts/echo.py
                inputs:
                  --value: ${steps.recipe.value}
                depends_on: [recipe]
              - id: summary
                script: scripts/echo.py
                inputs:
                  --value: ${steps.run.value}
                depends_on: [run]
            """,
        )
        result = run_workflow(wf, start_from="run")
        assert result["ok"] is True
        # recipe is re-executed because run depends on it.
        assert result["results"][0]["ok"] is True
        assert result["results"][0]["output"]["value"] == "gspo"
        # run successfully resolves recipe's output.
        assert result["results"][1]["ok"] is True
        assert result["results"][1]["output"]["value"] == "gspo"
        # summary resolves run's output.
        assert result["results"][2]["ok"] is True
        assert result["results"][2]["output"]["value"] == "gspo"

    def test_start_from_skips_unneeded_steps(self, wf_dir: Path):
        """Steps not needed by the resume point should be skipped."""
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: standalone
                script: scripts/echo.py
                inputs:
                  --value: xxx
                depends_on: []
              - id: main
                script: scripts/echo.py
                inputs:
                  --value: yyy
                depends_on: []
            """,
        )
        result = run_workflow(wf, start_from="main")
        assert result["ok"] is True
        # standalone is not depended upon by main, so it's skipped.
        assert result["results"][0]["status"] == "skipped_restart"
        assert result["results"][1]["ok"] is True
        assert result["results"][1]["output"]["value"] == "yyy"

    def test_start_from_unknown_step(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                script: scripts/echo.py
                depends_on: []
            """,
        )
        result = run_workflow(wf, start_from="nonexistent")
        assert result["ok"] is False
        assert "start-from step not found" in result["error"]
        assert "nonexistent" in result["error"]


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_does_not_execute(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                script: scripts/echo.py
                inputs:
                  --value: dry
                depends_on: []
            """,
        )
        result = run_workflow(wf, dry_run=True)
        assert result["ok"] is True
        assert result["results"][0]["output"]["dry_run"] is True
        assert "--value" in result["results"][0]["output"]["command"]
        assert "dry" in result["results"][0]["output"]["command"]


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_empty_inputs(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                script: scripts/echo.py
                depends_on: []
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is True
        assert result["results"][0]["output"]["value"] == "default"

    def test_none_value_omitted(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                script: scripts/echo.py
                inputs:
                  --value: null
                depends_on: []
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is True
        # --value is null -> omitted -> argparse default "default"
        assert result["results"][0]["output"]["value"] == "default"

    def test_empty_steps_list(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps: []
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is False
        assert "non-empty sequence" in result["error"]

    def test_interpolation_in_larger_string(self, wf_dir: Path):
        """An embedded placeholder (not full-string) should be stringified."""
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: first
                script: scripts/echo.py
                inputs:
                  --value: 42
                depends_on: []
              - id: second
                script: scripts/echo.py
                inputs:
                  --value: "prefix-${steps.first.value}-suffix"
                depends_on: [first]
            """,
        )
        result = run_workflow(wf)
        assert result["ok"] is True
        assert result["results"][1]["output"]["value"] == "prefix-42-suffix"


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_runs_example_workflow(self):
        """The CLI should successfully execute the recipe_to_summary example."""
        root = Path(__file__).resolve().parents[1]
        wf_path = root / "examples" / "workflows" / "recipe_to_summary.yaml"
        process = subprocess.run(
            [sys.executable, str(root / ".agents" / "scripts" / "run_workflow.py"), str(wf_path)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        result = json.loads(process.stdout)
        assert result["ok"] is True
        assert result["steps_executed"] == 3

    def test_cli_runs_serve_example(self):
        """The CLI should successfully execute the serve_to_summary example."""
        root = Path(__file__).resolve().parents[1]
        wf_path = root / "examples" / "workflows" / "serve_to_summary.yaml"
        process = subprocess.run(
            [sys.executable, str(root / ".agents" / "scripts" / "run_workflow.py"), str(wf_path)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        result = json.loads(process.stdout)
        assert result["ok"] is True
        assert result["steps_executed"] == 3

    def test_cli_dry_run(self, wf_dir: Path):
        wf = _write_wf(
            wf_dir / "wf.yaml",
            """
            name: test
            steps:
              - id: a
                script: scripts/echo.py
                inputs:
                  --value: x
                depends_on: []
            """,
        )
        root = Path(__file__).resolve().parents[1]
        process = subprocess.run(
            [
                sys.executable,
                str(root / ".agents" / "scripts" / "run_workflow.py"),
                str(wf),
                "--dry-run",
            ],
            cwd=wf_dir,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        result = json.loads(process.stdout)
        assert result["ok"] is True
        assert result["results"][0]["output"]["dry_run"] is True

    def test_cli_nonexistent_file(self, tmp_path: Path):
        root = Path(__file__).resolve().parents[1]
        process = subprocess.run(
            [
                sys.executable,
                str(root / ".agents" / "scripts" / "run_workflow.py"),
                str(tmp_path / "nonexistent.yaml"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert process.returncode == 1
        result = json.loads(process.stdout)
        assert result["ok"] is False
        assert "not found" in result["error"]