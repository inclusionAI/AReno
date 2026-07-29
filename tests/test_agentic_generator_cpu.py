"""CPU tests for the local agentic-project generator skill (#279).

Covers success, invalid input, and one boundary/failure path, plus a
deterministic smoke run of the generated fixed episode. No GPU, no network.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / ".agents/skills/areno-build-agentic-workflow/scripts/generate_agentic_project.py"

EXPECTED_FILES = [
    "game.py",
    "dataset_generator.py",
    "dataset_loader.py",
    "tool_defs.py",
    "run_agent.py",
    "reward.py",
    "run_episode.py",
    "README.md",
]


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GEN), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_generator_creates_valid_scaffold(tmp_path: Path) -> None:
    """Success path: all expected files exist, compile, and structured output is emitted."""
    out = tmp_path / "proj"
    result = _run(["--name", "my-grid", "--out", str(out)], tmp_path)

    assert result.returncode == 0, result.stderr
    for name in EXPECTED_FILES:
        assert (out / name).is_file(), f"missing generated file: {name}"

    # Every generated Python file must compile.
    for name in EXPECTED_FILES:
        if name.endswith(".py"):
            compile((out / name).read_text(encoding="utf-8"), str(out / name), "exec")

    # Structured JSON summary on the last stdout line.
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["ok"] is True
    assert summary["name"] == "my-grid"
    assert summary["episode_cmd"].endswith("run_episode.py")


def test_generator_rejects_invalid_name(tmp_path: Path) -> None:
    """Invalid input: a name that violates the lowercase-hyphen contract fails fast."""
    result = _run(["--name", "Bad_Name", "--out", str(tmp_path / "p")], tmp_path)
    assert result.returncode == 1
    assert "name" in result.stderr.lower()


def test_generator_refuses_nonempty_without_force(tmp_path: Path) -> None:
    """Boundary/failure path: refuse to overwrite existing user edits without --force."""
    out = tmp_path / "existing"
    out.mkdir()
    (out / "user_edit.py").write_text("x = 1\n", encoding="utf-8")

    result = _run(["--name", "my-grid", "--out", str(out)], tmp_path)
    assert result.returncode == 1
    assert "force" in result.stderr.lower()
    # Existing user edit must be preserved.
    assert (out / "user_edit.py").read_text(encoding="utf-8") == "x = 1\n"
    # No scaffold file should have been written.
    assert not (out / "game.py").exists()


def test_generator_force_overwrites(tmp_path: Path) -> None:
    """--force overwrites a non-empty directory and regenerates the scaffold."""
    out = tmp_path / "existing"
    out.mkdir()
    (out / "stale.txt").write_text("old", encoding="utf-8")

    result = _run(["--name", "my-grid", "--out", str(out), "--force"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert (out / "game.py").is_file()


def test_generated_episode_runs(tmp_path: Path) -> None:
    """Smoke: the generated no-model episode runs and prints observable reward output."""
    out = tmp_path / "proj"
    _run(["--name", "my-grid", "--out", str(out)], tmp_path)

    result = subprocess.run(
        [sys.executable, str(out / "run_episode.py")],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "reset obs=" in result.stdout
    assert "episode_total_reward=" in result.stdout


def test_generated_reward_fn_scores(tmp_path: Path) -> None:
    """Integration: the generated reward_fn scores legal, illegal, and missing tool calls."""
    out = tmp_path / "proj"
    _run(["--name", "my-grid", "--out", str(out)], tmp_path)

    spec = importlib.util.spec_from_file_location("generated_reward", out / "reward.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Record:
        def __init__(self, tool_calls, source_record):
            self.tool_calls = tool_calls
            self.source_record = source_record

    legal = _Record([{"name": "act", "arguments": {"action": 1}}], {"start": 0})
    illegal = _Record([{"name": "act", "arguments": {"action": 9}}], {"start": 0})
    missing = _Record([{"name": "nope", "arguments": {}}], {"start": 0})

    assert isinstance(mod.reward_fn(legal), float)
    assert mod.reward_fn(illegal) == -1.0
    assert mod.reward_fn(missing) == -1.0


def test_generator_is_deterministic(tmp_path: Path) -> None:
    """Snapshot: regenerating with the same seed reproduces identical output."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _run(["--name", "my-grid", "--out", str(out_a)], tmp_path)
    _run(["--name", "my-grid", "--out", str(out_b)], tmp_path)

    for name in EXPECTED_FILES:
        if name.endswith(".py") or name.endswith(".md"):
            assert (out_a / name).read_text(encoding="utf-8") == (out_b / name).read_text(
                encoding="utf-8"
            ), f"non-deterministic output for {name}"
