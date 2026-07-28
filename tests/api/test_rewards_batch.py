"""CPU tests for batched reward-hook execution (Issue #225).

These tests run without GPU, without external services, and without real model
rollouts. They cover:

- :func:`load_reward` discovering reward_fn, reward_batch, or both
- :func:`call_reward` batch path, scalar path, and fallback paths
- Cardinality validation pointing at the first mismatched index
- Error isolation in the scalar path (one bad sample surfaces its index)
- Deterministic agreement between batch and scalar paths
- CLI ``areno reward inspect`` with --json and human-readable output
- Default behavior (reward_use_batch=False) keeps the scalar path
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from areno.api.rewards import (
    RewardRecord,
    call_reward,
    load_reward,
    load_reward_fn,
)
from areno.cli.reward import reward_command


def _write_reward_module(tmp_path: Path, body: str) -> Path:
    """Write a Python reward module to tmp_path and return its path."""

    path = tmp_path / "reward_mod.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _make_records(n: int) -> list[RewardRecord]:
    return [RewardRecord(prompt=f"p{i}", completion=str(i), answer=int(i)) for i in range(n)]


# --- TestLoadReward --------------------------------------------------------


class TestLoadReward:
    def test_loads_reward_fn_only(self, tmp_path):
        path = _write_reward_module(
            tmp_path,
            """
            def reward_fn(record):
                return float(record.answer)
        """,
        )
        bundle = load_reward(str(path))
        assert bundle.reward_fn is not None
        assert bundle.reward_batch is None
        assert bundle.source_path.endswith("reward_mod.py")

    def test_loads_reward_batch_only(self, tmp_path):
        path = _write_reward_module(
            tmp_path,
            """
            def reward_batch(records):
                return [float(r.answer) for r in records]
        """,
        )
        bundle = load_reward(str(path))
        assert bundle.reward_fn is None
        assert bundle.reward_batch is not None

    def test_loads_both_hooks(self, tmp_path):
        path = _write_reward_module(
            tmp_path,
            """
            def reward_fn(record):
                return float(record.answer)
            def reward_batch(records):
                return [float(r.answer) for r in records]
        """,
        )
        bundle = load_reward(str(path))
        assert bundle.reward_fn is not None
        assert bundle.reward_batch is not None

    def test_raises_when_no_hook_defined(self, tmp_path):
        path = _write_reward_module(
            tmp_path,
            """
            # intentionally empty
        """,
        )
        with pytest.raises(ValueError, match="must define callable"):
            load_reward(str(path))


# --- TestCallReward --------------------------------------------------------


class TestCallReward:
    def test_batch_path_used_when_preferred(self, tmp_path):
        path = _write_reward_module(
            tmp_path,
            """
            def reward_fn(record):
                return float(record.answer)
            def reward_batch(records):
                return [float(r.answer) + 100 for r in records]
        """,
        )
        bundle = load_reward(str(path))
        records = _make_records(3)
        scores, stats = call_reward(bundle, records, prefer_batch=True)
        assert scores == [100.0, 101.0, 102.0]
        assert stats.path == "batch"
        assert stats.count == 3
        assert stats.error is None
        assert stats.wall_time_s >= 0.0
        assert stats.per_example_time_s > 0.0

    def test_scalar_path_used_when_preferred_false(self, tmp_path):
        path = _write_reward_module(
            tmp_path,
            """
            def reward_fn(record):
                return float(record.answer)
            def reward_batch(records):
                return [float(r.answer) + 100 for r in records]
        """,
        )
        bundle = load_reward(str(path))
        records = _make_records(3)
        scores, stats = call_reward(bundle, records, prefer_batch=False)
        assert scores == [0.0, 1.0, 2.0]
        assert stats.path == "scalar"

    def test_scalar_path_falls_back_when_only_batch_defined(self, tmp_path):
        path = _write_reward_module(
            tmp_path,
            """
            def reward_batch(records):
                return [float(r.answer) for r in records]
        """,
        )
        bundle = load_reward(str(path))
        records = _make_records(2)
        scores, stats = call_reward(bundle, records, prefer_batch=False)
        assert scores == [0.0, 1.0]
        assert stats.path == "scalar"

    def test_batch_scalar_agree(self, tmp_path):
        """Acceptance criterion: deterministic hook proves both paths agree."""

        path = _write_reward_module(
            tmp_path,
            """
            def reward_fn(record):
                return float(record.answer) * 2
            def reward_batch(records):
                return [float(r.answer) * 2 for r in records]
        """,
        )
        bundle = load_reward(str(path))
        records = _make_records(5)
        batch_scores, batch_stats = call_reward(bundle, records, prefer_batch=True)
        scalar_scores, scalar_stats = call_reward(bundle, records, prefer_batch=False)
        assert batch_scores == scalar_scores == [0.0, 2.0, 4.0, 6.0, 8.0]
        assert batch_stats.path == "batch"
        assert scalar_stats.path == "scalar"

    def test_cardinality_mismatch_reports_index(self, tmp_path):
        """Acceptance criterion: diagnose exact bad batch on length mismatch."""

        path = _write_reward_module(
            tmp_path,
            """
            def reward_batch(records):
                # Drop the last score to trigger a cardinality mismatch.
                return [float(r.answer) for r in records[:-1]]
        """,
        )
        bundle = load_reward(str(path))
        records = _make_records(4)
        with pytest.raises(ValueError) as excinfo:
            call_reward(bundle, records, prefer_batch=True)
        msg = str(excinfo.value)
        # Error message must include the count and the first mismatched index,
        # but must NOT include any prompt or completion content.
        assert "3" in msg  # got 3, expected 4
        assert "4" in msg  # expected 4
        assert "first mismatched index" in msg
        assert "p0" not in msg and "p1" not in msg

    def test_scalar_path_isolates_bad_sample(self, tmp_path):
        """Acceptance criterion: scalar path surfaces the failing index."""

        path = _write_reward_module(
            tmp_path,
            """
            def reward_fn(record):
                if record.answer == 2:
                    raise RuntimeError("bad sample")
                return float(record.answer)
        """,
        )
        bundle = load_reward(str(path))
        records = _make_records(4)
        with pytest.raises(RuntimeError) as excinfo:
            call_reward(bundle, records, prefer_batch=False)
        assert "index 2" in str(excinfo.value)

    def test_empty_records_returns_empty(self, tmp_path):
        path = _write_reward_module(
            tmp_path,
            """
            def reward_fn(record):
                return 0.0
        """,
        )
        bundle = load_reward(str(path))
        scores, stats = call_reward(bundle, [], prefer_batch=True)
        assert scores == []
        assert stats.count == 0
        assert stats.path == "scalar"

    def test_order_preserved(self, tmp_path):
        path = _write_reward_module(
            tmp_path,
            """
            def reward_batch(records):
                return [float(r.answer) for r in records]
        """,
        )
        bundle = load_reward(str(path))
        records = _make_records(5)
        scores, _ = call_reward(bundle, records, prefer_batch=True)
        assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]


# --- TestBackwardCompat ----------------------------------------------------


class TestBackwardCompat:
    def test_load_reward_fn_still_works(self, tmp_path):
        path = _write_reward_module(
            tmp_path,
            """
            def reward_fn(record):
                return float(record.answer)
        """,
        )
        fn = load_reward_fn(str(path))
        record = RewardRecord(prompt="p", completion="c", answer=42)
        assert fn(record) == 42.0

    def test_default_prefer_batch_true_but_no_batch_hook(self, tmp_path):
        """Default behavior unchanged when only reward_fn is defined."""

        path = _write_reward_module(
            tmp_path,
            """
            def reward_fn(record):
                return float(record.answer)
        """,
        )
        bundle = load_reward(str(path))
        records = _make_records(3)
        scores, stats = call_reward(bundle, records, prefer_batch=True)
        # Falls back to scalar because reward_batch is None.
        assert scores == [0.0, 1.0, 2.0]
        assert stats.path == "scalar"


# --- TestCli ---------------------------------------------------------------


class TestCli:
    def _write_fixtures(self, tmp_path: Path) -> Path:
        fixtures = tmp_path / "fixtures.jsonl"
        fixtures.write_text(
            "\n".join(json.dumps({"prompt": f"p{i}", "completion": str(i), "answer": i}) for i in range(3)) + "\n",
            encoding="utf-8",
        )
        return fixtures

    def test_inspect_human_readable(self, tmp_path):
        reward_path = _write_reward_module(
            tmp_path,
            """
            def reward_fn(record):
                return float(record.answer)
            def reward_batch(records):
                return [float(r.answer) for r in records]
        """,
        )
        fixtures_path = self._write_fixtures(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            reward_command,
            ["inspect", "--path", str(reward_path), "--fixtures", str(fixtures_path)],
        )
        assert result.exit_code == 0, result.output
        assert "reward_fn=yes" in result.output
        assert "reward_batch=yes" in result.output
        assert "score=0.0000" in result.output
        assert "score=2.0000" in result.output
        assert "path=batch" in result.output  # prefer_batch defaults to True

    def test_inspect_json_output(self, tmp_path):
        reward_path = _write_reward_module(
            tmp_path,
            """
            def reward_fn(record):
                return float(record.answer)
            def reward_batch(records):
                return [float(r.answer) for r in records]
        """,
        )
        fixtures_path = self._write_fixtures(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            reward_command,
            ["inspect", "--path", str(reward_path), "--fixtures", str(fixtures_path), "--json"],
        )
        assert result.exit_code == 0, result.output
        report = json.loads(result.output)
        assert report["has_reward_fn"] is True
        assert report["has_reward_batch"] is True
        assert report["prefer_batch"] is True
        assert report["scores"] == [0.0, 1.0, 2.0]
        assert len(report["stats"]) == 1
        assert report["stats"][0]["path"] == "batch"
        assert report["stats"][0]["count"] == 3

    def test_inspect_scalar_only_flag(self, tmp_path):
        reward_path = _write_reward_module(
            tmp_path,
            """
            def reward_fn(record):
                return float(record.answer)
            def reward_batch(records):
                return [float(r.answer) + 100 for r in records]
        """,
        )
        fixtures_path = self._write_fixtures(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            reward_command,
            ["inspect", "--path", str(reward_path), "--fixtures", str(fixtures_path), "--scalar-only", "--json"],
        )
        assert result.exit_code == 0, result.output
        report = json.loads(result.output)
        assert report["prefer_batch"] is False
        assert report["scores"] == [0.0, 1.0, 2.0]
        assert report["stats"][0]["path"] == "scalar"
