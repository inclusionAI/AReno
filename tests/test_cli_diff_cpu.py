"""CPU tests for ``areno diff``."""

from __future__ import annotations

import json
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from click.testing import CliRunner

from areno.cli.dashboard_registry import read_run_config
from areno.cli.main import main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RUN_A_ID = "abc123def456"
RUN_B_ID = "def456abc123"
PREFIX = "areno.cli.diff."


def _registry_entries():
    return [
        {
            "id": RUN_A_ID, "kind": "train", "name": "gspo Qwen/Qwen3-0.6B",
            "pid": 12345, "metrics_dir": "/tmp/areno/tfevent", "cwd": "/tmp",
            "created_at": 1753700000.0, "updated_at": 1753702730.0,
        },
        {
            "id": RUN_B_ID, "kind": "train", "name": "gspo Qwen/Qwen3-1.5B",
            "pid": 12346, "metrics_dir": "/tmp/areno/tfevent", "cwd": "/tmp",
            "created_at": 1753690000.0, "updated_at": 1753694320.0,
        },
    ]


def _run_config(ckpt: str, algo: str = "gspo") -> dict:
    return {
        "kind": "train", "pid": 12345, "summary_text": "...",
        "settings": {
            "sections": [
                {"title": "Basic", "items": [
                    {"key": "algo", "value": algo},
                    {"key": "ckpt", "value": ckpt},
                    {"key": "dataset_path", "value": "gsm8k:main"},
                ]},
                {"title": "Runtime", "items": [
                    {"key": "tp_size", "value": 4},
                    {"key": "eager_decode", "value": False},
                ]},
                {"title": "Optimizer", "items": [
                    {"key": "optimizer_lr", "value": 1e-6},
                ]},
            ]
        },
    }


def _dashboard_state(*, step: int = 500, status: str = "succeeded"):
    return {"pid": 12345, "stage": "", "status": status, "step": step, "updated_at": 1753702800.0}


def _tb_scalars(
    loss: float = 0.342, reward: float = 0.68, accuracy: float = 0.72,
    rollout_time: float = 12.3, train_time: float = 8.8,
):
    return {
        "train/loss": [(500, loss)],
        "rollout/rewards_mean": [(500, reward)],
        "rollout/accuracy": [(500, accuracy)],
        "rollout/response_len_mean": [(500, 256.0)],
        "rollout/num_sequences": [(500, 8.0)],
        "time/rollout": [(500, rollout_time)],
        "time/train": [(500, train_time)],
    }


def _by_pid(a_value, b_value):
    """Return a side_effect function returning a_value for pid 12345, b_value for 12346."""

    def _inner(metrics_dir, pid):
        return a_value if str(pid) == "12345" else b_value

    return _inner


def _resolve_patches(**overrides):
    """Build patch objects, resolving short names to full dotted paths."""
    base = {
        f"{PREFIX}read_registry": _registry_entries(),
        f"{PREFIX}pid_is_running": False,
        f"{PREFIX}read_dashboard_state": _by_pid(_dashboard_state(), _dashboard_state()),
        f"{PREFIX}read_run_config": _by_pid(_run_config("Qwen/Qwen3-0.6B"), _run_config("Qwen/Qwen3-1.5B")),
        f"{PREFIX}read_tensorboard_scalars": _by_pid(_tb_scalars(), _tb_scalars(loss=0.287, reward=0.73, accuracy=0.78, rollout_time=18.7, train_time=12.1)),
    }
    for key, value in overrides.items():
        full = key if "." in key else f"{PREFIX}{key}"
        base[full] = value
    result = []
    for path, value in base.items():
        if callable(value) and not isinstance(value, (dict, list, bool, int, float, str)):
            result.append(patch(path, side_effect=value))
        else:
            result.append(patch(path, return_value=value))
    return result


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class DiffUtilsTest(unittest.TestCase):
    def test_read_run_config_missing_file(self):
        self.assertIsNone(read_run_config("/nonexistent/path", 12345))


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class CliDiffTest(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def _invoke(self, *args: str):
        return self.runner.invoke(main, ["diff", *args])

    # --- Basic success paths ---

    def test_diff_two_runs_shows_summary(self):
        with ExitStack() as stack:
            for p in _resolve_patches():
                stack.enter_context(p)
            result = self._invoke(RUN_A_ID, RUN_B_ID)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Summary", result.output)
        self.assertIn("Qwen/Qwen3-0.6B", result.output)
        self.assertIn("Qwen/Qwen3-1.5B", result.output)

    def test_diff_shows_config_changes(self):
        with ExitStack() as stack:
            for p in _resolve_patches():
                stack.enter_context(p)
            result = self._invoke(RUN_A_ID, RUN_B_ID)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Config changes", result.output)
        self.assertIn("ckpt", result.output)

    def test_diff_shows_final_metrics(self):
        with ExitStack() as stack:
            for p in _resolve_patches():
                stack.enter_context(p)
            result = self._invoke(RUN_A_ID, RUN_B_ID)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Final metrics", result.output)
        self.assertIn("0.342", result.output)
        self.assertIn("0.287", result.output)

    def test_diff_shows_phase_timing(self):
        with ExitStack() as stack:
            for p in _resolve_patches():
                stack.enter_context(p)
            result = self._invoke(RUN_A_ID, RUN_B_ID)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Phase timing", result.output)
        self.assertIn("12.3s", result.output)
        self.assertIn("18.7s", result.output)

    def test_diff_json_output(self):
        with ExitStack() as stack:
            for p in _resolve_patches():
                stack.enter_context(p)
            result = self._invoke(RUN_A_ID, RUN_B_ID, "--json")
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("a", data)
        self.assertIn("b", data)
        self.assertIn("config_diff", data)
        self.assertIn("metrics", data)

    # --- Missing / partial data ---

    def test_diff_missing_run_id(self):
        with ExitStack() as stack:
            for p in _resolve_patches():
                stack.enter_context(p)
            result = self._invoke("nonexistent", RUN_B_ID)
        self.assertEqual(result.exit_code, 2)
        self.assertIn("not found", result.output)

    def test_diff_missing_metrics_shows_dash(self):
        empty_tb = {}
        patches = _resolve_patches(**{
            f"{PREFIX}read_tensorboard_scalars": _by_pid(_tb_scalars(), empty_tb),
        })
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = self._invoke(RUN_A_ID, RUN_B_ID)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("-", result.output)

    def test_diff_missing_config(self):
        patches = _resolve_patches(**{
            f"{PREFIX}read_run_config": _by_pid(_run_config("Qwen/Qwen3-0.6B"), None),
        })
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = self._invoke(RUN_A_ID, RUN_B_ID)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Summary", result.output)

    def test_diff_no_common_metrics_shows_empty(self):
        tb_empty = {"some/obscure/tag": [(1, 0.5)]}
        patches = _resolve_patches(**{
            f"{PREFIX}read_tensorboard_scalars": _by_pid(tb_empty, tb_empty),
        })
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = self._invoke(RUN_A_ID, RUN_B_ID)
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("Final metrics", result.output)

    # --- Active runs ---

    def test_diff_active_run_shows_label(self):
        patches = _resolve_patches(**{f"{PREFIX}pid_is_running": True})
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = self._invoke(RUN_A_ID, RUN_B_ID)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("(active)", result.output)

    # --- Different algorithms ---

    def test_diff_different_algorithms_shows_note(self):
        patches = _resolve_patches(**{
            f"{PREFIX}read_run_config": _by_pid(
                _run_config("Qwen/Qwen3-0.6B", algo="gspo"),
                _run_config("Qwen/Qwen3-1.5B", algo="grpo"),
            ),
        })
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = self._invoke(RUN_A_ID, RUN_B_ID)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Incomparable", result.output)
        self.assertIn("gspo", result.output)
        self.assertIn("grpo", result.output)

    def test_diff_unequal_steps_shows_note(self):
        patches = _resolve_patches(**{
            f"{PREFIX}read_dashboard_state": _by_pid(
                _dashboard_state(step=500),
                _dashboard_state(step=312),
            ),
        })
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = self._invoke(RUN_A_ID, RUN_B_ID)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Incomparable", result.output)
        self.assertIn("Unequal step counts", result.output)

    # --- Help ---

    def test_diff_help(self):
        result = self._invoke("--help")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("RUN_A", result.output)
        self.assertIn("RUN_B", result.output)
        self.assertIn("--json", result.output)