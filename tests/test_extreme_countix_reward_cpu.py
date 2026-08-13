from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_reward_module():
    path = Path(__file__).parents[1] / "examples" / "agentic" / "extreme_countix_av" / "reward.py"
    spec = importlib.util.spec_from_file_location("extreme_countix_reward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extreme_countix_reward_ignores_action_class():
    reward = _load_reward_module()
    source = {"action_class": "push ups", "repetition_count": 10}
    record = SimpleNamespace(
        source_record=source,
        completion="",
        tool_calls=[
            {
                "name": "report_repetitions",
                "arguments": {"action_class": "completely unrelated", "repetition_count": 10},
            }
        ],
    )

    assert reward.reward_fn(record) == 1.0


def test_extreme_countix_reward_tracks_count_error_only():
    reward = _load_reward_module()
    source = {"action_class": "push ups", "repetition_count": 10}
    record = SimpleNamespace(
        source_record=source,
        completion="",
        tool_calls=[
            {
                "name": "report_repetitions",
                "arguments": {"action_class": "push ups", "repetition_count": 5},
            }
        ],
    )

    assert reward.reward_fn(record) == 0.5
