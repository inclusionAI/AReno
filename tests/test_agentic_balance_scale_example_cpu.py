"""CPU tests for the odd-ball balance-scale agentic example.

Covers: core game logic, malformed input, boundary values, dataset
generation, reward scoring, agent tool execution, and budget enforcement.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "balance_scale"

# run_agent.py imports areno.api.agentic which transitively requires torch.
# Skip those tests when torch is unavailable (matches the behaviour of the
# existing shopping example tests in environments without GPU dependencies).
_TORCH_AVAILABLE = True
try:
    import torch  # noqa: F401
except ImportError:
    _TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")


class _SimpleItem:
    """Minimal stand-in for AgentItem used by _run_puzzle_loop."""

    def __init__(self, record: dict, prompt: str = "placeholder prompt"):
        self.record = record
        self.prompt = prompt


def asyncio_run(coro):
    """Run an async coroutine, compatible with Python 3.9+."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        # Already in a running loop (unlikely in pytest, but safe).
        return loop.run_until_complete(coro)
    return asyncio.run(coro)


def _load_module(name: str):
    """Load a module from the balance_scale example directory.

    The ``game`` module is inserted into ``sys.path`` so that sibling modules
    can ``import game`` at runtime. We clean up after loading to avoid
    polluting other tests.
    """

    path = EXAMPLE_DIR / f"{name}.py"
    mod_name = f"agentic_balance_scale_{name}_for_tests"
    previous_game = sys.modules.pop("game", None)
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("game", None)
        sys.modules.pop(mod_name, None)
        if previous_game is not None:
            sys.modules["game"] = previous_game


# ---------------------------------------------------------------------------
# Core game logic
# ---------------------------------------------------------------------------


def test_make_ball_set_creates_valid_instance():
    game = _load_module("game")

    bs = game.make_ball_set(num_balls=12, odd_ball_index=5, direction="heavier", max_weighings=3)

    assert bs.num_balls == 12
    assert bs.odd_ball_index == 5
    assert bs.direction == "heavier"
    assert bs.max_weighings == 3


def test_make_ball_set_seeded_is_deterministic():
    game = _load_module("game")

    bs1 = game.make_ball_set(num_balls=12, seed=42)
    bs2 = game.make_ball_set(num_balls=12, seed=42)

    assert bs1.odd_ball_index == bs2.odd_ball_index
    assert bs1.direction == bs2.direction


def test_ball_set_rejects_invalid_params():
    game = _load_module("game")

    # Too few balls
    try:
        game.BallSet(num_balls=1, odd_ball_index=0, direction="heavier", max_weighings=3)
        assert False, "should have raised"
    except ValueError:
        pass

    # odd_ball_index out of range
    try:
        game.BallSet(num_balls=12, odd_ball_index=12, direction="heavier", max_weighings=3)
        assert False, "should have raised"
    except ValueError:
        pass

    # Invalid direction
    try:
        game.BallSet(num_balls=12, odd_ball_index=0, direction="wrong", max_weighings=3)
        assert False, "should have raised"
    except ValueError:
        pass

    # Zero weighings
    try:
        game.BallSet(num_balls=12, odd_ball_index=0, direction="heavier", max_weighings=0)
        assert False, "should have raised"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Weighing tests — success path
# ---------------------------------------------------------------------------


def test_weigh_returns_balanced_when_odd_not_weighed():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=5, direction="heavier", max_weighings=3)
    result = game.weigh(bs, left=[0, 1, 2], right=[3, 4, 6], weighings_used=0)

    assert result == "balanced"


def test_weigh_returns_left_heavy_when_heavier_ball_on_left():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=0, direction="heavier", max_weighings=3)
    result = game.weigh(bs, left=[0, 1, 2], right=[3, 4, 5], weighings_used=0)

    assert result == "left_heavy"


def test_weigh_returns_right_heavy_when_lighter_ball_on_right():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=3, direction="lighter", max_weighings=3)
    result = game.weigh(bs, left=[0, 1, 2], right=[3, 4, 5], weighings_used=0)

    # Ball 3 is lighter and is on the right side → left side is heavier
    assert result == "left_heavy"


def test_weigh_returns_right_heavy_when_heavier_ball_on_right():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=5, direction="heavier", max_weighings=3)
    result = game.weigh(bs, left=[0, 1, 2], right=[3, 4, 5], weighings_used=0)

    assert result == "right_heavy"


# ---------------------------------------------------------------------------
# Weighing tests — invalid input and boundary
# ---------------------------------------------------------------------------


def test_weigh_unequal_group_sizes_raises():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=0, direction="heavier", max_weighings=3)
    try:
        game.weigh(bs, left=[0, 1, 2], right=[3, 4], weighings_used=0)
        assert False, "should have raised"
    except ValueError as exc:
        assert "equal size" in str(exc)


def test_weigh_overlapping_groups_raises():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=0, direction="heavier", max_weighings=3)
    try:
        game.weigh(bs, left=[0, 1, 2], right=[2, 3, 4], weighings_used=0)
        assert False, "should have raised"
    except ValueError as exc:
        assert "disjoint" in str(exc)


def test_weigh_out_of_range_ball_raises():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=0, direction="heavier", max_weighings=3)
    try:
        game.weigh(bs, left=[0, 1, 12], right=[3, 4, 5], weighings_used=0)
        assert False, "should have raised"
    except ValueError as exc:
        assert "out of range" in str(exc)


def test_weigh_budget_exceeded_raises():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=0, direction="heavier", max_weighings=2)
    # Use up the budget
    game.weigh(bs, left=[0, 1], right=[2, 3], weighings_used=0)
    game.weigh(bs, left=[4, 5], right=[6, 7], weighings_used=1)
    # Third call should fail
    try:
        game.weigh(bs, left=[8, 9], right=[10, 11], weighings_used=2)
        assert False, "should have raised"
    except ValueError as exc:
        assert "budget" in str(exc)


def test_weigh_empty_group_raises():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=0, direction="heavier", max_weighings=3)
    try:
        game.weigh(bs, left=[], right=[], weighings_used=0)
        assert False, "should have raised"
    except ValueError as exc:
        assert "empty" in str(exc)


# ---------------------------------------------------------------------------
# Answer verification
# ---------------------------------------------------------------------------


def test_check_answer_full_correct():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=5, direction="heavier", max_weighings=3)
    result = game.check_answer(bs, 5, "heavier")

    assert result["ball_correct"] is True
    assert result["direction_correct"] is True
    assert result["full_correct"] is True


def test_check_answer_identity_only():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=5, direction="heavier", max_weighings=3)
    result = game.check_answer(bs, 5, "lighter")

    assert result["ball_correct"] is True
    assert result["direction_correct"] is False
    assert result["full_correct"] is False


def test_check_answer_completely_wrong():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=5, direction="heavier", max_weighings=3)
    result = game.check_answer(bs, 3, "lighter")

    assert result["ball_correct"] is False
    assert result["direction_correct"] is False
    assert result["full_correct"] is False


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------


def test_generator_produces_valid_records():
    generator = _load_module("dataset_generator")

    records = generator.generate_records(16, seed=7, num_balls=12, max_weighings=3)

    assert len(records) == 16
    for record in records:
        assert "id" in record
        assert record["num_balls"] == 12
        assert 0 <= record["odd_ball_index"] < 12
        assert record["direction"] in ("heavier", "lighter")
        assert record["max_weighings"] == 3


def test_generator_is_deterministic_with_same_seed():
    generator = _load_module("dataset_generator")

    r1 = generator.generate_records(8, seed=42)
    r2 = generator.generate_records(8, seed=42)

    assert r1 == r2


def test_generator_different_seeds_produce_different_records():
    generator = _load_module("dataset_generator")

    r1 = generator.generate_records(16, seed=1)
    r2 = generator.generate_records(16, seed=2)

    assert r1 != r2


def test_generator_random_num_balls_range():
    """Records should have varying num_balls within the specified range."""
    generator = _load_module("dataset_generator")

    records = generator.generate_records(32, seed=7, num_balls_range=(3, 8))

    assert len(records) == 32
    ball_counts = {r["num_balls"] for r in records}
    assert len(ball_counts) > 1  # should have multiple different ball counts
    for record in records:
        assert 3 <= record["num_balls"] <= 8
        assert 0 <= record["odd_ball_index"] < record["num_balls"]
        assert record["direction"] in ("heavier", "lighter")
        assert record["max_weighings"] >= 1


def test_generator_split_records():
    """split_records should produce correct train/test sizes."""
    generator = _load_module("dataset_generator")

    records = generator.generate_records(30, seed=7, num_balls=6)
    train, test = generator.split_records(records, 0.33)

    assert len(train) + len(test) == 30
    assert len(test) == 9  # int(30 * 0.33) = 9
    assert len(train) == 21
    # Ensure no overlap
    train_ids = {r["id"] for r in train}
    test_ids = {r["id"] for r in test}
    assert train_ids & test_ids == set()


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def test_loader_formats_records_with_prompt():
    loader = _load_module("dataset_loader")
    game = _load_module("game")

    raw = {
        "id": "test-001",
        "num_balls": 8,
        "odd_ball_index": 3,
        "direction": "heavier",
        "max_weighings": 2,
    }

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(raw) + "\n")
        f.flush()
        records = loader.load_training_dataset(f.name)

    assert len(records) == 1
    assert records[0]["id"] == "test-001"
    assert records[0]["num_balls"] == 8
    assert records[0]["odd_ball_index"] == 3
    assert records[0]["direction"] == "heavier"
    assert records[0]["max_weighings"] == 2
    assert "8 balls" in records[0]["prompt"]


def test_loader_fallback_to_generator_when_file_missing():
    loader = _load_module("dataset_loader")

    records = loader.load_training_dataset("/nonexistent/path/to/file.jsonl")

    assert len(records) > 0
    assert "prompt" in records[0]


# ---------------------------------------------------------------------------
# Reward function (information-gain-aware continuous reward)
# ---------------------------------------------------------------------------

ALPHA = 0.05
REPEAT_PENALTY = 0.3
INVALID_PENALTY = 0.2
WRONG_ANSWER_PENALTY = -0.5


def _reward_record(source: dict, tool_calls: list, metadata: dict | None = None):
    return SimpleNamespace(
        source_record=source,
        tool_calls=tool_calls,
        metadata=metadata if metadata is not None else {},
    )


def _base_reward(num_balls: int) -> int:
    import math
    return max(1, math.ceil(math.log(num_balls * 2, 3)))


def test_reward_full_correct_no_weighings():
    """Correct answer with 0 weighings → reward = base (maximum)."""
    reward = _load_module("reward")

    source = {"num_balls": 12, "odd_ball_index": 5, "direction": "heavier", "max_weighings": 6}
    record = _reward_record(
        source,
        [{"name": "submit_answer", "arguments": json.dumps({"ball_index": 5, "direction": "heavier"})}],
    )

    base = _base_reward(12)
    assert reward.reward_fn(record) == float(base)
    assert record.metadata["full_answer_accuracy"] == 1.0
    assert record.metadata["valid_weighings"] == 0


def test_reward_full_correct_with_weighings():
    """Correct answer with 1 weighing → reward = base - alpha."""
    reward = _load_module("reward")

    source = {"num_balls": 12, "odd_ball_index": 5, "direction": "heavier", "max_weighings": 6}
    record = _reward_record(
        source,
        [
            {"name": "weigh", "arguments": json.dumps({"left": [0, 1, 2], "right": [3, 4, 5]})},
            {"name": "submit_answer", "arguments": json.dumps({"ball_index": 5, "direction": "heavier"})},
        ],
    )

    base = _base_reward(12)
    expected = float(base) - ALPHA
    assert reward.reward_fn(record) == expected
    assert record.metadata["full_answer_accuracy"] == 1.0
    assert record.metadata["valid_weighings"] == 1


def test_reward_identity_only():
    """Ball correct but direction wrong → reward = base/2 - alpha."""
    reward = _load_module("reward")

    source = {"num_balls": 12, "odd_ball_index": 5, "direction": "heavier", "max_weighings": 6}
    record = _reward_record(
        source,
        [
            {"name": "weigh", "arguments": json.dumps({"left": [0, 1], "right": [5, 2]})},
            {"name": "submit_answer", "arguments": json.dumps({"ball_index": 5, "direction": "lighter"})},
        ],
    )

    base = _base_reward(12)
    expected = float(base) / 2.0 - ALPHA
    assert reward.reward_fn(record) == expected
    assert record.metadata["full_answer_accuracy"] == 0.0
    assert record.metadata["identity_only_accuracy"] == 1.0


def test_reward_completely_wrong():
    """Wrong ball and direction → reward = WRONG_ANSWER_PENALTY - alpha."""
    reward = _load_module("reward")

    source = {"num_balls": 12, "odd_ball_index": 5, "direction": "heavier", "max_weighings": 6}
    record = _reward_record(
        source,
        [
            {"name": "weigh", "arguments": json.dumps({"left": [0, 1], "right": [2, 3]})},
            {"name": "submit_answer", "arguments": json.dumps({"ball_index": 3, "direction": "lighter"})},
        ],
    )

    expected = WRONG_ANSWER_PENALTY - ALPHA
    assert reward.reward_fn(record) == expected
    assert record.metadata["full_answer_accuracy"] == 0.0
    assert record.metadata["identity_only_accuracy"] == 0.0


def test_reward_no_submit_answer():
    """No submit_answer → reward = -1 (fixed penalty)."""
    reward = _load_module("reward")

    source = {"num_balls": 12, "odd_ball_index": 5, "direction": "heavier", "max_weighings": 6}
    record = _reward_record(
        source,
        [
            {"name": "weigh", "arguments": json.dumps({"left": [0, 1], "right": [2, 3]})},
            {"name": "weigh", "arguments": json.dumps({"left": [4, 5], "right": [6, 7]})},
        ],
    )

    assert reward.reward_fn(record) == -1.0
    assert record.metadata["full_answer_accuracy"] == 0.0
    assert record.metadata["valid_weighings"] == 2


def test_reward_repeated_weighing_penalized():
    """Same weighing repeated → repeat penalty applied."""
    reward = _load_module("reward")

    source = {"num_balls": 12, "odd_ball_index": 5, "direction": "heavier", "max_weighings": 6}
    record = _reward_record(
        source,
        [
            {"name": "weigh", "arguments": json.dumps({"left": [0, 1], "right": [2, 3]})},
            {"name": "weigh", "arguments": json.dumps({"left": [0, 1], "right": [2, 3]})},  # repeat
            {"name": "submit_answer", "arguments": json.dumps({"ball_index": 5, "direction": "heavier"})},
        ],
    )

    base = _base_reward(12)
    # 1 valid + 1 repeated: cost = (1+1)*alpha + 1*repeat_penalty
    expected = float(base) - 2 * ALPHA - REPEAT_PENALTY
    assert reward.reward_fn(record) == expected
    assert record.metadata["valid_weighings"] == 1
    assert record.metadata["repeated_weighings"] == 1


def test_reward_invalid_weighing_penalized():
    """Invalid weighing (unequal size) → invalid penalty applied."""
    reward = _load_module("reward")

    source = {"num_balls": 12, "odd_ball_index": 5, "direction": "heavier", "max_weighings": 6}
    record = _reward_record(
        source,
        [
            {"name": "weigh", "arguments": json.dumps({"left": [0, 1, 2], "right": [3, 4]})},  # invalid
            {"name": "submit_answer", "arguments": json.dumps({"ball_index": 5, "direction": "heavier"})},
        ],
    )

    base = _base_reward(12)
    expected = float(base) - INVALID_PENALTY  # 0 valid weighings, 1 invalid
    assert reward.reward_fn(record) == expected
    assert record.metadata["valid_weighings"] == 0
    assert record.metadata["invalid_weighings"] == 1


def test_reward_scales_with_num_balls():
    """Larger num_balls → higher base reward, auto-adaptive."""
    reward = _load_module("reward")

    # 6 balls → base = ceil(log3(12)) = 3
    source_small = {"num_balls": 6, "odd_ball_index": 0, "direction": "heavier", "max_weighings": 6}
    r_small = _reward_record(
        source_small,
        [{"name": "submit_answer", "arguments": json.dumps({"ball_index": 0, "direction": "heavier"})}],
    )

    # 100 balls → base = ceil(log3(200)) = 5
    source_large = {"num_balls": 100, "odd_ball_index": 0, "direction": "heavier", "max_weighings": 12}
    r_large = _reward_record(
        source_large,
        [{"name": "submit_answer", "arguments": json.dumps({"ball_index": 0, "direction": "heavier"})}],
    )

    r_small_val = reward.reward_fn(r_small)
    r_large_val = reward.reward_fn(r_large)
    assert r_large_val > r_small_val  # bigger puzzle → bigger reward


def test_reward_metadata_tracks_all_components():
    """Verify metadata records all reward components for metric aggregation."""
    reward = _load_module("reward")

    source = {"num_balls": 12, "odd_ball_index": 5, "direction": "heavier", "max_weighings": 6}
    record = _reward_record(
        source,
        [
            {"name": "weigh", "arguments": json.dumps({"left": [0, 1], "right": [2, 3]})},
            {"name": "weigh", "arguments": json.dumps({"left": [0, 1], "right": [2, 3]})},  # repeat
            {"name": "weigh", "arguments": json.dumps({"left": [0], "right": [1, 2]})},  # invalid
            {"name": "submit_answer", "arguments": json.dumps({"ball_index": 5, "direction": "heavier"})},
        ],
    )

    reward.reward_fn(record)

    assert record.metadata["valid_weighings"] == 1
    assert record.metadata["repeated_weighings"] == 1
    assert record.metadata["invalid_weighings"] == 1
    assert record.metadata["full_answer_accuracy"] == 1.0
    assert "reward_components" in record.metadata
    assert record.metadata["reward_components"]["k"] is not None
    assert record.metadata["reward_components"]["repeat_cost"] == REPEAT_PENALTY
    assert record.metadata["reward_components"]["invalid_cost"] == INVALID_PENALTY


# ---------------------------------------------------------------------------
# Agent tool execution (requires torch because run_agent.py imports
# areno.api.agentic which transitively needs torch for the rollout proxy)
# ---------------------------------------------------------------------------


@requires_torch
def test_run_agent_weigh_tool_executes_correctly():
    run_agent = _load_module("run_agent")
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=5, direction="heavier", max_weighings=3)
    args = json.dumps({"left": [0, 1, 2], "right": [3, 4, 5]})

    result, did_weigh = run_agent._run_weigh(args, bs, weighings_used=0)

    assert did_weigh is True
    assert result["result"] == "right_heavy"
    assert result["weighings_used"] == 1


@requires_torch
def test_run_agent_weigh_tool_rejects_invalid_input():
    run_agent = _load_module("run_agent")
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=5, direction="heavier", max_weighings=3)
    args = json.dumps({"left": [0, 1, 2], "right": [3, 4]})

    result, did_weigh = run_agent._run_weigh(args, bs, weighings_used=0)

    assert did_weigh is False
    assert "error" in result


@requires_torch
def test_run_agent_weigh_tool_budget_exceeded():
    run_agent = _load_module("run_agent")
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=5, direction="heavier", max_weighings=2)
    args = json.dumps({"left": [0, 1], "right": [2, 3]})

    # Exhaust budget
    run_agent._run_weigh(args, bs, weighings_used=0)
    run_agent._run_weigh(args, bs, weighings_used=1)

    # Third attempt should fail
    result, did_weigh = run_agent._run_weigh(args, bs, weighings_used=2)

    assert did_weigh is False
    assert "budget" in result["error"]


@requires_torch
def test_run_agent_submit_answer_returns_submission():
    run_agent = _load_module("run_agent")

    args = json.dumps({"ball_index": 5, "direction": "heavier"})
    result = run_agent._run_submit_answer(args)

    assert result["submitted"] is True
    assert result["ball_index"] == 5
    assert result["direction"] == "heavier"


@requires_torch
def test_run_agent_submit_answer_rejects_invalid_direction():
    run_agent = _load_module("run_agent")

    args = json.dumps({"ball_index": 5, "direction": "wrong"})
    result = run_agent._run_submit_answer(args)

    assert "error" in result


@requires_torch
def test_run_agent_submit_answer_rejects_missing_fields():
    run_agent = _load_module("run_agent")

    args = json.dumps({"ball_index": 5})
    result = run_agent._run_submit_answer(args)

    assert "error" in result


@requires_torch
def test_run_agent_tool_result_message_format():
    run_agent = _load_module("run_agent")

    call = {"id": "call-1", "function": {"name": "weigh", "arguments": "{}"}}
    result = {"result": "balanced"}
    msg = run_agent._tool_result_message(call, result)

    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call-1"
    assert msg["name"] == "weigh"
    assert json.loads(msg["content"])["result"] == "balanced"


# ---------------------------------------------------------------------------
# Orchestration logic isolated behind fakes (GPU-only behaviour)
#
# Issue #185: "For distributed or GPU-only behaviour, isolate orchestration
# logic behind fakes and document the minimal GPU validation that remains."
#
# The tests below drive _run_puzzle_loop with a fake model callback that
# returns scripted tool calls, verifying budget enforcement, tool dispatch,
# and message accumulation without any GPU, network, or torch dependency
# beyond the import of areno.api.agentic for AgentTrajectoryTurn.
# ---------------------------------------------------------------------------


def _fake_tool_call(name: str, arguments: dict, call_id: str = "call-1"):
    """Build a normalized tool call dict matching the run_agent format."""

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _make_fake_model_callback(scripted_calls: list[dict]):
    """Return an async callback that replays scripted tool calls in order.

    Each entry in ``scripted_calls`` is a dict with keys:
      - ``tool_calls``: list of tool call dicts (from _fake_tool_call)
      - ``content`` (optional): assistant text content
    """

    call_index = {"i": 0}

    async def callback(messages, tools, tool_choice):
        idx = call_index["i"]
        call_index["i"] += 1
        if idx < len(scripted_calls):
            entry = scripted_calls[idx]
            return {
                "response": None,
                "content": entry.get("content"),
                "tool_calls": entry.get("tool_calls", []),
            }
        return {"response": None, "content": None, "tool_calls": []}

    return callback


@requires_torch
def test_orchestration_completes_when_submit_answer_called():
    """Verify the loop terminates after submit_answer and accumulates messages."""

    run_agent = _load_module("run_agent")
    game = _load_module("game")

    bs = game.BallSet(num_balls=6, odd_ball_index=2, direction="heavier", max_weighings=3)
    item = _SimpleItem(record={"num_balls": 6, "odd_ball_index": 2, "direction": "heavier", "max_weighings": 3})

    callback = _make_fake_model_callback([
        {"tool_calls": [_fake_tool_call("weigh", {"left": [0, 1], "right": [2, 3]}, "c1")]},
        {"tool_calls": [_fake_tool_call("submit_answer", {"ball_index": 2, "direction": "heavier"}, "c2")]},
    ])

    turns, messages = asyncio_run(run_agent._run_puzzle_loop(item, bs, callback))

    assert len(turns) == 2
    # system + user + assistant(weigh) + tool(weigh) + assistant(submit) + tool(submit)
    assert len(messages) == 6
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "weigh"
    assert messages[4]["tool_calls"][0]["function"]["name"] == "submit_answer"


@requires_torch
def test_orchestration_enforces_budget_and_forces_submit():
    """When weighings are exhausted, the loop forces submit_answer."""

    run_agent = _load_module("run_agent")
    game = _load_module("game")

    bs = game.BallSet(num_balls=4, odd_ball_index=1, direction="lighter", max_weighings=1)
    item = _SimpleItem(record={"num_balls": 4, "odd_ball_index": 1, "direction": "lighter", "max_weighings": 1})

    callback = _make_fake_model_callback([
        # Turn 1: weigh (uses the single allowed weighing)
        {"tool_calls": [_fake_tool_call("weigh", {"left": [0], "right": [1]}, "c1")]},
        # Turn 2: budget exhausted, hint forces submit_answer — model complies
        {"tool_calls": [_fake_tool_call("submit_answer", {"ball_index": 1, "direction": "lighter"}, "c2")]},
    ])

    turns, messages = asyncio_run(run_agent._run_puzzle_loop(item, bs, callback))

    assert len(turns) == 2
    # The second turn should include a user hint about budget exhaustion
    hint_msg = turns[1].messages[-1]  # last message in turn 2
    assert hint_msg["role"] == "user"
    assert "used all" in hint_msg["content"]


@requires_torch
def test_orchestration_stops_when_model_returns_no_tool_call():
    """If the model returns no tool calls, the loop stops gracefully."""

    run_agent = _load_module("run_agent")
    game = _load_module("game")

    bs = game.BallSet(num_balls=6, odd_ball_index=0, direction="heavier", max_weighings=3)
    item = _SimpleItem(record={"num_balls": 6, "odd_ball_index": 0, "direction": "heavier", "max_weighings": 3})

    callback = _make_fake_model_callback([
        {"tool_calls": [], "content": "I give up."},
    ])

    turns, messages = asyncio_run(run_agent._run_puzzle_loop(item, bs, callback))

    assert len(turns) == 1
    assert len(messages) == 2  # system + user only


@requires_torch
def test_orchestration_weigh_result_appended_to_messages():
    """Verify the weigh tool result is correctly appended as a tool message."""

    run_agent = _load_module("run_agent")
    game = _load_module("game")

    bs = game.BallSet(num_balls=6, odd_ball_index=0, direction="heavier", max_weighings=3)
    item = _SimpleItem(record={"num_balls": 6, "odd_ball_index": 0, "direction": "heavier", "max_weighings": 3})

    callback = _make_fake_model_callback([
        {"tool_calls": [_fake_tool_call("weigh", {"left": [0, 1], "right": [2, 3]}, "c1")]},
        {"tool_calls": [_fake_tool_call("submit_answer", {"ball_index": 0, "direction": "heavier"}, "c2")]},
    ])

    turns, messages = asyncio_run(run_agent._run_puzzle_loop(item, bs, callback))

    # Find the tool result message for the weigh call
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2  # weigh result + submit result
    weigh_result = json.loads(tool_msgs[0]["content"])
    assert weigh_result["result"] == "left_heavy"  # ball 0 heavier on left
    assert weigh_result["weighings_used"] == 1


@requires_torch
def test_orchestration_unknown_tool_returns_error():
    """An unknown tool name produces an error tool result without crashing."""

    run_agent = _load_module("run_agent")
    game = _load_module("game")

    bs = game.BallSet(num_balls=6, odd_ball_index=0, direction="heavier", max_weighings=3)
    item = _SimpleItem(record={"num_balls": 6, "odd_ball_index": 0, "direction": "heavier", "max_weighings": 3})

    callback = _make_fake_model_callback([
        {"tool_calls": [_fake_tool_call("guess", {}, "c1")]},
        {"tool_calls": [_fake_tool_call("submit_answer", {"ball_index": 0, "direction": "heavier"}, "c2")]},
    ])

    turns, messages = asyncio_run(run_agent._run_puzzle_loop(item, bs, callback))

    tool_msgs = [m for m in messages if m["role"] == "tool"]
    unknown_result = json.loads(tool_msgs[0]["content"])
    assert "error" in unknown_result
    assert "unknown tool" in unknown_result["error"]


@requires_torch
def test_orchestration_multi_weigh_sequence():
    """Verify a full 3-weigh sequence with correct ball (12 balls, ball 5 heavier)."""

    run_agent = _load_module("run_agent")
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=5, direction="heavier", max_weighings=3)
    item = _SimpleItem(
        record={"num_balls": 12, "odd_ball_index": 5, "direction": "heavier", "max_weighings": 3},
    )

    callback = _make_fake_model_callback([
        {"tool_calls": [_fake_tool_call("weigh", {"left": [0, 1, 2, 3], "right": [4, 5, 6, 7]}, "c1")]},
        {"tool_calls": [_fake_tool_call("weigh", {"left": [4], "right": [5]}, "c2")]},
        {"tool_calls": [_fake_tool_call("submit_answer", {"ball_index": 5, "direction": "heavier"}, "c3")]},
    ])

    turns, messages = asyncio_run(run_agent._run_puzzle_loop(item, bs, callback))

    assert len(turns) == 3
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 3  # 2 weigh results + 1 submit result
    assert json.loads(tool_msgs[0]["content"])["result"] == "right_heavy"
    assert json.loads(tool_msgs[1]["content"])["result"] == "right_heavy"


# ---------------------------------------------------------------------------
# GPU validation boundary (documentation)
# ---------------------------------------------------------------------------
#
# The following aspects require a real GPU environment (e.g. Kaggle with
# CUDA) and are NOT covered by CPU tests:
#
# 1. run_agent() end-to-end: requires a live OpenAI-compatible rollout proxy
#    backed by AReno's engine workers (tensor-parallel inference on GPU).
#    The fake-callback tests above isolate the orchestration logic.
#
# 2. Multi-turn trajectory tokenisation: AgentTrajectoryTurn construction
#    with a real model response triggers areno.api.agentic tokenisation
#    and loss-mask logic that depends on the tokenizer + engine stack.
#
# 3. Reward signal integration with GSPO/GRPO training loop: the reward_fn
#    is tested standalone, but the training loop that consumes it requires
#    GPU rollout and gradient computation.
#
# Minimal GPU validation checklist (run on Kaggle or similar):
#   - python examples/agentic/balance_scale/dataset_generator.py --count 4
#   - areno train --ckpt <model> --dataset-path <jsonl> \
#       --dataset-loader-fn examples/agentic/balance_scale/dataset_loader.py \
#       --reward-fn-path examples/agentic/balance_scale/reward.py \
#       --agent-fn examples/agentic/balance_scale/run_agent.py \
#       --algo gspo --batch-size 2 --n-samples 4 --max-new-tokens 256
#   - Verify reward > 0 on at least some samples after 1 step.


# ---------------------------------------------------------------------------
# SFT data generator and loader
# ---------------------------------------------------------------------------


def test_sft_solver_solves_correctly():
    """The ternary search solver should produce correct answers."""
    sft_gen = _load_module("sft_data_generator")
    game = _load_module("game")

    import random
    rng = random.Random(42)
    correct = 0
    total = 50
    for _ in range(total):
        n = rng.randint(3, 12)
        idx = rng.randint(0, n - 1)
        d = rng.choice(["heavier", "lighter"])
        bs = game.BallSet(num_balls=n, odd_ball_index=idx, direction=d, max_weighings=10)
        turns = sft_gen.solve_puzzle(bs)
        last = turns[-1]
        assert last["action"] == "submit_answer"
        if last["ball_index"] == idx and last["direction"] == d:
            correct += 1

    assert correct == total, f"Solver only got {correct}/{total} correct"


def test_sft_solver_uses_multiple_weighings():
    """Solver should use 2+ weighings for non-trivial puzzles."""
    sft_gen = _load_module("sft_data_generator")
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=5, direction="heavier", max_weighings=8)
    turns = sft_gen.solve_puzzle(bs)

    weighings = [t for t in turns if t["action"] == "weigh"]
    assert len(weighings) >= 2, f"Expected 2+ weighings, got {len(weighings)}"


def test_sft_loader_flattens_records():
    """SFT loader should flatten multi-turn conversations into prompt/response rows."""
    sft_gen = _load_module("sft_data_generator")
    sft_loader = _load_module("sft_loader")
    game = _load_module("game")

    # Generate a few SFT records
    records = sft_gen.generate_sft_records(4, seed=42, num_balls=6)

    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()

    rows = sft_loader.load_training_dataset(f.name)

    assert len(rows) > 4  # Each puzzle produces multiple rows (one per assistant turn)
    for row in rows:
        assert "prompt" in row
        assert "response" in row
        assert len(row["response"]) > 0
        # Response should be JSON (weigh or submit_answer arguments)
        parsed = json.loads(row["response"])
        assert isinstance(parsed, dict)


def test_sft_generator_skips_fewer_than_3_balls():
    """SFT generator should skip puzzles with fewer than 3 balls."""
    sft_gen = _load_module("sft_data_generator")

    records = sft_gen.generate_sft_records(10, seed=42, num_balls=2)
    # num_balls=2 means all puzzles have 2 balls, which should all be skipped
    assert len(records) == 0


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def test_format_prompt_contains_key_info():
    game = _load_module("game")

    bs = game.BallSet(num_balls=12, odd_ball_index=0, direction="heavier", max_weighings=3)
    prompt = game.format_prompt(bs)

    assert "12" in prompt
    assert "weigh" in prompt
    assert "submit_answer" in prompt
    assert "heavier" in prompt
    assert "lighter" in prompt
    # Few-shot example should be present
    assert "left" in prompt
    assert "right" in prompt
    assert "ball_index" in prompt
    assert "direction" in prompt


def test_format_system_prompt_describes_task():
    game = _load_module("game")

    prompt = game.format_system_prompt()

    assert "balance-scale" in prompt
    assert "weigh" in prompt
    assert "submit_answer" in prompt
