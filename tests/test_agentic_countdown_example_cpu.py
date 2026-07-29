"""CPU unit tests for the Countdown arithmetic agentic RL example.

These tests exercise the pure-Python pieces of the example
(`dataset_loader.py`, `reward.py`, `run_agent.py`) without starting a
serving backend or hitting the network. They follow the same pattern as
`test_agentic_shopping_example_cpu.py`: modules are loaded via
`importlib.util` from the example directory so we don't pollute sys.path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "countdown"


def _load_module(name: str):
    """Load a Countdown example module from disk without polluting sys.path.

    We use importlib.util so the example files (which live outside the
    `areno` package) can be imported in isolation. This mirrors how
    AReno's CLI loads user-provided `--dataset-loader-fn` / `--reward-fn-path`
    / `--agent-fn` files at runtime.
    """
    path = EXAMPLE_DIR / f"{name}.py"
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"agentic_countdown_{name}_for_tests", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))


# ---------------------------------------------------------------------------
# dataset_loader.py
# ---------------------------------------------------------------------------


def test_countdown_loader_formats_prompt_with_numbers_and_target():
    loader = _load_module("dataset_loader")
    source = {"numbers": [25, 50, 75, 100, 3, 6], "target": 952, "id": "1"}

    records = loader.load_training_dataset("unused", default_loader=lambda _: [source])

    assert len(records) == 1
    record = records[0]
    # The prompt must surface every available number and the target so the
    # model knows what it's solving.
    assert "25" in record["prompt"]
    assert "50" in record["prompt"]
    assert "75" in record["prompt"]
    assert "100" in record["prompt"]
    assert "3" in record["prompt"]
    assert "6" in record["prompt"]
    assert "952" in record["prompt"]
    # Original fields are preserved for the reward function to read later.
    assert record["numbers"] == [25, 50, 75, 100, 3, 6]
    assert record["target"] == 952
    assert record["id"] == "1"


def test_countdown_loader_falls_back_to_synthetic_id_when_missing():
    loader = _load_module("dataset_loader")
    # No "id" field -- the loader should synthesize one so AReno can still
    # track individual samples through rollout.
    source = {"numbers": [1, 2, 3], "target": 6}

    records = loader.load_training_dataset("unused", default_loader=lambda _: [source])

    assert records[0]["id"] == "countdown-00001"


def test_countdown_loader_preserves_existing_id():
    loader = _load_module("dataset_loader")
    source = {"numbers": [1, 2, 3], "target": 6, "id": "custom-42"}

    records = loader.load_training_dataset("unused", default_loader=lambda _: [source])

    assert records[0]["id"] == "custom-42"


def test_countdown_loader_handles_multiple_rows():
    loader = _load_module("dataset_loader")
    sources = [
        {"numbers": [1, 2], "target": 3, "id": "a"},
        {"numbers": [4, 5], "target": 9, "id": "b"},
        {"numbers": [10, 20], "target": 30, "id": "c"},
    ]

    records = loader.load_training_dataset("unused", default_loader=lambda _: sources)

    assert len(records) == 3
    assert [r["id"] for r in records] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# reward.py
# ---------------------------------------------------------------------------


def _reward_record(target, tool_calls):
    """Build a minimal record compatible with reward_fn's expectations."""
    return SimpleNamespace(source_record={"target": target}, tool_calls=tool_calls)


def test_countdown_reward_exact_match_returns_one():
    reward = _load_module("reward")
    record = _reward_record(
        target=952,
        tool_calls=[{"name": "finish", "arguments": json.dumps({"answer": 952})}],
    )
    assert reward.reward_fn(record) == 1.0


def test_countdown_reward_within_ten_percent_returns_zero_seven():
    reward = _load_module("reward")
    # target=100, answer=105 -> relative_diff=0.05 -> 0.7
    record = _reward_record(
        target=100,
        tool_calls=[{"name": "finish", "arguments": json.dumps({"answer": 105})}],
    )
    assert reward.reward_fn(record) == 0.7


def test_countdown_reward_within_thirty_percent_returns_zero_three():
    reward = _load_module("reward")
    # target=100, answer=120 -> relative_diff=0.2 -> 0.3
    record = _reward_record(
        target=100,
        tool_calls=[{"name": "finish", "arguments": json.dumps({"answer": 120})}],
    )
    assert reward.reward_fn(record) == 0.3


def test_countdown_reward_linear_decay_between_thirty_and_one_hundred_percent():
    reward = _load_module("reward")
    # target=100, answer=150 -> relative_diff=0.5 -> 0.3 - (0.5-0.3)*(0.3/0.7)
    # = 0.3 - 0.2 * 0.4286... ≈ 0.2143
    record = _reward_record(
        target=100,
        tool_calls=[{"name": "finish", "arguments": json.dumps({"answer": 150})}],
    )
    result = reward.reward_fn(record)
    assert 0.0 < result < 0.3
    assert abs(result - (0.3 - 0.2 * (0.3 / 0.7))) < 1e-9


def test_countdown_reward_beyond_one_hundred_percent_clamps_to_zero():
    reward = _load_module("reward")
    # target=100, answer=500 -> relative_diff=4.0 -> max(0, 0.3 - 3.7*0.4286) < 0 -> 0.0
    record = _reward_record(
        target=100,
        tool_calls=[{"name": "finish", "arguments": json.dumps({"answer": 500})}],
    )
    assert reward.reward_fn(record) == 0.0


def test_countdown_reward_no_finish_call_returns_zero():
    reward = _load_module("reward")
    # Model made tool calls but never finished -- no signal about correctness.
    record = _reward_record(
        target=100,
        tool_calls=[
            {"name": "add", "arguments": json.dumps({"a": 1, "b": 2})},
            {"name": "multiply", "arguments": json.dumps({"a": 3, "b": 4})},
        ],
    )
    assert reward.reward_fn(record) == 0.0


def test_countdown_reward_empty_tool_calls_returns_zero():
    reward = _load_module("reward")
    record = _reward_record(target=100, tool_calls=[])
    assert reward.reward_fn(record) == 0.0


def test_countdown_reward_malformed_json_arguments_returns_negative_one():
    reward = _load_module("reward")
    # The model emitted a finish call but with broken JSON -- penalize so it
    # learns to emit valid tool-call arguments.
    record = _reward_record(
        target=100,
        tool_calls=[{"name": "finish", "arguments": "{not valid json"}],
    )
    assert reward.reward_fn(record) == -1.0


def test_countdown_reward_non_numeric_answer_returns_negative_one():
    reward = _load_module("reward")
    record = _reward_record(
        target=100,
        tool_calls=[{"name": "finish", "arguments": json.dumps({"answer": "not a number"})}],
    )
    assert reward.reward_fn(record) == -1.0


def test_countdown_reward_finish_without_answer_key_returns_zero():
    reward = _load_module("reward")
    # Valid JSON, but no "answer" field -- we can't extract a number, so we
    # fall through to the "no usable finish" branch (0.0), not the penalty
    # branch (-1.0), because the arguments were at least valid JSON.
    record = _reward_record(
        target=100,
        tool_calls=[{"name": "finish", "arguments": json.dumps({"reasoning": "I gave up"})}],
    )
    assert reward.reward_fn(record) == 0.0


def test_countdown_reward_accepts_dict_arguments_directly():
    reward = _load_module("reward")
    # AReno may hand us pre-parsed dict arguments instead of a JSON string;
    # the reward function should handle both shapes.
    record = _reward_record(
        target=100,
        tool_calls=[{"name": "finish", "arguments": {"answer": 100}}],
    )
    assert reward.reward_fn(record) == 1.0


def test_countdown_reward_target_zero_exact_match_returns_one():
    reward = _load_module("reward")
    # target == 0 can't be scored by relative error, so we fall back to an
    # exact-match check.
    record = _reward_record(
        target=0,
        tool_calls=[{"name": "finish", "arguments": json.dumps({"answer": 0})}],
    )
    assert reward.reward_fn(record) == 1.0


def test_countdown_reward_target_zero_non_zero_answer_returns_zero():
    reward = _load_module("reward")
    record = _reward_record(
        target=0,
        tool_calls=[{"name": "finish", "arguments": json.dumps({"answer": 5})}],
    )
    assert reward.reward_fn(record) == 0.0


# ---------------------------------------------------------------------------
# run_agent.py -- _execute_tool
# ---------------------------------------------------------------------------


def _assistant_message_with_call(name: str, arguments: dict | str) -> dict:
    """Build an assistant message containing a single tool call."""
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def test_countdown_execute_tool_add():
    run_agent = _load_module("run_agent")
    result = run_agent._execute_tool(_assistant_message_with_call("add", {"a": 3, "b": 5}))
    assert result["name"] == "add"
    assert result["result"] == 8.0
    assert result["error"] is None


def test_countdown_execute_tool_subtract():
    run_agent = _load_module("run_agent")
    result = run_agent._execute_tool(_assistant_message_with_call("subtract", {"a": 10, "b": 3}))
    assert result["name"] == "subtract"
    assert result["result"] == 7.0
    assert result["error"] is None


def test_countdown_execute_tool_multiply():
    run_agent = _load_module("run_agent")
    result = run_agent._execute_tool(_assistant_message_with_call("multiply", {"a": 4, "b": 6}))
    assert result["name"] == "multiply"
    assert result["result"] == 24.0
    assert result["error"] is None


def test_countdown_execute_tool_divide_integer_result():
    run_agent = _load_module("run_agent")
    result = run_agent._execute_tool(_assistant_message_with_call("divide", {"a": 10, "b": 2}))
    assert result["name"] == "divide"
    assert result["result"] == 5
    assert result["error"] is None


def test_countdown_execute_tool_divide_non_integer_result_errors():
    run_agent = _load_module("run_agent")
    # Countdown rules require integer division results. The error flag is the
    # signal the model learns from; the raw float is left in `result` since
    # the divide already evaluated it before the integer check.
    result = run_agent._execute_tool(_assistant_message_with_call("divide", {"a": 10, "b": 3}))
    assert result["name"] == "divide"
    assert result["result"] == 10 / 3
    assert result["error"] == "Result must be an integer"


def test_countdown_execute_tool_divide_by_zero_errors():
    run_agent = _load_module("run_agent")
    result = run_agent._execute_tool(_assistant_message_with_call("divide", {"a": 5, "b": 0}))
    assert result["name"] == "divide"
    assert result["result"] is None
    assert result["error"] == "Cannot divide by zero"


def test_countdown_execute_tool_finish_returns_answer_and_reasoning():
    run_agent = _load_module("run_agent")
    result = run_agent._execute_tool(
        _assistant_message_with_call("finish", {"answer": 952, "reasoning": "(100+6)*(75-50)-3*25"})
    )
    assert result["name"] == "finish"
    assert result["answer"] == 952
    assert result["reasoning"] == "(100+6)*(75-50)-3*25"


def test_countdown_execute_tool_finish_without_reasoning_defaults_to_empty():
    run_agent = _load_module("run_agent")
    result = run_agent._execute_tool(_assistant_message_with_call("finish", {"answer": 42}))
    assert result["name"] == "finish"
    assert result["answer"] == 42
    assert result["reasoning"] == ""


def test_countdown_execute_tool_unknown_tool_errors():
    run_agent = _load_module("run_agent")
    result = run_agent._execute_tool(_assistant_message_with_call("power", {"a": 2, "b": 3}))
    assert result["name"] == "power"
    assert result["result"] is None
    assert "Unknown tool" in result["error"]


def test_countdown_execute_tool_no_tool_calls_returns_none():
    run_agent = _load_module("run_agent")
    # No tool call at all -- the caller treats this as end-of-episode.
    result = run_agent._execute_tool({"role": "assistant", "content": "I'm stuck", "tool_calls": []})
    assert result is None


def test_countdown_execute_tool_no_tool_calls_key_returns_none():
    run_agent = _load_module("run_agent")
    result = run_agent._execute_tool({"role": "assistant", "content": "hello"})
    assert result is None


def test_countdown_execute_tool_invalid_json_arguments_errors():
    run_agent = _load_module("run_agent")
    result = run_agent._execute_tool(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{broken"},
                }
            ],
        }
    )
    assert result["name"] == "add"
    assert result["result"] is None
    assert result["error"] == "Invalid JSON arguments"


def test_countdown_execute_tool_arguments_not_object_errors():
    run_agent = _load_module("run_agent")
    # Valid JSON, but not an object (e.g. a bare number) -- should be rejected.
    result = run_agent._execute_tool(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "add", "arguments": "5"},
                }
            ],
        }
    )
    assert result["name"] == "add"
    assert result["result"] is None
    assert result["error"] == "Arguments must be an object"


# ---------------------------------------------------------------------------
# run_agent.py -- _tool_messages
# ---------------------------------------------------------------------------


def test_countdown_tool_messages_emits_assistant_then_tool():
    run_agent = _load_module("run_agent")
    assistant_message = _assistant_message_with_call("add", {"a": 1, "b": 2})
    tool_result = run_agent._execute_tool(assistant_message)

    messages = run_agent._tool_messages(assistant_message, tool_result)

    assert len(messages) == 2
    # First message is the assistant's tool call echoed back into the
    # conversation so the model can continue from it.
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "add"
    # Second message is the tool result the model sees next.
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_0"
    assert messages[1]["name"] == "add"
    parsed = json.loads(messages[1]["content"])
    assert parsed["result"] == 3.0
    assert parsed["error"] is None