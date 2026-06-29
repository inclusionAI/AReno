from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "agentic" / "coding"))

from agent_loop import _response_usage  # noqa: E402
from code_cli import _compact_messages  # noqa: E402


def test_response_usage_extracts_openai_token_counts() -> None:
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=123, completion_tokens=7, total_tokens=130))

    assert _response_usage(response) == {
        "prompt_tokens": 123,
        "completion_tokens": 7,
        "total_tokens": 130,
    }


def test_response_usage_accepts_dict_usage() -> None:
    response = SimpleNamespace(usage={"prompt_tokens": "321", "total_tokens": 400})

    assert _response_usage(response) == {
        "prompt_tokens": 321,
        "total_tokens": 400,
    }


def test_code_cli_compacts_using_reported_prompt_tokens() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": '{"content":"old"}'},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "run_command", "arguments": '{"command":"pytest"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "name": "run_command", "content": '{"stdout":"ok"}'},
    ]

    compacted = _compact_messages(messages, prompt_tokens=4097, keep_recent=2)

    assert compacted is messages
    assert compacted[0]["role"] == "system"
    assert compacted[1]["role"] == "user"
    assert compacted[2]["role"] == "user"
    assert compacted[2]["content"].startswith("Compacted prior conversation:")
    assert "assistant called read_file" in compacted[2]["content"]
    assert compacted[-1]["name"] == "run_command"
