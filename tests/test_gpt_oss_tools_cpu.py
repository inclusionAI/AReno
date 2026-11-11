"""Harmony tool-call framing and shared OpenAI response contracts."""

import json
from types import SimpleNamespace

import pytest

from areno.api.openai_chat import build_chat_completion_response
from areno.api.tool_call_parser import get_tool_call_parser, infer_tool_call_parser_name

TOOLS = [
    {"type": "function", "function": {"name": "choose_square", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "inspect_board", "parameters": {"type": "object"}}},
]


@pytest.mark.parametrize("channel,handoff", [("channel", "call"), ("meta_sep", "ghissue")])
@pytest.mark.parametrize(
    "header",
    [
        "to=functions.choose_square<|{channel}|>commentary json",
        "<|{channel}|>commentary to=functions.choose_square",
        "<|start|>assistant to=choose_square<|{channel}|>commentary",
    ],
)
def test_gpt_oss_harmony_headers_cpu(channel, handoff, header):
    text = header.format(channel=channel) + '<|message|>{"square":3}<|' + handoff + "|>"
    parsed = get_tool_call_parser("gpt_oss").parse(text, TOOLS, "auto")
    assert parsed.normal_text == ""
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call["function"]["name"] == "choose_square"
    assert json.loads(call["function"]["arguments"]) == {"square": 3}
    assert call["id"].startswith("call_")


def test_gpt_oss_analysis_then_tool_cpu():
    text = (
        '<|channel|>analysis<|message|>Example: {"square":9}. I should win instead.<|end|>'
        '<|start|>assistant to=functions.choose_square<|channel|>commentary<|message|>{"square":3}<|call|>'
    )
    parsed = get_tool_call_parser("gpt_oss").parse(text, TOOLS, "required")
    assert len(parsed.tool_calls) == 1
    assert json.loads(parsed.tool_calls[0]["function"]["arguments"]) == {"square": 3}


@pytest.mark.parametrize(
    "text",
    [
        '{"square":3}',
        '<|channel|>analysis<|message|>{"square":3}<|end|>',
        '<|channel|>final<|message|>{"square":3}<|return|>',
        'to=functions.choose_square<|channel|>commentary<|message|>{"square":3}',
        'to=functions.choose_square<|message|>{"square":3}<|end|>',
        'to=functions.choose_square<|message|>{"square":3}<|return|>',
        'to=functions.choose_square<|message|>{"square":<|call|>',
        "to=functions.choose_square<|message|>[3]<|call|>",
        'to=functions.unknown<|message|>{"square":3}<|call|>',
        '<|start|>user to=functions.choose_square<|message|>{"square":3}<|call|>',
        '<|start|>functions.inspect_board to=functions.choose_square<|message|>{"square":3}<|call|>',
    ],
)
def test_gpt_oss_rejects_non_calls_cpu(text):
    forced = {"type": "function", "function": {"name": "choose_square"}}
    parsed = get_tool_call_parser("gpt_oss").parse(text, TOOLS, forced)
    assert not parsed.tool_calls
    assert parsed.normal_text == text


@pytest.mark.parametrize("choice", ["none", {"type": "function", "function": {"name": "inspect_board"}}])
def test_gpt_oss_respects_tool_choice_cpu(choice):
    text = 'to=functions.choose_square<|message|>{"square":3}<|call|>'
    assert not get_tool_call_parser("gpt_oss").parse(text, TOOLS, choice).tool_calls


def test_gpt_oss_multiple_messages_and_nested_arguments_cpu():
    arguments = {"board": [["X", None]], "note": 'quote " and brace }', "enabled": True}
    text = (
        'to=functions.choose_square<|message|>{"square":3}<|call|>'
        "<|start|>assistant<|meta_sep|>commentary to=functions.inspect_board<|message|>"
        + json.dumps(arguments)
        + "<|ghissue|>"
    )
    calls = get_tool_call_parser("gpt_oss").parse(text, TOOLS, "auto").tool_calls
    assert len(calls) == 2 and calls[0]["id"] != calls[1]["id"]
    assert json.loads(calls[1]["function"]["arguments"]) == arguments


@pytest.mark.parametrize("source", ["path", "config", "template"])
def test_gpt_oss_parser_inference_cpu(tmp_path, source):
    tokenizer = SimpleNamespace(chat_template="")
    path = str(tmp_path)
    if source == "path":
        path += "/gpt-oss-20b"
    elif source == "config":
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt_oss"}))
    else:
        tokenizer.chat_template = "<|start|>assistant to=functions.example<|call|>"
    trainer = SimpleNamespace(_model_path=path, get_tokenizer=lambda: tokenizer)
    assert infer_tool_call_parser_name(trainer) == "gpt_oss"


def test_gpt_oss_shared_openai_response_cpu():
    text = 'to=functions.choose_square<|channel|>commentary<|message|>{"square":3}<|call|>'
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: text)
    response = build_chat_completion_response(
        tokenizer=tokenizer,
        model="policy",
        prompt_tokens=5,
        response_ids=[[1, 2, 3]],
        finish_reasons=["stop"],
        tools=TOOLS,
        tool_call_parser=get_tool_call_parser("gpt_oss"),
    )
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "choose_square"
