from __future__ import annotations

from types import SimpleNamespace

from areno.api.tool_call_parser import QwenToolCallParser
from areno.cli import serve as serve_mod


def test_create_app_passes_eager_decode_runtime_config(monkeypatch):
    captured = {}

    class FakeEngine:
        config = SimpleNamespace(model=SimpleNamespace(max_position_embeddings=1024))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args
            captured["runtime_config"] = kwargs["runtime_config"]
            return cls()

    monkeypatch.setattr(serve_mod, "load_tokenizer", lambda model_path: SimpleNamespace(eos_token_id=1))
    monkeypatch.setattr(serve_mod, "ArenoEngine", FakeEngine)

    serve_mod.create_app(
        model_path="model",
        tp_size=1,
        world_size=1,
        max_running_prompts=4,
        default_max_tokens=16,
        decode_progress_interval_s=0.0,
        eager_decode=True,
        attn_backend="native",
    )

    assert captured["runtime_config"].eager_decode is True
    assert captured["runtime_config"].attn_backend == "native"


def test_chat_completion_request_defaults_match_sampling_params():
    request = serve_mod.ChatCompletionRequest(messages=[serve_mod.ChatMessage(role="user", content="hi")])

    assert request.temperature == 1.0
    assert request.top_p == 1.0
    assert request.top_k == -1


def test_serve_response_reuses_tool_call_parser():
    tokenizer = _TokenTokenizer(
        {1: "<tool_call>", 2: '{"name":"choose_move","arguments":{"direction":"left"}}', 3: "</tool_call>"}
    )
    request = serve_mod.ChatCompletionRequest(
        model="areno",
        messages=[serve_mod.ChatMessage(role="user", content="choose")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "choose_move",
                    "parameters": {
                        "type": "object",
                        "properties": {"direction": {"type": "string", "enum": ["left", "right"]}},
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "choose_move"}},
    )

    response = serve_mod._build_response_from(
        tokenizer,
        "model",
        QwenToolCallParser(),
        request,
        [10, 11],
        [[1, 2, 3]],
        ["stop"],
    )

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message["content"] is None
    assert choice.message["tool_calls"][0]["function"]["name"] == "choose_move"
    assert '"direction":"left"' in choice.message["tool_calls"][0]["function"]["arguments"]


def test_serve_chat_template_receives_tools_and_tool_messages():
    tokenizer = _ToolAwareTokenizer()
    messages = [
        serve_mod.ChatMessage(role="user", content="choose"),
        serve_mod.ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "choose_move", "arguments": "{}"}}],
        ),
        serve_mod.ChatMessage(role="tool", content="{}", tool_call_id="call-1", name="choose_move"),
    ]
    tools = [{"type": "function", "function": {"name": "choose_move"}}]

    assert serve_mod._encode_messages(tokenizer, messages, tools=tools) == [3]
    rendered_messages, rendered_tools = tokenizer.calls[0]
    assert rendered_tools == tools
    assert rendered_messages[1]["content"] == ""
    assert rendered_messages[1]["tool_calls"][0]["function"]["name"] == "choose_move"
    assert rendered_messages[2]["tool_call_id"] == "call-1"


class _TokenTokenizer:
    def __init__(self, pieces):
        self._pieces = pieces

    def decode(self, token_ids, *, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(self._pieces[token_id] for token_id in token_ids)


class _ToolAwareTokenizer:
    chat_template = "tool template"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs.get("tools")))
        return [len(messages)]
