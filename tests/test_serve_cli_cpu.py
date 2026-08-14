from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from areno.api.tokenizer import configure_chat_template_enable_thinking
from areno.api.tool_call_parser import QwenToolCallParser
from areno.cli import serve as serve_mod
from areno.engine.config import ModelConfig


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


def test_create_app_can_disable_chat_template_thinking(monkeypatch):
    class FakeEngine:
        config = SimpleNamespace(model=SimpleNamespace(max_position_embeddings=1024))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

    tokenizer = _ToolAwareTokenizer()
    monkeypatch.setattr(serve_mod, "load_tokenizer", lambda model_path: tokenizer)
    monkeypatch.setattr(serve_mod, "ArenoEngine", FakeEngine)

    app = serve_mod.create_app(
        model_path="model",
        tp_size=1,
        world_size=1,
        max_running_prompts=4,
        default_max_tokens=16,
        decode_progress_interval_s=0.0,
        attn_backend="native",
        chat_template_enable_thinking=False,
    )

    assert serve_mod._encode_messages(
        app.state.areno_serve.tokenizer, [serve_mod.ChatMessage(role="user", content="hi")]
    )
    assert tokenizer.calls[0][1]["enable_thinking"] is False


def test_create_app_falls_back_to_native_for_flash_unsupported_model(monkeypatch):
    captured = {}

    class FakeEngine:
        config = SimpleNamespace(model=SimpleNamespace(max_position_embeddings=1024))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args
            captured["runtime_config"] = kwargs["runtime_config"]
            return cls()

    model_config = ModelConfig(
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=16,
        vocab_size=32,
        head_dim=512,
    )
    monkeypatch.setattr(serve_mod, "load_tokenizer", lambda model_path: SimpleNamespace(eos_token_id=1))
    monkeypatch.setattr(serve_mod, "ArenoEngine", FakeEngine)
    monkeypatch.setattr(serve_mod, "config_from_hf", lambda model_path: model_config)
    monkeypatch.setattr(serve_mod, "flash_attention_unsupported_gpu_reason", lambda devices: None)

    with pytest.warns(RuntimeWarning, match="qk head dim 512.*attn_backend='native'.*slower"):
        serve_mod.create_app(
            model_path="model",
            tp_size=1,
            world_size=1,
            max_running_prompts=4,
            default_max_tokens=16,
            decode_progress_interval_s=0.0,
            attn_backend="flash",
        )

    assert captured["runtime_config"].attn_backend == "native"


def test_serve_default_max_running_prompts_is_16():
    option = next(param for param in serve_mod.serve_command.params if param.name == "max_running_prompts")

    assert option.default == 16


def test_serve_default_model_hub_is_modelscope():
    option = next(param for param in serve_mod.serve_command.params if param.name == "model_hub")

    assert option.default == "modelscope"


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


def test_serve_text_response_preserves_decoded_content():
    class ThinkTokenizer:
        def decode(self, token_ids, *, skip_special_tokens=False):
            del token_ids
            if skip_special_tokens:
                return "plan the answer</think>\n\nFinal answer"
            return "plan the answer</think>\n\nFinal answer<|im_end|>"

    request = serve_mod.ChatCompletionRequest(
        model="areno",
        messages=[serve_mod.ChatMessage(role="user", content="describe")],
    )

    response = serve_mod._build_response_from(
        ThinkTokenizer(),
        "model",
        QwenToolCallParser(),
        request,
        [10, 11],
        [[1, 2, 3]],
        ["stop"],
    )

    choice = response.choices[0]
    assert "reasoning_content" not in choice.message
    assert choice.message["content"] == "plan the answer</think>\n\nFinal answer"


def test_serve_tool_call_response_does_not_attach_reasoning_content():
    tokenizer = _TokenTokenizer(
        {
            1: "<think>block the fork</think>",
            2: "<tool_call>",
            3: '{"name":"choose_move","arguments":{"direction":"left"}}',
            4: "</tool_call>",
        }
    )
    request = serve_mod.ChatCompletionRequest(
        model="areno",
        messages=[serve_mod.ChatMessage(role="user", content="choose")],
        tools=[
            {
                "type": "function",
                "function": {"name": "choose_move"},
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
        [[1, 2, 3, 4]],
        ["stop"],
    )

    choice = response.choices[0]
    assert choice.message["content"] is None
    assert "reasoning_content" not in choice.message
    assert choice.message["tool_calls"][0]["function"]["name"] == "choose_move"


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
    rendered_messages, rendered_kwargs = tokenizer.calls[0]
    assert rendered_kwargs["tools"] == tools
    assert rendered_messages[1]["content"] == ""
    assert rendered_messages[1]["tool_calls"][0]["function"]["name"] == "choose_move"
    assert rendered_messages[2]["tool_call_id"] == "call-1"


def test_serve_chat_template_can_disable_thinking():
    tokenizer = _ToolAwareTokenizer()
    configure_chat_template_enable_thinking(tokenizer, False)

    assert serve_mod._encode_messages(tokenizer, [serve_mod.ChatMessage(role="user", content="hello")]) == [1]
    assert tokenizer.calls[0][1]["enable_thinking"] is False


def test_serve_multimodal_processor_chat_template_can_disable_thinking():
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP8z8BQDwAFgwJ/lwJw6QAAAABJRU5ErkJggg=="
    )

    class FakeProcessor:
        image_token_id = 99

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, dict(kwargs)))
            return "<image> describe"

        def __call__(self, *, text, images, return_tensors):
            del text, images, return_tensors
            return {
                "input_ids": torch.tensor([[1, 99, 2]]),
                "image_embeds": torch.ones(1, 4),
            }

    processor = FakeProcessor()
    configure_chat_template_enable_thinking(processor, False)
    messages = [
        serve_mod.ChatMessage(
            role="user",
            content=[
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": "describe"},
            ],
        )
    ]

    serve_mod._encode_messages_with_features(SimpleNamespace(eos_token_id=1), processor, messages)

    assert processor.calls[0][1]["enable_thinking"] is False


def test_serve_multimodal_encoder_uses_processor_for_image_data_url():
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP8z8BQDwAFgwJ/lwJw6QAAAABJRU5ErkJggg=="
    )

    class FakeProcessor:
        image_token_id = 99

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            self.messages = messages
            self.template_args = (tokenize, add_generation_prompt)
            return "<image> describe"

        def __call__(self, *, text, images, return_tensors):
            self.call_args = (text, len(images), return_tensors)
            return {
                "input_ids": torch.tensor([[1, 99, 2]]),
                "image_embeds": torch.ones(1, 4),
            }

    tokenizer = SimpleNamespace(eos_token_id=1)
    processor = FakeProcessor()
    messages = [
        serve_mod.ChatMessage(
            role="user",
            content=[
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": "describe"},
            ],
        )
    ]

    tokens, features = serve_mod._encode_messages_with_features(tokenizer, processor, messages)

    assert tokens == [1, 99, 2]
    assert features["image_token_id"] == 99
    assert torch.equal(features["image_embeds"], torch.ones(1, 4))
    assert processor.call_args == (["<image> describe"], 1, "pt")
    assert processor.messages[0]["content"][0]["type"] == "image"
    assert processor.messages[0]["content"][0]["image"] == image
    assert "image_url" not in processor.messages[0]["content"][0]


def test_serve_multimodal_encoder_passes_tools_to_processor_chat_template():
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP8z8BQDwAFgwJ/lwJw6QAAAABJRU5ErkJggg=="
    )

    class FakeProcessor:
        image_token_id = 99

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None):
            self.messages = messages
            self.kwargs = {
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "tools": tools,
            }
            return "<image> choose"

        def __call__(self, *, text, images, return_tensors):
            self.call_args = (text, len(images), return_tensors)
            return {
                "input_ids": torch.tensor([[1, 99, 2]]),
                "image_embeds": torch.ones(1, 4),
            }

    tools = [{"type": "function", "function": {"name": "choose_square"}}]
    processor = FakeProcessor()
    messages = [
        serve_mod.ChatMessage(
            role="user",
            content=[
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": "choose"},
            ],
        )
    ]

    tokens, features = serve_mod._encode_messages_with_features(
        SimpleNamespace(eos_token_id=1),
        processor,
        messages,
        tools=tools,
    )

    assert tokens == [1, 99, 2]
    assert features["image_token_id"] == 99
    assert processor.kwargs["tools"] == tools
    assert processor.messages[0]["content"][0]["type"] == "image"
    assert processor.call_args == (["<image> choose"], 1, "pt")


def test_serve_multimodal_encoder_expands_qwen_image_grid_tokens():
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP8z8BQDwAFgwJ/lwJw6QAAAABJRU5ErkJggg=="
    )

    class FakeProcessor:
        image_token_id = 99

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            del messages, tokenize, add_generation_prompt
            return "<image> describe"

        def __call__(self, *, text, images, return_tensors):
            del text, images, return_tensors
            return {
                "input_ids": torch.tensor([[1, 99, 2]]),
                "pixel_values": torch.zeros(256, 1536),
                "image_grid_thw": torch.tensor([[1, 16, 16]]),
            }

    tokens, features = serve_mod._encode_messages_with_features(
        SimpleNamespace(eos_token_id=1),
        FakeProcessor(),
        [
            serve_mod.ChatMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": "describe"},
                ],
            )
        ],
    )

    assert len(tokens) == 66
    assert tokens.count(99) == 64
    assert features["image_token_id"] == 99


def test_serve_multimodal_encoder_rejects_images_without_processor():
    messages = [serve_mod.ChatMessage(role="user", content=[{"type": "image", "image": "/tmp/missing.png"}])]

    with pytest.raises(ValueError, match="requires a checkpoint processor"):
        serve_mod._encode_messages_with_features(SimpleNamespace(), None, messages)


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
        self.calls.append((messages, dict(kwargs)))
        return [len(messages)]
