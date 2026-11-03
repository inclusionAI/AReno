"""GPT-OSS config, quantization and mathematical contracts (CPU safe)."""

import pytest
import torch

from areno.accel.grouped_bias import limited_swiglu
from areno.accel.sink_attention import sink_attention
from areno.engine.checkpoints.mxfp4 import dequantize_mxfp4
from areno.models.gpt_oss import GptOssAdapter


def tiny_config():
    return dict(
        model_type="gpt_oss",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        torch_dtype="float32",
        sliding_window=4,
        rope_theta=150000.0,
        rope_scaling=None,
    )


def test_config_cpu():
    config = GptOssAdapter().config_from_hf(tiny_config())
    assert config.model_type == "gpt_oss" and config.layer_types == ("sliding_attention", "full_attention")
    assert not config.sequence_parallel and not config.qk_norm and config.qkv_bias
    assert config.num_experts == 4 and config.num_experts_per_tok == 2
    with pytest.raises(ValueError):
        GptOssAdapter().config_from_hf({**tiny_config(), "rope_scaling": {"rope_type": "unknown"}})


def test_lazy_registration_cpu():
    from areno.models.registry import build_model

    model = build_model(GptOssAdapter().config_from_hf(tiny_config()))
    assert len(model.layers) == 2


def test_mxfp4_cpu():
    # Every E2M1 code, low nibble before high nibble; repeated block.
    blocks = torch.arange(16, dtype=torch.uint8).mul(17).view(1, 1, 1, 16)
    scales = torch.full((1, 1, 1), 128, dtype=torch.uint8)
    actual = dequantize_mxfp4(blocks, scales, torch.float32)
    expected = (
        torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        ).repeat_interleave(2)
        * 2
    )
    torch.testing.assert_close(actual.flatten(), expected, atol=0, rtol=0)
    with pytest.raises(ValueError):
        dequantize_mxfp4(blocks, scales.float())


def dense_reference(q, k, v, sinks, sequence_ids=None, window=None):
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], 1)
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    n = q.shape[-2]
    row = torch.arange(n, device=q.device)[:, None]
    col = torch.arange(n, device=q.device)[None, :]
    valid = col <= row
    if window is not None:
        valid = valid & (col > row - window)
    if sequence_ids is not None:
        valid = valid & (sequence_ids[:, None] == sequence_ids[None, :])
    scores = scores.masked_fill(~valid, -float("inf"))
    combined = torch.cat((scores, sinks.view(1, -1, 1, 1).expand(q.shape[0], -1, n, -1)), -1)
    probabilities = combined.softmax(-1)[..., :-1]
    return (probabilities @ v).transpose(1, 2)


@pytest.mark.parametrize("window", [None, 2])
def test_sink_forward_and_gradient_cpu(window):
    torch.manual_seed(7)
    q = torch.randn(1, 6, 4, 16, requires_grad=True)
    k = torch.randn(1, 6, 2, 16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    sinks = torch.randn(4, requires_grad=True)
    seq = torch.tensor([0, 0, 0, 1, 1, 1])
    inputs = (q, k, v, sinks)
    actual = sink_attention(*inputs, sequence_ids=seq, window=window)
    expected = dense_reference(*inputs, sequence_ids=seq, window=window)
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=2e-6)
    upstream = torch.randn_like(actual)
    a = torch.autograd.grad(actual, inputs, upstream, retain_graph=True)
    b = torch.autograd.grad(expected, inputs, upstream)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_limited_swiglu_cpu():
    x = torch.tensor([[-10.0, -20.0, 10.0, 20.0, 1.0, 2.0]], requires_grad=True)
    out = limited_swiglu(x)
    gate = x[:, ::2].clamp(max=7)
    up = x[:, 1::2].clamp(-7, 7)
    torch.testing.assert_close(out, (up + 1) * gate * torch.sigmoid(gate * 1.702))
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_harmony_prompt_is_not_wrapped_twice_cpu():
    from areno.api.tokenizer import encode_generation_prompt

    class Tokenizer:
        chat_template = "harmony"

        def encode(self, text):
            assert text.startswith("<|start|>")
            return [1, 2, 3]

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("preformatted Harmony must not be wrapped twice")

    assert encode_generation_prompt(Tokenizer(), "<|start|>user<|message|>Hi<|end|>") == [1, 2, 3]
