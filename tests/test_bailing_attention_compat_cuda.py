"""Execute migrated GQA/MLA projections and FLA against native CUDA."""

from __future__ import annotations

import importlib

import pytest
import torch
from torch import nn

from areno.adapters import LoraConfig
from areno.adapters.lora import initialize_lora
from areno.engine.config import ModelConfig
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native CUDA and FLA")


@pytest.fixture(autouse=True)
def _cuda_tp_context():
    previous = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cuda", 0), group=None))
    yield
    set_tp_context(previous)


@pytest.mark.parametrize("family", ("bailing", "bailing_v3"))
@pytest.mark.parametrize("mla", (False, True))
@pytest.mark.parametrize("factor", (0.5, 1.0))
def test_projection_fft_lora_parity_and_backward(family, mla, factor):
    module = importlib.import_module(f"areno.models.{family}.model")
    config = ModelConfig(
        hidden_size=128,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        kv_lora_rank=16 if mla else None,
        partial_rotary_factor=factor,
        dtype=torch.bfloat16,
        attn_backend="native",
        sequence_parallel=False,
    )
    policy = nn.Module()
    policy.config = config
    policy.attention = module.BailingSoftmaxAttention(config, 0).to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.normal_(std=0.05)
    hidden = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(32, device="cuda").unsqueeze(0)
    expected = tuple(t.detach() for t in policy.attention._project(hidden, positions))
    initialize_lora(policy, LoraConfig(rank=4, alpha=4, target_modules=("attention.q_proj",)), seed=7)
    actual = policy.attention._project(hidden, positions)
    for before, after in zip(expected, actual, strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    sum(t.float().square().mean() for t in actual).backward()
    grads = [p.grad for n, p in policy.named_parameters() if "lora_B" in n]
    assert grads and all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("family", ("bailing", "bailing_v3"))
@pytest.mark.parametrize("packed", (False, True))
def test_fla_052_training_call_backward(family, packed):
    module = importlib.import_module(f"areno.models.{family}.model")
    config = ModelConfig(
        hidden_size=128,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        dtype=torch.bfloat16,
        sequence_parallel=False,
    )
    attention = module.BailingLinearAttention(config, 0).to("cuda", dtype=torch.bfloat16)
    tensors = [torch.randn(1, 64, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    cu_seqlens = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32) if packed else None
    out = attention._forward_lightning(*tensors, cu_seqlens=cu_seqlens)
    assert out.shape == tensors[0].shape and torch.isfinite(out).all()
    out.float().square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in tensors)
