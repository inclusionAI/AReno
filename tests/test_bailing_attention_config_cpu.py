"""Checkpoint head dimensions and backend-specific FLA call contracts."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from areno.accel.ops import chunk_lightning_attn
from areno.engine.config import ModelConfig
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context


@pytest.fixture(autouse=True)
def _tp_context():
    previous = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    yield
    set_tp_context(previous)


@pytest.mark.parametrize("family", ("bailing", "bailing_v3"))
@pytest.mark.parametrize(
    "rotary",
    (
        {},
        {"partial_rotary_factor": 0.5},
        {"rotary_dim": 4},
        {"qk_nope_head_dim": 4, "qk_rope_head_dim": 4},
        {"qk_nope_head_dim": 0, "qk_rope_head_dim": 8},
        {"partial_rotary_factor": 0.0},
    ),
)
def test_gqa_checkpoint_dimensions(family, rotary):
    module = importlib.import_module(f"areno.models.{family}.model")
    adapter = module.BailingMoeLinearV2Adapter() if family == "bailing" else module.BailingMoeV3Adapter()
    config = adapter.config_from_hf(
        dict(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
            vocab_size=32,
            intermediate_size=64,
            num_experts=2,
            **rotary,
        )
    )
    attention = module.BailingSoftmaxAttention(config, 0)
    rope_dim = rotary.get("qk_rope_head_dim", rotary.get("rotary_dim", int(8 * rotary.get("partial_rotary_factor", 1))))
    assert (attention.qk_nope_head_dim, attention.qk_rope_head_dim) == (8 - rope_dim, rope_dim)
    assert attention.head_dim == 8
    assert attention.query_key_value.weight.shape == (64, 32)


@pytest.mark.parametrize("family", ("bailing", "bailing_v3"))
@pytest.mark.parametrize("factor", (0.0, 0.5, 1.0))
def test_direct_gqa_defaults(family, factor):
    module = importlib.import_module(f"areno.models.{family}.model")
    attention = module.BailingSoftmaxAttention(
        ModelConfig(
            hidden_size=32,
            head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            partial_rotary_factor=factor,
            attn_backend="native",
        ),
        0,
    )
    assert attention.head_dim == 8
    assert attention.qk_rope_head_dim == int(8 * factor)


@pytest.mark.parametrize("family", ("bailing", "bailing_v3"))
@pytest.mark.parametrize(
    "split", ({}, {"qk_nope_head_dim": 12, "qk_rope_head_dim": 4}, {"qk_nope_head_dim": 0, "qk_rope_head_dim": 8})
)
def test_mla_preserves_independent_qk_dimensions(family, split):
    module = importlib.import_module(f"areno.models.{family}.model")
    adapter = module.BailingMoeLinearV2Adapter() if family == "bailing" else module.BailingMoeV3Adapter()
    config = adapter.config_from_hf(
        dict(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
            vocab_size=32,
            intermediate_size=64,
            num_experts=2,
            kv_lora_rank=4,
            v_head_dim=8,
            **split,
        )
    )
    attention = module.BailingSoftmaxAttention(config, 0)
    assert attention.query_key_value is None
    assert (attention.qk_nope_head_dim, attention.qk_rope_head_dim) == (
        split.get("qk_nope_head_dim", 8),
        split.get("qk_rope_head_dim", 8),
    )
    assert attention.kv_b_proj.weight.shape == (4 * (attention.qk_nope_head_dim + 8), 4)


@pytest.mark.parametrize("family", ("bailing", "bailing_v3"))
def test_inconsistent_gqa_dimensions_rejected(family):
    module = importlib.import_module(f"areno.models.{family}.model")
    with pytest.raises(ValueError, match="sum to checkpoint head_dim"):
        module.BailingSoftmaxAttention(
            ModelConfig(
                hidden_size=32,
                head_dim=8,
                num_attention_heads=4,
                num_key_value_heads=2,
                qk_nope_head_dim=8,
                qk_rope_head_dim=8,
            ),
            0,
        )


@pytest.mark.parametrize("head_first", (False, True))
def test_cuda_fla_removes_obsolete_layout_keyword(monkeypatch, head_first):
    def implementation(q, k, v, *, layer_idx, num_layers):
        assert q.shape == (1, 3, 2, 4)
        return q + k + v, None

    monkeypatch.setitem(sys.modules, "fla.ops.lightning_attn", SimpleNamespace(chunk_lightning_attn=implementation))
    q = torch.randn(1, 3, 2, 4)
    if head_first:
        q = q.transpose(1, 2)
    out, state = chunk_lightning_attn(q, q, q, head_first=head_first, layer_idx=0, num_layers=2)
    torch.testing.assert_close(out, 3 * q)
    assert state is None


def test_npu_fla_preserves_backend_arguments(monkeypatch):
    received = {}

    def implementation(q, k, v, **kwargs):
        received.update(kwargs)
        return q, None

    monkeypatch.setitem(sys.modules, "areno.accel.npu.seg_la", SimpleNamespace(chunk_lightning_attn=implementation))
    q = SimpleNamespace(device=SimpleNamespace(type="npu"))
    chunk_lightning_attn(q, q, q, head_first=False, layer_idx=0, num_layers=2)
    assert received == dict(head_first=False, layer_idx=0, num_layers=2)
