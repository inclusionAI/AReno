from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


class _Attention:
    pass


class _SoftmaxAttention(_Attention):
    pass


class _LinearAttention(_Attention):
    local_heads = 2
    head_dim = 4

    def set_state_cache(self, state: torch.Tensor) -> None:
        self.state_cache = state


class _KDAAttention(_Attention):
    local_heads = 2
    head_dim = 4
    v_head_dim = 6
    local_proj_dim = 8
    conv_kernel_size = 3

    def set_state_cache(self, state: torch.Tensor, conv_state: torch.Tensor) -> None:
        self.state_cache = state
        self.conv_state = conv_state


def _model_with(attention: _Attention) -> SimpleNamespace:
    parameter = torch.nn.Parameter(torch.zeros(1))
    return SimpleNamespace(
        layers=[SimpleNamespace(attention=attention)],
        parameters=lambda: iter((parameter,)),
    )


def test_bailing_linear_attention_respects_explicit_num_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("triton")
    from areno.models.bailing import model as bailing

    attention = _LinearAttention()
    model = _model_with(attention)
    monkeypatch.setattr(bailing, "BailingSoftmaxAttention", _SoftmaxAttention)
    monkeypatch.setattr(bailing, "BailingLinearAttention", _LinearAttention)

    bailing.BailingMoeLinearV2ForCausalLM.set_kv_caches(model, [], num_slots=5)

    assert tuple(attention.state_cache.shape) == (5, 2, 4, 4)


def test_bailing_v3_kda_attention_respects_explicit_num_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("triton")
    from areno.models.bailing_v3 import model as bailing_v3

    attention = _KDAAttention()
    model = _model_with(attention)
    monkeypatch.setattr(bailing_v3, "BailingSoftmaxAttention", _SoftmaxAttention)
    monkeypatch.setattr(bailing_v3, "BailingKDAAttention", _KDAAttention)

    bailing_v3.BailingMoeV3ForCausalLM.set_kv_caches(model, [], num_slots=7)

    assert tuple(attention.state_cache.shape) == (7, 2, 4, 6)
    assert tuple(attention.conv_state.shape) == (7, 3, 8, 2)
