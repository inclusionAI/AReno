"""Library-boundary tests; numerical Ascend acceptance runs on the target node."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from areno.accel.npu import attention, kda, seg_la
from areno.accel.ops import SegLaMeta


def test_flash_imports_are_device_specific_and_preserve_dependency_errors(monkeypatch):
    loader = importlib.import_module("areno.accel.flash_attention")
    calls = []

    def load(name):
        calls.append(name)
        return name

    monkeypatch.setattr(loader, "import_module", load)
    assert loader.flash_attention(SimpleNamespace(type="cuda")) == "flash_attn"
    assert loader.flash_attention(SimpleNamespace(type="npu")) == "flash_attn_npu"
    assert calls == ["flash_attn", "flash_attn_npu"]
    for missing in ("flash_attn_npu", "torch_npu", "flash_attn_2_npu"):

        def fail(name):
            raise ModuleNotFoundError(name=missing)

        monkeypatch.setattr(loader, "import_module", fail)
        error = RuntimeError if missing == "flash_attn_npu" else ModuleNotFoundError
        with pytest.raises(error):
            loader.flash_attention(SimpleNamespace(type="npu"))


def test_dense_offset_layout_and_library_autograd(monkeypatch):
    calls = []

    def flash(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return q + k[:, -q.shape[1] :] + v[:, -q.shape[1] :]

    monkeypatch.setattr(attention, "flash_attention", lambda device: SimpleNamespace(flash_attn_func=flash))
    q = torch.randn(2, 3, 4, 16, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 3, 10, 16, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    result = attention.causal_attention(q, k, v, 2, 3, 0.25)
    torch.testing.assert_close(result, q + k[:, :, 2:6] + v[:, :, 2:6])
    result.sum().backward()
    assert q.grad.eq(1).all()
    for tensor in (k, v):
        assert tensor.grad[:, :, 2:6].eq(1).all()
        assert tensor.grad[:, :, :2].eq(0).all() and tensor.grad[:, :, 6:].eq(0).all()
    assert calls[0][0].shape == (2, 4, 3, 16)
    assert calls[0][1].shape == (2, 6, 3, 16)
    assert calls[0][3] == dict(causal=True, window_size=(3, 0), softmax_scale=0.25)


def test_packed_keeps_device_boundaries_and_gqa(monkeypatch):
    calls = []

    def flash(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return q

    monkeypatch.setattr(attention, "flash_attention", lambda device: SimpleNamespace(flash_attn_varlen_func=flash))
    q = torch.empty(11, 4, 16, dtype=torch.float16)
    k = torch.empty(11, 2, 16, dtype=torch.float16)
    cu = torch.tensor([0, 0, 3, 11, 11], dtype=torch.int32)
    assert attention.varlen_causal_attention(q, k, k, cu, -1, 0.25) is q
    args = calls[0][3]
    assert args["cu_seqlens_q"] is cu and args["cu_seqlens_k"] is cu
    assert args["max_seqlen_q"] >= 8 and args["max_seqlen_k"] >= 8
    assert calls[0][1].shape[1] == 2 and args["window_size"] == (-1, -1)


def test_paged_keeps_cache_identity_and_forwards_updates(monkeypatch):
    calls = []

    def flash(q, kc, vc, **kwargs):
        calls.append((kc, vc, kwargs))
        return q

    monkeypatch.setattr(attention, "flash_attention", lambda device: SimpleNamespace(flash_attn_with_kvcache=flash))
    q = torch.empty(2, 4, 16, dtype=torch.bfloat16)
    update = torch.empty(2, 2, 16, dtype=torch.bfloat16)
    kc = torch.empty(4, 256, 2, 16, dtype=torch.bfloat16)
    vc = torch.empty_like(kc)
    table, lengths = torch.tensor([[1], [3]], dtype=torch.int32), torch.tensor([3, 8], dtype=torch.int32)
    result = attention.paged_causal_attention_decode(q, update, update, kc, vc, table, lengths, 7, 1, 0.25)
    assert result.shape == q.shape
    assert calls[0][0] is kc and calls[0][1] is vc
    kwargs = calls[0][2]
    assert kwargs["cache_seqlens"] is lengths and kwargs["block_table"] is table
    assert kwargs["k"].shape == (2, 1, 2, 16)
    assert kwargs["window_size"] == (7, 0) and kwargs["num_splits"] == 1
    with pytest.raises(RuntimeError, match="inference-only"):
        attention.paged_causal_attention_decode(q.requires_grad_(), update, update, kc, vc, table, lengths, 7, 1, 0.25)


def test_unsupported_attention_does_not_silently_use_math():
    q = torch.ones(1, 1, 1, 16)
    with pytest.raises(ValueError, match="FP16/BF16"):
        attention.causal_attention(q, q, q, 0, -1, 1.0)
    q = torch.ones(1, 1, 1, 512, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="256"):
        attention.causal_attention(q, q, q, 0, -1, 1.0)


def test_empty_attention_preserves_zero_gradients():
    q = torch.empty(0, 4, 16, dtype=torch.bfloat16, requires_grad=True)
    k = torch.empty(0, 2, 16, dtype=torch.bfloat16, requires_grad=True)
    v = torch.empty_like(k, requires_grad=True)
    out = attention.varlen_causal_attention(q, k, v, torch.zeros(3, dtype=torch.int32), -1, 1.0)
    out.sum().backward()
    assert all(x.grad is not None and x.grad.numel() == 0 for x in (q, k, v))


def test_kda_reuses_training_wrapper_and_maps_decode_state(monkeypatch):
    from areno.accel.kernels.kda_fla.kda import chunk_kda

    assert kda.chunk_kda is chunk_kda
    state = torch.arange(4 * 2 * 3 * 5, dtype=torch.float32).reshape(4, 2, 3, 5)
    before = state.clone()
    indices = torch.tensor([3, 1], dtype=torch.int32)
    q = torch.zeros(1, 2, 2, 5, dtype=torch.bfloat16)
    v = torch.zeros(1, 2, 2, 3, dtype=torch.bfloat16)
    gate, beta = torch.zeros(1, 2, 10), torch.zeros(1, 2, 2)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32)
    calls = []

    def recurrent(**kwargs):
        calls.append(kwargs)
        return v, kwargs["initial_state"] + 10

    monkeypatch.setitem(sys.modules, "fla.ops.kda", SimpleNamespace(fused_recurrent_kda=recurrent))
    result = kda.fused_sigmoid_gating_delta_rule_update(
        q=q,
        k=q,
        v=v,
        a=gate,
        b=beta,
        A_log=torch.ones(2),
        dt_bias=torch.ones(10),
        softplus_beta=1.0,
        softplus_threshold=20.0,
        initial_state_source=state,
        initial_state_indices=indices,
        scale=0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        is_kda=True,
        lower_bound=-6.0,
    )
    assert result is v
    assert calls[0]["state_v_first"] and calls[0]["use_beta_sigmoid_in_kernel"]
    assert calls[0]["use_gate_in_kernel"] and calls[0]["g"].shape == (1, 2, 2, 5)
    assert calls[0]["cu_seqlens"] is cu
    torch.testing.assert_close(state[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(state[indices.long()], before[indices.long()] + 10)


@pytest.mark.parametrize("decode", [False, True])
def test_seg_la_passes_state_and_decay_to_fla(monkeypatch, decode):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return args[0], kwargs["initial_state"] + 1

    monkeypatch.setitem(
        sys.modules,
        "fla.ops.simple_gla",
        SimpleNamespace(
            chunk_simple_gla=run,
            fused_recurrent_simple_gla=run,
        ),
    )
    state = torch.full((3, 2, 4, 4), 7.0)
    q = torch.zeros(2 if decode else 5, 2, 4)
    rates = torch.tensor([0.1, 0.2])
    meta = SegLaMeta(2, 3, torch.tensor([0, 2, 5]), torch.tensor([2, 0]), torch.tensor([2, 3]), torch.tensor([0, 1]))
    out = seg_la.seg_la_fwd(q, q, q, state, rates, meta, softmax_scale=0.5)
    assert out.shape == q.shape
    torch.testing.assert_close(calls[0]["g_gamma"], -rates)
    assert state[1].eq(7).all() and state[0].eq(8).all()
    assert state[2].eq(8 if decode else 1).all()
    if decode:
        assert calls[0]["cu_seqlens"] is None
    else:
        assert calls[0]["cu_seqlens"].tolist() == [0, 2, 5]
    with pytest.raises(NotImplementedError, match="snapshots"):
        seg_la.seg_la_fwd(q, q, q, state, rates, meta, caches=torch.empty(0))
