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


def test_unsupported_dtype_does_not_silently_use_math():
    q = torch.ones(1, 1, 1, 16, dtype=torch.float64)
    with pytest.raises(ValueError, match="FP32/FP16/BF16"):
        attention.causal_attention(q, q, q, 0, -1, 1.0)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype,dim", [(torch.float32, 64), (torch.bfloat16, 512), (torch.float16, 1025)])
def test_native_attention_reuses_shared_autograd_and_preserves_errors(monkeypatch, packed, dtype, dim):
    from areno.accel import attention as shared

    calls = []

    def forward(q, k, v, *args):
        calls.append(("forward", args))
        assert all(t.is_contiguous() for t in (q, k, v))
        return 2 * q + 3 * k + 4 * v

    def backward(grad, q, k, v, saved, *args):
        calls.append(("backward", args))
        torch.testing.assert_close(saved, 2 * q + 3 * k + 4 * v)
        return tuple(grad.float() * factor for factor in (2, 3, 4))

    def unexpected_library(*args):
        pytest.fail("out-of-range attention must not import flash-attn-npu")

    name = "areno_varlen_causal_attention" if packed else "areno_causal_attention"
    native = SimpleNamespace(**{name + "_forward": forward, name + "_backward": backward})
    monkeypatch.setattr(attention, "flash_attention", unexpected_library)
    monkeypatch.setattr(shared, "_extension", lambda device: native)
    shape = (3, 2, dim) if packed else (1, 2, 3, dim)
    tensors = [torch.randn(*shape, 2, dtype=dtype)[..., 0].requires_grad_() for _ in range(3)]
    if packed:
        cu = torch.tensor([0, 1, 3], dtype=torch.int32)
        result = attention.varlen_causal_attention(*tensors, cu, 2, 0.125)
    else:
        result = attention.causal_attention(*tensors, 0, 2, 0.125)
    result.sum().backward()
    assert [call[0] for call in calls] == ["forward", "backward"]
    assert calls[0][1][-2:] == calls[1][1][-2:] == (2, 0.125)
    for tensor, factor in zip(tensors, (2, 3, 4), strict=True):
        assert tensor.grad.dtype == dtype
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, factor))

    def missing(device):
        raise RuntimeError("native extension missing")

    monkeypatch.setattr(shared, "_extension", missing)
    with pytest.raises(RuntimeError, match="native extension missing"):
        if packed:
            attention.varlen_causal_attention(*tensors, cu, 2, 0.125)
        else:
            attention.causal_attention(*tensors, 0, 2, 0.125)


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


@pytest.mark.parametrize("device", ["cuda", "npu"])
def test_lightning_dispatch_preserves_original_arguments(monkeypatch, device):
    from areno.accel import ops

    q = SimpleNamespace(device=SimpleNamespace(type=device))
    k, v, decay, cu = object(), object(), object(), object()
    result = object()
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    def unexpected(*args, **kwargs):
        pytest.fail("Lightning Attention imported the wrong device implementation")

    monkeypatch.setitem(
        sys.modules,
        "fla.ops.lightning_attn",
        SimpleNamespace(chunk_lightning_attn=run if device == "cuda" else unexpected),
    )
    monkeypatch.setattr(seg_la, "chunk_lightning_attn", run if device == "npu" else unexpected)
    kwargs = dict(g_gamma=decay, head_first=False, cu_seqlens=cu)
    assert ops.chunk_lightning_attn(q, k, v, 3, 12, **kwargs) is result
    assert calls == [((q, k, v, 3, 12), kwargs)]


@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_lightning_preserves_explicit_decay_layout_and_autograd(monkeypatch, head_first, packed):
    shape = (1, 7, 2, 4) if packed else (3, 7, 2, 4)
    leaves = [torch.randn(shape, requires_grad=True) for _ in range(3)]
    q, k, v = [x.transpose(1, 2) if head_first else x for x in leaves]
    state = torch.randn(3, 2, 4, 4, requires_grad=True)
    decay = torch.tensor([-0.01, -0.003])
    cu = torch.tensor([0, 2, 5, 7]) if packed else None
    calls = []

    def simple_gla(**kwargs):
        calls.append(kwargs)
        assert "head_first" not in kwargs and "layer_idx" not in kwargs and "num_layers" not in kwargs
        return kwargs["q"] + 2 * kwargs["k"] + 3 * kwargs["v"], 2 * kwargs["initial_state"]

    monkeypatch.setitem(sys.modules, "fla.ops.simple_gla", SimpleNamespace(chunk_simple_gla=simple_gla))
    out, final = seg_la.chunk_lightning_attn(
        q,
        k,
        v,
        3,
        12,
        scale=0.125,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
        g_gamma=decay,
        head_first=head_first,
        chunk_size=64,
    )
    torch.testing.assert_close(out, q + 2 * k + 3 * v)
    assert calls[0]["g_gamma"] is decay and calls[0]["initial_state"] is state
    assert calls[0]["cu_seqlens"] is cu and calls[0]["output_final_state"]
    assert calls[0]["scale"] == 0.125 and calls[0]["chunk_size"] == 64
    assert calls[0]["q"].shape == shape
    (out.sum() + final.sum()).backward()
    for leaf, expected in zip(leaves, (1, 2, 3), strict=True):
        torch.testing.assert_close(leaf.grad, torch.full_like(leaf, expected))
    torch.testing.assert_close(state.grad, torch.full_like(state, 2))


def test_lightning_without_explicit_decay_delegates_to_upstream(monkeypatch):
    q = torch.ones(1, 3, 2, 4)
    calls = []

    def lightning(**kwargs):
        calls.append(kwargs)
        return kwargs["q"], None

    monkeypatch.setitem(sys.modules, "fla.ops.lightning_attn", SimpleNamespace(chunk_lightning_attn=lightning))
    out, final = seg_la.chunk_lightning_attn(q, q, q, 3, 12, head_first=False)
    assert out is q and final is None
    assert calls[0]["layer_idx"] == 3 and calls[0]["num_layers"] == 12
    assert "head_first" not in calls[0] and "g_gamma" not in calls[0]

    def fail(**kwargs):
        raise RuntimeError("FLA kernel unavailable")

    monkeypatch.setitem(sys.modules, "fla.ops.simple_gla", SimpleNamespace(chunk_simple_gla=fail))
    with pytest.raises(RuntimeError, match="FLA kernel unavailable"):
        seg_la.chunk_lightning_attn(q, q, q, 3, 12, g_gamma=torch.ones(2))


@pytest.mark.parametrize("family", ["bailing", "bailing_v3"])
@pytest.mark.parametrize("packed", [False, True])
def test_bailing_training_forwards_tp_slopes_and_packed_boundaries(monkeypatch, family, packed):
    from areno.accel import ops

    model = importlib.import_module(f"areno.models.{family}.model")
    assert model.chunk_lightning_attn is ops.chunk_lightning_attn
    layer = model.BailingLinearAttention.__new__(model.BailingLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_idx, layer.num_layers = 3, 12
    layer.slope = model._build_slope_tensor(2, 8, 3, 12, 2, 4)
    full_slopes = model._build_slope_tensor(8, 8, 3, 12, 0, 1)
    torch.testing.assert_close(layer.slope, full_slopes[4:6])
    q = torch.randn(1, 7, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(q.shape, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    cu = torch.tensor([0, 2, 5, 7], dtype=torch.int32)
    calls = []

    def lightning(q, k, v, **kwargs):
        calls.append(kwargs)
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        return q + k + v, None

    monkeypatch.setitem(sys.modules, "fla.ops.lightning_attn", SimpleNamespace(chunk_lightning_attn=lightning))
    meta = SimpleNamespace(packed=True, cu_seqlens=cu) if packed else None
    layer._forward_train(q, k, v, meta).sum().backward()
    torch.testing.assert_close(calls[0]["g_gamma"], -full_slopes[4:6])
    assert calls[0]["cu_seqlens"] is (cu if packed else None)
    assert calls[0]["head_first"] is False
    for tensor in (q, k, v):
        torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))


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
