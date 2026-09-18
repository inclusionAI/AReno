"""Numerical acceptance of NPU attention, including native availability fallback."""

import importlib.util

import pytest
import torch

from areno.accel.attention import (
    areno_causal_attention,
    areno_paged_causal_attention_decode,
    areno_varlen_causal_attention,
)
from tests.test_attention_native import close, dense_reference, packed_reference, paged_reference, values


@pytest.fixture(scope="module")
def npu():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Ascend hardware and torch_npu are required")
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
    torch.npu.set_device(0)
    return "npu:0"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim,start,window", [(16, 0, -1), (64, 3, -1), (128, 2, 3), (256, 3, 0)])
def test_npu_dense_forward_and_backward(npu, dtype, dim, start, window):
    pairs = [
        values((2, 3, length, dim), dtype, npu, seed, strided=True, requires_grad=True)
        for length, seed in ((7, 11), (13, 12), (13, 13))
    ]
    q, k, v = [x[0] for x in pairs]
    refs = [x[1].double().requires_grad_() for x in pairs]
    out = areno_causal_attention(q, k, v, query_start=start, window_left=window, softmax_scale=0.125)
    expected = dense_reference(*refs, start, window, 0.125)
    close(out, expected)
    grad, cpu_grad = values(out.shape, dtype, npu, 14)
    out.backward(grad)
    expected.backward(cpu_grad.double())
    for tensor, ref in zip((q, k, v), refs, strict=True):
        close(tensor.grad, ref.grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("window", [-1, 2])
def test_npu_packed_gqa_empty_segments_and_backward(npu, dtype, window):
    pairs = [
        values((9, heads, 64), dtype, npu, seed, strided=True, requires_grad=True)
        for heads, seed in ((4, 21), (2, 22), (2, 23))
    ]
    q, k, v = [x[0] for x in pairs]
    refs = [x[1].double().requires_grad_() for x in pairs]
    boundaries = [0, 0, 1, 4, 4, 9]
    out = areno_varlen_causal_attention(
        q,
        k,
        v,
        torch.tensor(boundaries, dtype=torch.int32, device=npu),
        window_left=window,
        softmax_scale=0.125,
    )
    expected = packed_reference(*refs, boundaries, window, 0.125)
    close(out, expected)
    grad, cpu_grad = values(out.shape, dtype, npu, 24)
    out.backward(grad)
    expected.backward(cpu_grad.double())
    for tensor, ref in zip((q, k, v), refs, strict=True):
        close(tensor.grad, ref.grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("window", [-1, 7])
def test_npu_paged_decode_updates_original_cache(npu, dtype, window):
    q, qr = values((3, 4, 64), dtype, npu, 31, strided=True)
    ku, kur = values((3, 2, 64), dtype, npu, 32, strided=True)
    vu, vur = values((3, 2, 64), dtype, npu, 33, strided=True)
    kc, kcr = values((9, 256, 2, 64), dtype, npu, 34)
    vc, vcr = values((9, 256, 2, 64), dtype, npu, 35)
    table = torch.tensor([[2, 8], [1, 3], [7, 6]], dtype=torch.int32)
    lengths = torch.tensor([0, 256, 270], dtype=torch.int32)
    for row, position in enumerate(lengths.tolist()):
        block, offset = table[row, position // 256], position % 256
        kcr[block, offset], vcr[block, offset] = kur[row], vur[row]
    out = areno_paged_causal_attention_decode(
        q,
        ku,
        vu,
        kc,
        vc,
        table.to(npu),
        lengths.to(npu),
        window_left=window,
        num_splits=1,
        softmax_scale=0.125,
    )
    close(out, paged_reference(qr, kcr, vcr, table, lengths, window, 0.125))
    torch.testing.assert_close(kc.cpu(), kcr, atol=0, rtol=0)
    torch.testing.assert_close(vc.cpu(), vcr, atol=0, rtol=0)


def test_npu_nondefault_stream_and_device(npu):
    device = torch.device("npu", 1 if torch.npu.device_count() > 1 else 0)
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(stream):
        q = torch.full((1, 2, 3, 64), 0.25, dtype=torch.bfloat16, device=device, requires_grad=True)
        v = torch.full_like(q, 0.5, requires_grad=True)
        out = areno_causal_attention(q, q, v)
        out.sum().backward()
    stream.synchronize()
    assert out.device == device and v.grad is not None
    close(out, torch.full(out.shape, 0.5))
