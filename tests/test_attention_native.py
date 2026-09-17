"""CUDA/NPU native attention contracts, compared with CPU references.

The NPU fixture selects compatibility kernels explicitly; production dispatch
and the fast library path have separate boundary and hardware tests.
"""

import importlib.util
import math

import pytest
import torch

from areno.accel._extension import extension
from areno.accel.attention import (
    areno_causal_attention,
    areno_paged_causal_attention_decode,
    areno_varlen_causal_attention,
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture(scope="module", params=["cuda", "npu"])
def backend(request):
    device = request.param
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA hardware and compiled kernels are required")
        yield device, extension(device)
        return
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Ascend hardware and compiled kernels are required")
    import torch_npu  # noqa: F401

    from areno.accel.npu import attention

    assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
    torch.npu.set_device(0)
    native = extension(device)
    assert native.attention_implementation == "ascendc"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(attention, "_flash_supported", lambda q: False)
        yield device, native


def values(shape, dtype, device, seed, *, strided=False, requires_grad=False):
    shape = tuple(shape)
    physical = (*shape[:-1], shape[-1] * 2) if strided else shape
    reference = (torch.randn(physical, generator=torch.Generator().manual_seed(seed)) / 3).to(dtype)
    # Both contiguous and strided inputs have a nonzero storage offset.
    count = math.prod(physical)
    data = torch.full((count + 16,), -77.0, dtype=dtype, device=device)
    tensor = data[8:-8].view(physical)
    tensor.copy_(reference)
    if strided:
        tensor, reference = tensor[..., ::2], reference[..., ::2]
    return tensor.requires_grad_(requires_grad), reference


def dense_reference(q, k, v, start, window, scale):
    q, k, v = q.double(), k.double(), v.double()
    qp = torch.arange(start, start + q.shape[-2])[:, None]
    kp = torch.arange(k.shape[-2])[None, :]
    mask = kp <= qp
    if window >= 0:
        mask &= kp >= qp - window
    probs = (q @ k.transpose(-1, -2) * scale).masked_fill(~mask, -torch.inf).softmax(-1)
    return probs @ v


def dense_gradients(q, k, v, grad, saved, start, window, scale):
    """CUDA's native derivative uses the storage-rounded saved forward output."""
    q, k, v, grad, saved = (x.double() for x in (q, k, v, grad, saved))
    qp = torch.arange(start, start + q.shape[-2])[:, None]
    kp = torch.arange(k.shape[-2])[None, :]
    mask = kp <= qp
    if window >= 0:
        mask &= kp >= qp - window
    p = (q @ k.transpose(-1, -2) * scale).masked_fill(~mask, -torch.inf).softmax(-1)
    ds = p * ((grad @ v.transpose(-1, -2)) - (grad * saved).sum(-1, keepdim=True)) * scale
    return ds @ k, ds.transpose(-1, -2) @ q, p.transpose(-1, -2) @ grad


def packed_reference(q, k, v, boundaries, window, scale):
    out = []
    repeat = q.shape[1] // k.shape[1]
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        out.append(
            dense_reference(
                q[begin:end].transpose(0, 1)[None],
                k[begin:end].repeat_interleave(repeat, dim=1).transpose(0, 1)[None],
                v[begin:end].repeat_interleave(repeat, dim=1).transpose(0, 1)[None],
                0,
                window,
                scale,
            )[0].transpose(0, 1)
        )
    return torch.cat(out, dim=0)


def close(actual, reference, storage=None):
    dtype = storage or actual.dtype
    atol, rtol = {torch.float32: (4e-5, 5e-4), torch.float16: (1e-3, 5e-3), torch.bfloat16: (8e-3, 4e-2)}[dtype]
    torch.testing.assert_close(actual.detach().cpu(), reference.to(actual.dtype), atol=atol, rtol=rtol)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "batch,heads,qlen,klen,dim,start,window",
    [
        (1, 1, 1, 1, 1, 0, -1),
        (2, 3, 7, 11, 31, 3, -1),
        (1, 2, 17, 19, 64, 2, 0),
        (2, 1, 5, 13, 257, 4, 2),
        (1, 2, 5, 7, 512, 2, 3),
        (1, 1, 4, 9, 513, 3, 17),
        (1, 2, 3, 7, 1025, 1, 1),
    ],
)
def test_dense_attention_forward_backward(backend, dtype, strided, batch, heads, qlen, klen, dim, start, window):
    device, _ = backend
    tensors = [
        values((batch, heads, length, dim), dtype, device, seed, strided=strided, requires_grad=True)
        for length, seed in ((qlen, 11), (klen, 12), (klen, 13))
    ]
    q, k, v = [pair[0] for pair in tensors]
    qr, kr, vr = [pair[1] for pair in tensors]
    scale = 0.125
    output = areno_causal_attention(q, k, v, query_start=start, window_left=window, softmax_scale=scale)
    expected = dense_reference(qr, kr, vr, start, window, scale)
    close(output, expected)
    grad, gr = values(output.shape, dtype, device, 14, strided=strided)
    output.backward(grad)
    # Feed the actual saved output to isolate native backward from the small
    # FP32 forward/reduction differences between implementations.
    gradients = dense_gradients(qr, kr, vr, gr, output.detach().cpu(), start, window, scale)
    for tensor, expected in zip((q, k, v), gradients, strict=True):
        close(tensor.grad, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "lengths,qheads,kvheads,dim,window",
    [
        ([1], 1, 1, 1, -1),
        ([0, 1, 0, 3, 7, 0], 4, 2, 31, -1),
        ([5, 17], 6, 2, 128, 2),
        ([0, 9, 1], 4, 1, 513, 0),
        ([3, 4, 0, 1], 3, 1, 1025, 3),
    ],
)
def test_packed_attention_gqa_empty_segments_and_gradients(backend, dtype, lengths, qheads, kvheads, dim, window):
    device, _ = backend
    tokens = sum(lengths)
    pairs = [
        values((tokens, heads, dim), dtype, device, seed, strided=True, requires_grad=True)
        for heads, seed in ((qheads, 21), (kvheads, 22), (kvheads, 23))
    ]
    q, k, v = [pair[0] for pair in pairs]
    refs = [pair[1].double().requires_grad_() for pair in pairs]
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    # Non-contiguous boundary metadata is normalized by the shared wrapper.
    physical = torch.stack((boundaries, torch.full_like(boundaries, -1)), dim=1).to(device)
    output = areno_varlen_causal_attention(q, k, v, physical[:, 0], window_left=window, softmax_scale=0.125)
    expected = packed_reference(*refs, boundaries.tolist(), window, 0.125)
    close(output, expected)
    grad, gr = values(output.shape, dtype, device, 24, strided=True)
    output.backward(grad)
    expected.backward(gr.double())
    for tensor, ref in zip((q, k, v), refs, strict=True):
        close(tensor.grad, ref.grad)


@pytest.mark.parametrize("dtype", DTYPES)
def test_native_attention_gradients_are_fp32_and_use_saved_output(backend, dtype):
    device, native = backend
    pairs = [values((1, 2, length, 33), dtype, device, seed) for length, seed in ((3, 31), (7, 32), (7, 33))]
    q, k, v = [pair[0] for pair in pairs]
    qr, kr, vr = [pair[1] for pair in pairs]
    saved = native.areno_causal_attention_forward(q, k, v, 2, 3, 0.125)
    # Deliberately perturb the saved tensor to catch recomputation that silently
    # changes CUDA's derivative contract for FP16/BF16 forward outputs.
    saved.add_(0.125)
    grad, gr = values(q.shape, dtype, device, 34)
    actual = native.areno_causal_attention_backward(grad, q, k, v, saved, 2, 3, 0.125)
    expected = dense_gradients(qr, kr, vr, gr, saved.cpu(), 2, 3, 0.125)
    for value, ref in zip(actual, expected, strict=True):
        assert value.dtype == torch.float32
        close(value, ref)


@pytest.mark.parametrize("scale", [0.0, -0.5, 40.0])
def test_attention_softmax_stability_and_nondefault_scale(backend, scale):
    device, _ = backend
    q, qr = values((1, 2, 5, 17), torch.float32, device, 35)
    k, kr = values((1, 2, 9, 17), torch.float32, device, 36)
    v, vr = values((1, 2, 9, 17), torch.float32, device, 37)
    q.mul_(100)
    k.mul_(100)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    output = areno_causal_attention(q, k, v, query_start=2, softmax_scale=scale)
    close(output, dense_reference(qr * 100, kr * 100, vr, 2, -1, scale))
    output.sum().backward()
    expected = dense_gradients(qr * 100, kr * 100, vr, torch.ones_like(qr), output.detach().cpu(), 2, -1, scale)
    for tensor, reference in zip((q, k, v), expected, strict=True):
        close(tensor.grad, reference)


def paged_data(device, dtype, dim, qheads=4, kvheads=2):
    q, qr = values((3, qheads, dim), dtype, device, 41, strided=True)
    ku, kur = values((3, kvheads, dim), dtype, device, 42, strided=True)
    vu, vur = values((3, kvheads, dim), dtype, device, 43, strided=True)
    kc, kcr = values((10, 4, kvheads, dim), dtype, device, 44)
    vc, vcr = values((10, 4, kvheads, dim), dtype, device, 45)
    table = torch.tensor([[2, 8, 0], [1, 3, 9], [7, 6, 4]], dtype=torch.int32)
    lengths = torch.tensor([0, 4, 10], dtype=torch.int32)
    for b, pos in enumerate(lengths.tolist()):
        block, offset = table[b, pos // 4], pos % 4
        kcr[block, offset] = kur[b]
        vcr[block, offset] = vur[b]
    args = (q, ku, vu, kc, vc, table.to(device), lengths.to(device))
    return args, (qr, kcr, vcr, table, lengths)


def paged_reference(q, kc, vc, table, lengths, window, scale):
    out = []
    for b, before in enumerate(lengths.tolist()):
        positions = torch.arange(before + 1)
        blocks, offsets = table[b, positions // kc.shape[1]].long(), positions % kc.shape[1]
        k = kc[blocks, offsets].repeat_interleave(q.shape[1] // kc.shape[2], dim=1).transpose(0, 1)[None]
        v = vc[blocks, offsets].repeat_interleave(q.shape[1] // vc.shape[2], dim=1).transpose(0, 1)[None]
        out.append(dense_reference(q[b : b + 1, :, None], k, v, before, window, scale).squeeze(2))
    return torch.cat(out, dim=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "dim,window,splits", [(1, -1, 1), (31, -1, 8), (128, 2, 3), (257, 0, 32), (513, 7, 4), (1025, 2, 17)]
)
def test_paged_decode_updates_cache_and_merges_empty_splits(backend, dtype, dim, window, splits):
    device, _ = backend
    args, refs = paged_data(device, dtype, dim)
    q, ku, vu, kc, vc, table, lengths = args
    result = areno_paged_causal_attention_decode(*args, window_left=window, num_splits=splits, softmax_scale=0.125)
    close(result, paged_reference(*refs, window, 0.125))
    torch.testing.assert_close(kc.cpu(), refs[1], atol=0, rtol=0)
    torch.testing.assert_close(vc.cpu(), refs[2], atol=0, rtol=0)
    torch.testing.assert_close(table.cpu(), refs[3], atol=0, rtol=0)
    torch.testing.assert_close(lengths.cpu(), refs[4], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_paged_cache_copies_preserve_bits_and_outer_canaries(backend, dtype):
    device, _ = backend
    q = torch.ones(1, 2, 4, dtype=dtype, device=device)
    update_cpu = torch.tensor([[[0.0, -0.0, torch.inf, torch.nan]]], dtype=dtype)
    update = update_cpu.to(device)
    k_raw = torch.full((4, 2, 1, 4), -77.0, dtype=dtype, device=device)
    v_raw = torch.full_like(k_raw, -88.0)
    kc, vc = k_raw[1:-1], v_raw[1:-1]
    areno_paged_causal_attention_decode(
        q,
        update,
        update,
        kc,
        vc,
        torch.tensor([[1]], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
    )
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    for raw, cache, canary in ((k_raw, kc, -77.0), (v_raw, vc, -88.0)):
        torch.testing.assert_close(cache[1, 0].cpu().view(bits), update_cpu[0].view(bits), atol=0, rtol=0)
        assert torch.all(raw[[0, -1]].cpu() == canary)
        assert torch.all(cache[0].cpu() == canary)
        assert torch.all(cache[1, 1].cpu() == canary)


@pytest.mark.parametrize("dtype", DTYPES)
def test_prefill_and_incremental_paged_decode_agree(backend, dtype):
    device, _ = backend
    q, qr = values((5, 4, 33), dtype, device, 51)
    k, kr = values((5, 2, 33), dtype, device, 52)
    v, vr = values((5, 2, 33), dtype, device, 53)
    boundaries = torch.tensor([0, 5], dtype=torch.int32, device=device)
    for window in (-1, 2):
        full = areno_varlen_causal_attention(q, k, v, boundaries, window_left=window)
        kc = torch.zeros(3, 2, 2, 33, dtype=dtype, device=device)
        vc = torch.zeros_like(kc)
        table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device=device)
        for pos in range(5):
            result = areno_paged_causal_attention_decode(
                q[pos : pos + 1],
                k[pos : pos + 1],
                v[pos : pos + 1],
                kc,
                vc,
                table,
                torch.tensor([pos], dtype=torch.int32, device=device),
                window_left=window,
                num_splits=8,
            )
            close(result[0], full[pos].cpu())


def test_attention_current_stream_and_tensor_device(backend):
    device, _ = backend
    api = getattr(torch, device)
    target = torch.device(device, 1 if api.device_count() > 1 else 0)
    q = torch.empty(1, 2, 3, 33, device=target, requires_grad=True)
    k = torch.empty_like(q, requires_grad=True)
    v = torch.empty_like(q, requires_grad=True)
    stream = api.Stream(device=target)
    stream.wait_stream(api.current_stream(target))
    with api.stream(stream):
        with torch.no_grad():
            q.fill_(0.125)
            k.fill_(0.25)
            v.fill_(0.5)
        result = areno_causal_attention(q, k, v)
        result.sum().backward()
        packed_inputs = [tensor.detach()[0].transpose(0, 1) for tensor in (q, k, v)]
        packed = areno_varlen_causal_attention(*packed_inputs, torch.tensor([0, 3], dtype=torch.int32, device=target))
        kc = torch.empty(1, 2, 2, 33, device=target)
        vc = torch.empty_like(kc)
        paged = areno_paged_causal_attention_decode(
            q.detach()[:, :, 0],
            k.detach()[:, :, 0],
            v.detach()[:, :, 0],
            kc,
            vc,
            torch.zeros(1, 1, dtype=torch.int32, device=target),
            torch.zeros(1, dtype=torch.int32, device=target),
        )
    stream.synchronize()
    assert result.device == target
    assert packed.device == paged.device == target
    close(result, torch.full((1, 2, 3, 33), 0.5))
    close(packed, torch.full((3, 2, 33), 0.5))
    close(paged, torch.full((1, 2, 33), 0.5))
    close(q.grad, torch.zeros(1, 2, 3, 33))
    close(k.grad, torch.zeros(1, 2, 3, 33))
    close(v.grad, torch.tensor([1 + 1 / 2 + 1 / 3, 1 / 2 + 1 / 3, 1 / 3])[None, None, :, None].expand(1, 2, 3, 33))


def test_attention_device_graph_replay(backend):
    device, native = backend
    api = getattr(torch, device)
    pairs = [values((1, 2, length, 17), torch.float32, device, seed) for length, seed in ((3, 61), (7, 62), (7, 63))]
    q, k, v = [pair[0] for pair in pairs]
    qr, kr, vr = [pair[1] for pair in pairs]
    grad = torch.ones_like(q)

    def run():
        out = areno_causal_attention(q, k, v, query_start=2, window_left=2, softmax_scale=0.125)
        return out, *native.areno_causal_attention_backward(grad, q, k, v, out, 2, 2, 0.125)

    stream = api.Stream()
    stream.wait_stream(api.current_stream())
    with api.stream(stream):
        for _ in range(3):
            run()
    api.current_stream().wait_stream(stream)
    graph = api.CUDAGraph() if device == "cuda" else api.NPUGraph()
    with api.graph(graph, stream=stream):
        outputs = run()
    for multiplier in (1, -1, 2):
        q.copy_(qr * multiplier)
        graph.replay()
        close(outputs[0], dense_reference(qr * multiplier, kr, vr, 2, 2, 0.125))
        expected = dense_gradients(qr * multiplier, kr, vr, torch.ones_like(qr), outputs[0].cpu(), 2, 2, 0.125)
        for actual, ref in zip(outputs[1:], expected, strict=True):
            close(actual, ref)
