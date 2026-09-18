"""The same routing contract for CUDA and native Ascend C kernels."""

import importlib.util

import pytest
import torch

from areno.accel._extension import extension
from areno.accel.router import areno_grouped_topk_router
from areno.accel.topk import areno_topk_softmax

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture(scope="module", params=["cuda", "npu"])
def backend(request):
    device = request.param
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA hardware and the compiled extension are required")
    else:
        if importlib.util.find_spec("torch_npu") is None:
            pytest.skip("Ascend torch_npu and hardware are required")
        import torch_npu  # noqa: F401

        assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
        torch.npu.set_device(0)
        assert extension(device).routing_implementation == "ascendc"
    return device, extension(device)


def values(shape, dtype, device, *, strided=False, seed=37):
    physical = (*shape[:-1], shape[-1] * 2) if strided else shape
    cpu = (torch.randn(physical, generator=torch.Generator().manual_seed(seed)) * 2).to(dtype)
    tensor = cpu.to(device)
    if strided:
        tensor, cpu = tensor[..., ::2], cpu[..., ::2]
    return tensor, cpu


def softmax_reference(x, k, renormalize, gradient=None, indices=None):
    logits = x.detach().float().requires_grad_()
    probs = logits.softmax(-1)
    if indices is None:
        indices = probs.argsort(dim=-1, descending=True, stable=True)[..., :k]
    weight = probs.gather(1, indices)
    if renormalize:
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-20)
    dx = None if gradient is None else torch.autograd.grad(weight, logits, gradient)[0].to(x.dtype)
    return indices, weight.detach(), dx


def grouped_reference(x, bias, k, groups, topk_group):
    score = x.float().sigmoid()
    route = score + bias
    # Stable sorting gives the CUDA tie rule: lower group/expert index first.
    grouped = route.reshape(x.shape[0], groups, -1)
    top = grouped.sort(dim=-1, descending=True, stable=True).values[..., : k // topk_group]
    chosen = top.sum(-1).argsort(dim=-1, descending=True, stable=True)[..., :topk_group]
    enabled = torch.zeros(x.shape[0], groups, dtype=torch.bool).scatter_(1, chosen, True)
    mask = enabled.repeat_interleave(x.shape[1] // groups, dim=1)
    indices = route.masked_fill(~mask, -torch.inf).argsort(dim=-1, descending=True, stable=True)[..., :k]
    weight = score.gather(1, indices)
    return indices, weight / weight.sum(-1, keepdim=True).clamp_min(1e-20)


def close(actual, expected):
    atol, rtol = {
        torch.float32: (3e-6, 3e-5),
        torch.float16: (3e-4, 2e-3),
        torch.bfloat16: (2e-3, 1e-2),
    }[expected.dtype]
    torch.testing.assert_close(actual.detach().cpu(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "tokens,experts,k", [(1, 1, 1), (7, 17, 4), (35, 65, 7), (3, 255, 16), (2, 256, 16), (5, 512, 16)]
)
def test_softmax_forward_and_gradients(backend, dtype, renormalize, strided, tokens, experts, k):
    device, _ = backend
    logits, x = values((tokens, experts), dtype, device, strided=strided)
    gradient, g = values((tokens, k), torch.float32, device, strided=True, seed=41)
    logits.requires_grad_()
    indices, weight = areno_topk_softmax(logits, k, renormalize)
    weight.backward(gradient)
    ids, w, dx = softmax_reference(x, k, renormalize, g)
    assert indices.dtype == torch.int64 and weight.dtype == torch.float32
    torch.testing.assert_close(indices.cpu(), ids, atol=0, rtol=0)
    close(weight, w)
    close(logits.grad, dx)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "tokens,experts,k,groups,topk_group",
    [(1, 8, 2, 2, 1), (7, 32, 4, 4, 2), (3, 64, 6, 4, 2), (5, 512, 16, 64, 8), (35, 256, 5, 8, 2)],
)
def test_grouped_router_selection_and_unbiased_weights(backend, dtype, strided, tokens, experts, k, groups, topk_group):
    device, _ = backend
    logits, x = values((tokens, experts), dtype, device, strided=strided)
    bias, b = values((experts,), torch.float32, device, strided=strided, seed=41)
    ids, weights = areno_grouped_topk_router(logits, bias, k, groups, topk_group)
    expected_ids, expected_weights = grouped_reference(x, b, k, groups, topk_group)
    torch.testing.assert_close(ids.cpu(), expected_ids, atol=0, rtol=0)
    close(weights, expected_weights)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("renormalize", [False, True])
def test_softmax_ties_use_lower_expert_index(backend, dtype, renormalize):
    device, _ = backend
    logits = torch.zeros(3, 512, dtype=dtype, device=device, requires_grad=True)
    gradient = torch.arange(16, dtype=torch.float32).expand(3, -1)
    ids, weights = areno_topk_softmax(logits, 16, renormalize)
    weights.backward(gradient.to(device))
    expected_ids, expected_weights, expected_grad = softmax_reference(logits.detach().cpu(), 16, renormalize, gradient)
    torch.testing.assert_close(ids.cpu(), expected_ids, atol=0, rtol=0)
    close(weights, expected_weights)
    close(logits.grad, expected_grad)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias_kind", ["zero", "selected", "negative"])
def test_grouped_ties_and_bias_only_affects_selection(backend, dtype, bias_kind):
    device, _ = backend
    logits = torch.zeros(2, 32, dtype=dtype, device=device)
    b = torch.zeros(32)
    if bias_kind == "selected":
        b[16:24] = 3
    elif bias_kind == "negative":
        b[:16] = -3
    ids, weights = areno_grouped_topk_router(logits, b.to(device), 4, 4, 2)
    expected_ids, expected_weights = grouped_reference(logits.cpu(), b, 4, 4, 2)
    torch.testing.assert_close(ids.cpu(), expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(weights.cpu(), expected_weights, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("renormalize", [False, True])
def test_softmax_large_logits_and_negative_infinity_mask(backend, dtype, renormalize):
    device, _ = backend
    x = torch.tensor(
        [[1000, 999, 998, -1000], [-1000, -1001, -1002, -1003], [0, -torch.inf, 0, -torch.inf]], dtype=dtype
    )
    logits = x.to(device).requires_grad_()
    gradient = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 1.0]])
    ids, weights = areno_topk_softmax(logits, 2, renormalize)
    weights.backward(gradient.to(device))
    expected_ids, expected_weights, expected_grad = softmax_reference(x, 2, renormalize, gradient)
    torch.testing.assert_close(ids.cpu(), expected_ids, atol=0, rtol=0)
    close(weights, expected_weights)
    close(logits.grad, expected_grad)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value", [-1000.0, 1000.0])
def test_grouped_sigmoid_saturation_and_denominator_floor(backend, dtype, value):
    device, _ = backend
    logits = torch.full((3, 64), value, dtype=dtype, device=device)
    bias = torch.zeros(64, device=device)
    ids, weights = areno_grouped_topk_router(logits, bias, 8, 8, 2)
    expected_ids, expected_weights = grouped_reference(logits.cpu(), bias.cpu(), 8, 8, 2)
    torch.testing.assert_close(ids.cpu(), expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(weights.cpu(), expected_weights, atol=0, rtol=0)


@pytest.mark.parametrize("renormalize", [False, True])
def test_softmax_backward_accumulates_repeated_selected_indices(backend, renormalize):
    device, native = backend
    logits, x = values((3, 17), torch.float32, device)
    ids = torch.tensor([[1, 1, 3, 5], [2, 4, 4, 2], [0, 0, 0, 0]])
    gradient, g = values((3, 4), torch.float32, device, seed=41)
    dx = native.areno_topk_softmax_backward(gradient, logits, ids.to(device), renormalize)
    _, _, expected = softmax_reference(x, 4, renormalize, g, ids)
    close(dx, expected)


def test_routing_storage_offsets_and_current_stream(backend):
    device, native = backend
    api = getattr(torch, device)
    index = 1 if api.device_count() > 1 else 0
    target = torch.device(device, index)
    logits = torch.empty(4, 65, device=target)[1:]
    gradient = torch.empty(4, 4, device=target)[1:]
    bias = torch.empty(66, device=target)[1:]
    stream = api.Stream(device=target)
    with api.stream(stream):
        logits.fill_(0.0)
        gradient.fill_(1.0)
        bias.fill_(0.0)
        ids, weights = native.areno_topk_softmax_forward(logits, 4, False)
        dx = native.areno_topk_softmax_backward(gradient, logits, ids, False)
        grouped_ids, grouped_weights = areno_grouped_topk_router(logits, bias, 4, 5, 1)
    stream.synchronize()
    x, g = torch.zeros(3, 65), torch.ones(3, 4)
    expected_ids, expected_weights, expected_grad = softmax_reference(x, 4, False, g)
    for result in (ids, weights, dx, grouped_ids, grouped_weights):
        assert result.device == target
    torch.testing.assert_close(ids.cpu(), expected_ids, atol=0, rtol=0)
    close(weights, expected_weights)
    close(dx, expected_grad)
    torch.testing.assert_close(grouped_ids.cpu(), expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(grouped_weights.cpu(), torch.full((3, 4), 0.25), atol=0, rtol=0)


def test_routing_empty_tokens_and_native_contract(backend):
    device, native = backend
    if device != "npu":
        pytest.skip("Ascend validation; CUDA's existing empty-grid behavior is unchanged")
    empty = torch.empty(0, 17, device=device, requires_grad=True)
    ids, weight = areno_topk_softmax(empty, 4)
    weight.sum().backward()
    assert ids.shape == weight.shape == (0, 4) and empty.grad.shape == empty.shape
    ids, weight = areno_grouped_topk_router(empty, torch.zeros(17, device=device), 4, 1, 1)
    assert ids.shape == weight.shape == (0, 4)
    x = torch.ones(3, 17, device=device)
    for logits, k, error in (
        (x, 0, "top_k"),
        (x, 18, "top_k"),
        (x[:, :2], 4, "contiguous|top_k"),
        (x[0], 4, "2D"),
        (x[:, ::2], 4, "contiguous"),
    ):
        with pytest.raises(RuntimeError, match=error):
            native.areno_topk_softmax_forward(logits, k, True)
    with pytest.raises(RuntimeError, match="groups"):
        native.areno_grouped_topk_router(x, torch.zeros(17, device=device), 4, 2, 1)
    with pytest.raises(RuntimeError, match="dtype"):
        native.areno_topk_softmax_backward(
            torch.ones(3, 4, dtype=torch.float16, device=device),
            x,
            torch.zeros(3, 4, dtype=torch.int64, device=device),
            True,
        )


@pytest.mark.parametrize("renormalize", [False, True])
def test_routing_graph_replay(backend, renormalize):
    device, native = backend
    if device != "cuda":
        pytest.skip("CUDA graph regression for the common selection helper extraction")
    logits, x = values((3, 32), torch.float32, device)
    bias = torch.zeros(32, device=device)
    gradient = torch.ones(3, 4, device=device)

    def run():
        ids, weight = native.areno_topk_softmax_forward(logits, 4, renormalize)
        dx = native.areno_topk_softmax_backward(gradient, logits, ids, renormalize)
        grouped = areno_grouped_topk_router(logits, bias, 4, 4, 2)
        return ids, weight, dx, *grouped

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    for sign in (1, -1, 1):
        logits.copy_(x * sign)
        graph.replay()
        expected = (
            *softmax_reference(x * sign, 4, renormalize, gradient.cpu()),
            *grouped_reference(x * sign, bias.cpu(), 4, 4, 2),
        )
        for actual, reference in zip(outputs, expected, strict=True):
            if actual.dtype == torch.int64:
                torch.testing.assert_close(actual.cpu(), reference, atol=0, rtol=0)
            else:
                close(actual, reference)
