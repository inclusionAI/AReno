"""Permutation and block-alignment contracts shared by CUDA and Ascend."""

import importlib.util

import pytest
import torch

from areno.accel._extension import extension
from areno.accel.moe import areno_moe_permute, areno_moe_topk_permute, areno_moe_unpermute
from areno.accel.routing import areno_moe_align

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
        assert extension(device).moe_implementation == "ascendc"
    return device, extension(device)


def values(shape, dtype, device, *, strided=False):
    physical = (*shape[:-1], 2 * shape[-1]) if strided else shape
    cpu = (torch.arange(torch.tensor(physical).prod().item()).reshape(physical) % 17 / 32).to(dtype)
    tensor = cpu.to(device)
    if strided:
        tensor, cpu = tensor[..., ::2], cpu[..., ::2]
    return tensor, cpu


def route_data(tokens, k, experts):
    routes = torch.arange(tokens * k).reshape(tokens, k)
    ids = (routes * 13 + 1) % max(experts, 1)
    weights = ((routes % 11) - 5).float() / 16
    ids.reshape(-1)[::17] = -1
    return ids, weights


def valid_routes(ids, weights, start, experts):
    return (ids >= start) & (ids < start + experts) & (weights != 0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("tokens,experts,hidden", [(1, 1, 1), (7, 17, 31), (129, 33, 257), (35, 257, 1025)])
def test_dense_permutation_order_and_shared_backward(backend, dtype, strided, tokens, experts, hidden):
    device, _ = backend
    input, x = values((tokens, hidden), dtype, device, strided=strided)
    p = ((torch.arange(tokens * experts).reshape(tokens, experts) % 13) - 6).float() / 16
    mask = torch.arange(tokens * experts).reshape(tokens, experts) % 11 < 2
    routed_experts, routed_tokens = mask.T.nonzero(as_tuple=True)
    probs = p.to(device).requires_grad_()
    input.requires_grad_()
    output, weight, ids = areno_moe_permute(input, probs, mask.to(device), routed_tokens.numel())
    torch.testing.assert_close(ids.cpu(), routed_tokens, atol=0, rtol=0)
    torch.testing.assert_close(output.detach().cpu(), x[routed_tokens], atol=0, rtol=0)
    torch.testing.assert_close(weight.detach().cpu(), p[routed_tokens, routed_experts], atol=0, rtol=0)
    (output.float().sum() / 32 + weight.sum()).backward()
    expected = (mask.sum(-1, keepdim=True).expand_as(x) / 32).to(dtype)
    torch.testing.assert_close(input.grad.cpu(), expected, atol=0, rtol=0)
    assert probs.grad is None  # The existing dense-map wrapper does not backpropagate probabilities.


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "tokens,k,experts,hidden",
    [
        (1, 1, 1, 1),
        (7, 4, 17, 31),
        (129, 4, 33, 257),
        (513, 3, 257, 1025),
        (9, 4, 0, 17),
        (5, 0, 17, 0),
        (3, 4, 513, 0),
    ],
)
def test_topk_permutation_metadata_counts_and_inverse(backend, dtype, tokens, k, experts, hidden):
    device, native = backend
    input, x = values((tokens, hidden), dtype, device)
    start = 3
    topk, weights = route_data(tokens, k, start + experts + 3)
    result = native.areno_moe_topk_permute_forward(input, topk.to(device), weights.to(device), start, experts)
    out, weight, ids, pos, counts = [t.cpu() for t in result]
    valid = valid_routes(topk, weights, start, experts)
    routes = ids * k + pos.long()
    expected_routes = valid.reshape(-1).nonzero().flatten()
    torch.testing.assert_close(routes.sort().values, expected_routes, atol=0, rtol=0)
    torch.testing.assert_close(out, x[ids], atol=0, rtol=0)
    torch.testing.assert_close(weight, weights.flatten()[routes], atol=0, rtol=0)
    expected_counts = torch.bincount(topk[valid] - start, minlength=experts)
    torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
    assert counts.dtype == ids.dtype == torch.int64 and pos.dtype == torch.int32
    # CUDA may reorder rows within one expert because it reserves rows with
    # atomicAdd. Compare the route set and its expert partition, not that order.
    expected_experts = torch.repeat_interleave(torch.arange(experts) + start, expected_counts)
    torch.testing.assert_close(topk.flatten()[routes], expected_experts, atol=0, rtol=0)
    grad_weight = (routes.remainder(7).float() - 3) / 32
    actual = native.areno_moe_topk_weight_backward(grad_weight.to(device), result[2], result[3], tokens, k)
    expected = torch.zeros(tokens, k)
    expected.reshape(-1)[routes] = grad_weight
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    if ids.numel():
        grad = torch.full(out.shape, 1 / 32, dtype=dtype, device=device)
        dx = native.areno_moe_unpermute_forward(grad, result[2], tokens, hidden)
        expected = (valid.sum(-1, keepdim=True).expand(tokens, hidden) / 32).to(dtype)
        torch.testing.assert_close(dx.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_topk_shared_autograd_and_zero_weight_filter(backend, dtype):
    device, _ = backend
    input, x = values((17, 33), dtype, device, strided=True)
    indices, w = route_data(17, 4, 19)
    weights = w.to(device).requires_grad_()
    input.requires_grad_()
    output, route_weight, ids, counts = areno_moe_topk_permute(input, indices.to(device), weights, 3, 13)
    valid = valid_routes(indices, w, 3, 13)
    assert output.shape[0] == valid.sum().item() == counts.sum().item()
    (output.float().sum() / 32 + route_weight.square().sum()).backward()
    torch.testing.assert_close(
        input.grad.cpu(), (valid.sum(-1, keepdim=True).expand_as(x) / 32).to(dtype), atol=0, rtol=0
    )
    torch.testing.assert_close(weights.grad.cpu(), torch.where(valid, 2 * w, 0.0), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden", [1, 17, 1025])
def test_unpermute_backward_gathers_repeated_token_ids(backend, dtype, hidden):
    device, _ = backend
    input, x = values((7, hidden), dtype, device, strided=True)
    ids = torch.tensor([2, 0, 2, 4, 0, 4, 2], dtype=torch.int64)
    token_index = torch.stack((ids, torch.full_like(ids, -1)), dim=1).to(device)[:, 0]
    input.requires_grad_()
    output = areno_moe_unpermute(input, token_index, (6, hidden))
    expected = torch.zeros(6, hidden).index_add_(0, ids, x.float()).to(dtype)
    torch.testing.assert_close(output.detach().cpu(), expected, atol=0, rtol=0)
    gradient, g = values((6, hidden), dtype, device, strided=True)
    output.backward(gradient)
    torch.testing.assert_close(input.grad.cpu(), g[ids], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_preserves_stored_nonfinite_values_and_signed_zero(backend, dtype):
    device, native = backend
    x = torch.tensor([[0.0, -0.0, torch.nan, torch.inf], [-torch.inf, 1.0, -2.0, -0.0]], dtype=dtype)
    input = torch.cat((torch.zeros(1, 4, dtype=dtype), x)).to(device)[1:]
    ids = torch.tensor([1, 0, 1], dtype=torch.int64, device=device)
    actual = native.areno_moe_gather_by_token_index(input, ids).cpu()
    integer_type = torch.int32 if dtype == torch.float32 else torch.int16
    torch.testing.assert_close(actual.view(integer_type), x[[1, 0, 1]].view(integer_type), atol=0, rtol=0)


def test_topk_nonfinite_route_weights_are_kept_but_both_zero_signs_are_dropped(backend):
    device, native = backend
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(device)
    ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], device=device)
    weights = torch.tensor([[0.0, -0.0, torch.nan, torch.inf], [-torch.inf, 1.0, 0.0, -0.0]], device=device)
    out, w, tokens, pos, counts = native.areno_moe_topk_permute_forward(x, ids, weights, 0, 4)
    routes = tokens.cpu() * 4 + pos.cpu().long()
    torch.testing.assert_close(routes.sort().values, torch.tensor([2, 3, 4, 5]), atol=0, rtol=0)
    torch.testing.assert_close(w.cpu(), weights.cpu().flatten()[routes], atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(out.cpu(), x.cpu()[tokens.cpu()], atol=0, rtol=0)
    torch.testing.assert_close(counts.cpu(), torch.ones(4, dtype=torch.int64), atol=0, rtol=0)


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("initialize", [False, True])
@pytest.mark.parametrize(
    "routes,experts,block",
    [(0, 7, 16), (1, 3, 1), (257, 17, 16), (1023, 63, 32), (1024, 65, 64), (2051, 257, 16), (517, 513, 7)],
)
def test_alignment_padding_invalid_expert_bucket_and_canaries(backend, id_dtype, initialize, routes, experts, block):
    device, _ = backend
    keys = (torch.arange(routes, dtype=id_dtype) * 13) % (experts + 1) - 1
    slots = experts + 1
    capacity = routes + slots * (block - 1)
    block_capacity = (capacity + block - 1) // block
    raw_routes = torch.full((capacity + 2,), -777, dtype=torch.int32, device=device)
    raw_blocks = torch.full((block_capacity + 2,), -888, dtype=torch.int32, device=device)
    raw_total = torch.full((3,), -999, dtype=torch.int32, device=device)
    raw_scratch = torch.full((slots + 3,), -555, dtype=torch.int32, device=device)
    routed, block_ids, total, scratch = raw_routes[1:-1], raw_blocks[1:-1], raw_total[1:2], raw_scratch[1:-1]
    if not initialize:
        routed.fill_(routes)
    areno_moe_align(keys.to(device), slots, block, routed, block_ids, total, scratch, initialize)
    actual, actual_blocks = routed.cpu(), block_ids.cpu()
    offset = 0
    for expert in range(-1, experts):
        expected_routes = (keys == expert).nonzero().flatten().int()
        padded = (expected_routes.numel() + block - 1) // block * block
        current = actual[offset : offset + padded]
        valid = current[current < routes]
        torch.testing.assert_close(valid.sort().values, expected_routes, atol=0, rtol=0)
        assert torch.all(current[current >= routes] == routes)
        torch.testing.assert_close(
            actual_blocks[offset // block : (offset + padded) // block],
            torch.full((padded // block,), expert, dtype=torch.int32),
            atol=0,
            rtol=0,
        )
        offset += padded
    assert total.cpu().item() == offset
    assert torch.all(actual[offset:] == routes)
    for tensor, canary in ((raw_routes, -777), (raw_blocks, -888), (raw_total, -999), (raw_scratch, -555)):
        assert tensor[0].cpu().item() == tensor[-1].cpu().item() == canary


@pytest.mark.parametrize("id_dtype", [torch.int8, torch.uint8, torch.int16])
def test_alignment_other_integral_storage_types(backend, id_dtype):
    device, _ = backend
    keys = torch.arange(17, dtype=id_dtype) % 4
    routed = torch.empty(17 + 5 * 3, dtype=torch.int32, device=device)
    block_ids = torch.empty((routed.numel() + 3) // 4, dtype=torch.int32, device=device)
    total = torch.empty(1, dtype=torch.int32, device=device)
    scratch = torch.empty(6, dtype=torch.int32, device=device)
    areno_moe_align(keys.to(device), 5, 4, routed, block_ids, total, scratch)
    actual = routed.cpu()[: total.cpu().item()]
    torch.testing.assert_close(actual[actual < 17].sort().values, torch.arange(17, dtype=torch.int32), atol=0, rtol=0)


def test_expert_shards_recombine_with_shared_autograd(backend):
    device, _ = backend
    input, x = values((19, 17), torch.float32, device)
    indices, w = route_data(19, 4, 17)
    input.requires_grad_()
    weights = w.to(device).requires_grad_()
    outputs = []
    for start, count in ((0, 7), (7, 10)):
        out, rw, ids, _ = areno_moe_topk_permute(input, indices.to(device), weights, start, count)
        outputs.append(areno_moe_unpermute(out * rw[:, None], ids, tuple(input.shape)))
    result = outputs[0] + outputs[1]
    valid = valid_routes(indices, w, 0, 17)
    expected = x * torch.where(valid, w, 0.0).sum(-1, keepdim=True)
    torch.testing.assert_close(result.detach().cpu(), expected, atol=0, rtol=0)
    result.sum().backward()
    torch.testing.assert_close(
        input.grad.cpu(), torch.where(valid, w, 0.0).sum(-1, keepdim=True).expand_as(x), atol=0, rtol=0
    )
    torch.testing.assert_close(weights.grad.cpu(), torch.where(valid, x.sum(-1, keepdim=True), 0.0), atol=0, rtol=0)


def test_moe_current_stream_and_tensor_device(backend):
    device, _ = backend
    api = getattr(torch, device)
    index = 1 if api.device_count() > 1 else 0
    target = torch.device(device, index)
    input = torch.empty(3, 257, device=target, requires_grad=True)
    keys = torch.empty(3, 2, dtype=torch.int64, device=target)
    weights = torch.empty(3, 2, device=target, requires_grad=True)
    stream = api.Stream(device=target)
    stream.wait_stream(api.current_stream(target))
    with api.stream(stream):
        with torch.no_grad():
            input.fill_(0.25)
            keys.fill_(1)
            weights.fill_(0.5)
        out, w, ids, counts = areno_moe_topk_permute(input, keys, weights, 0, 3)
        result = areno_moe_unpermute(out * w[:, None], ids, (3, 257))
        result.sum().backward()
    stream.synchronize()
    assert result.device == target and counts.device == target
    torch.testing.assert_close(result.cpu(), torch.full((3, 257), 0.25), atol=0, rtol=0)
    torch.testing.assert_close(input.grad.cpu(), torch.ones(3, 257), atol=0, rtol=0)
    torch.testing.assert_close(weights.grad.cpu(), torch.full((3, 2), 257 / 4), atol=0, rtol=0)


def test_moe_empty_shapes_and_native_validation(backend):
    device, native = backend
    if device != "npu":
        pytest.skip("Ascend validation; existing CUDA zero-grid behavior is unchanged")
    for shape in ((0, 17), (3, 0)):
        input = torch.empty(shape, device=device, requires_grad=True)
        indices = torch.zeros(shape[0], 2, dtype=torch.int64, device=device)
        weights = torch.ones(shape[0], 2, device=device, requires_grad=True)
        out, rw, ids, _ = areno_moe_topk_permute(input, indices, weights, 0, 1)
        restored = areno_moe_unpermute(out, ids, shape)
        (restored.sum() + rw.sum()).backward()
        assert input.grad.shape == shape
        torch.testing.assert_close(weights.grad.cpu(), torch.ones_like(weights.cpu()), atol=0, rtol=0)
    input = torch.ones(3, 17, device=device)
    with pytest.raises(RuntimeError, match="shape"):
        native.areno_moe_unpermute_forward(input, torch.zeros(2, dtype=torch.int64, device=device), 3, 17)
    with pytest.raises(RuntimeError, match="dtype"):
        native.areno_moe_permute_forward(
            input,
            torch.ones(3, 2, dtype=torch.float16, device=device),
            torch.ones(3, 2, dtype=torch.bool, device=device),
            6,
        )
    with pytest.raises(RuntimeError, match="expert range"):
        native.areno_moe_topk_permute_forward(
            input, torch.zeros(3, 2, dtype=torch.int64, device=device), torch.ones(3, 2, device=device), 0, -1
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_fixed_shape_moe_graph_replay(backend, dtype):
    device, native = backend
    if device != "cuda":
        pytest.skip("CUDA graph regression for the shared autograd change")
    input, x = values((7, 17), dtype, device)
    map_cpu = torch.arange(7 * 5).reshape(7, 5) % 3 == 0
    mapping = map_cpu.to(device)
    probs = torch.full((7, 5), 0.25, device=device)
    rows = map_cpu.sum().item()
    keys_cpu = torch.arange(17, dtype=torch.int64) % 5 - 1
    keys = keys_cpu.to(device)
    routed = torch.empty(17 + 5 * 3, dtype=torch.int32, device=device)
    block_ids = torch.empty((routed.numel() + 3) // 4, dtype=torch.int32, device=device)
    total = torch.empty(1, dtype=torch.int32, device=device)
    scratch = torch.empty(6, dtype=torch.int32, device=device)

    def run():
        out, weight, ids = areno_moe_permute(input, probs, mapping, rows)
        restored = areno_moe_unpermute(out, ids, (7, 17))
        gathered = native.areno_moe_gather_by_token_index(restored, ids)
        areno_moe_align(keys, 5, 4, routed, block_ids, total, scratch)
        return out, weight, ids, restored, gathered

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    for shift in (0, 1, 3):
        input.copy_(x + shift)
        probs.fill_(0.25 * (shift + 1))
        updated_map = map_cpu.roll(shift, dims=1)
        mapping.copy_(updated_map)
        keys.copy_(keys_cpu.roll(shift))
        graph.replay()
        _, tokens = updated_map.T.nonzero(as_tuple=True)
        restored = (x + shift) * updated_map.sum(-1, keepdim=True).to(dtype)
        expected = ((x + shift)[tokens], torch.full((rows,), 0.25 * (shift + 1)), tokens, restored, restored[tokens])
        for actual, reference in zip(outputs, expected, strict=True):
            torch.testing.assert_close(actual.cpu(), reference, atol=0, rtol=0)
        offset = 0
        actual = routed.cpu()
        for expert in range(-1, 4):
            ids = (keys_cpu.roll(shift) == expert).nonzero().flatten().int()
            padded = (ids.numel() + 3) // 4 * 4
            chunk = actual[offset : offset + padded]
            torch.testing.assert_close(chunk[chunk < 17].sort().values, ids, atol=0, rtol=0)
            assert torch.all(chunk[chunk >= 17] == 17)
            assert torch.all(block_ids[offset // 4 : (offset + padded) // 4].cpu() == expert)
            offset += padded
        assert total.cpu().item() == offset
