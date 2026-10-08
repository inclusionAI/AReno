"""One fused-expert contract for CUDA/Triton and Ascend C/Cube.

This inference entry has no backward on either backend. Training continues to
use the shared permutation, grouped-linear and activation autograd wrappers.
"""

import importlib.util
import math

import pytest
import torch
import torch.nn.functional as F

from areno.accel._extension import extension
from areno.accel.ops import FusedMoeConfig, areno_fused_experts


@pytest.fixture(scope="module", params=["cuda", "npu"])
def backend(request):
    device = request.param
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA hardware, Triton and compiled kernels are required")
    else:
        if importlib.util.find_spec("torch_npu") is None:
            pytest.skip("Ascend hardware and torch_npu are required")
        import torch_npu  # noqa: F401

        assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
        torch.npu.set_device(0)
        assert extension(device).fused_experts_implementation == "ascendc_cube"
    return device


def reference(x, w1, w2, weights, ids, activation="silu", scale=1.0, *, early_round=False):
    """FP32 arithmetic with CUDA's storage casts at each pipeline boundary."""
    output = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = ids[token, slot].item()
            if expert == -1:
                continue
            gate_up = (x[token].float() @ w1[expert].float().T).to(x.dtype).float()
            gate, up = gate_up.chunk(2)
            activated = F.silu(gate) if activation == "silu" else F.gelu(gate, approximate="tanh")
            hidden = (activated * up).to(x.dtype)
            down = hidden.float() @ w2[expert].float().T
            if early_round:
                down = down.to(x.dtype).float()
            output[token] += (down * weights[token, slot].float()).to(x.dtype).float()
    return (output * scale).to(x.dtype)


def upload(x, device, strided=False):
    """Guarded storage with a nonzero offset; optional stride in the last axis."""
    shape = (*x.shape[:-1], x.shape[-1] * (2 if strided else 1))
    base = torch.full((math.prod(shape) + 16,), -7, dtype=x.dtype, device=device)
    view = base[8:-8].view(shape)
    if strided:
        view = view[..., ::2]
    view.copy_(x)
    return view


def inputs(tokens, hidden, intermediate, experts, top_k, dtype):
    generator = torch.Generator().manual_seed(2701)
    x = (torch.randn(tokens, hidden, generator=generator) / 3).to(dtype)
    w1 = (torch.randn(experts, 2 * intermediate, hidden, generator=generator) / 3).to(dtype)
    w2 = (torch.randn(experts, hidden, intermediate, generator=generator) / 3).to(dtype)
    ids = torch.randint(-1, experts, (tokens, top_k), generator=generator)
    weights = torch.randn(tokens, top_k, generator=generator) / top_k
    weights[::3, -1] = 0
    if top_k > 1:
        ids[::2, -1] = ids[::2, 0]  # Repeated experts still have distinct weights.
    return x, w1, w2, weights, ids


def evaluate(tensors, activation="silu", scale=1.0):
    x, w1, w2, weights, ids = tensors
    config = FusedMoeConfig(w1.shape[0], x.shape[-1], w2.shape[-1], ids.shape[-1], scale)
    return areno_fused_experts(x, w1, w2, weights, ids, config, activation=activation)


def close(actual, expected):
    atol, rtol = (2e-3, 6e-3) if expected.dtype == torch.float16 else (2e-2, 4e-2)
    torch.testing.assert_close(actual.detach().cpu(), expected, atol=atol, rtol=rtol)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1, 1, 1), (1, 65, 129, 17, 4), (17, 31, 33, 5, 3), (33, 257, 513, 3, 2), (1031, 5, 7, 19, 4)],
)
def test_fused_experts_forward(backend, dtype, activation, strided, shape):
    data = inputs(*shape, dtype)
    actual = evaluate([upload(x, backend, strided) for x in data], activation, 1.7)
    close(actual, reference(*data, activation, 1.7))
    assert actual.device.type == backend


def precision_case(dtype, overflow=False):
    intermediate = 16 if overflow else 1
    x = torch.ones(1, 16, dtype=dtype)
    w1 = torch.zeros(1, 2 * intermediate, 16, dtype=dtype)
    w1[:, :intermediate, 0] = 4
    w1[:, intermediate:, 0] = 1
    value = 2048 if overflow else (1.0078125 if dtype == torch.float16 else 1.0390625)
    w2 = torch.full((1, 16, intermediate), value, dtype=dtype)
    return x, w1, w2, torch.tensor([[0.1]]), torch.zeros(1, 1, dtype=torch.int64)


@pytest.mark.parametrize("dtype,overflow", [(torch.float16, False), (torch.bfloat16, False), (torch.float16, True)])
def test_cpu_reference_distinguishes_down_projection_rounding(dtype, overflow):
    data = precision_case(dtype, overflow)
    expected = reference(*data)
    wrong = reference(*data, early_round=True)
    assert torch.isfinite(expected).all()
    assert torch.all(expected != wrong)
    if overflow:
        assert torch.isinf(wrong).all()


@pytest.mark.parametrize("dtype,overflow", [(torch.float16, False), (torch.bfloat16, False), (torch.float16, True)])
def test_route_weight_precedes_down_projection_storage_cast(backend, dtype, overflow):
    data = precision_case(dtype, overflow)
    actual = evaluate([x.to(backend) for x in data])
    torch.testing.assert_close(actual.cpu(), reference(*data), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sum_uses_topk_slot_order_in_fp32(backend, dtype):
    x = torch.ones(1, 16, dtype=dtype)
    w1 = torch.zeros(3, 2, 16, dtype=dtype)
    w1[:, 0, 0], w1[:, 1, 0] = 16, 1 / 16
    w2 = torch.tensor([65504, -65504, 1 / 1024], dtype=dtype)[:, None, None].expand(3, 16, 1)
    ids = torch.tensor([[0, 2, 1]], dtype=torch.int32)
    data = x, w1, w2, torch.ones(1, 3), ids
    actual = evaluate([x.to(backend) for x in data])
    # large + tiny - large == 0 in FP32; expert-order addition gives tiny.
    torch.testing.assert_close(actual.cpu(), torch.zeros_like(x), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_routing_weights(backend, dtype):
    data = list(inputs(7, 17, 19, 3, 2, dtype))
    data[-2], data[-1] = data[-2].to(dtype), data[-1].int()
    close(evaluate([t.to(backend) for t in data]), reference(*data))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_minus_one_routes_ignore_nonfinite_input_and_weights(backend, dtype):
    data = list(inputs(3, 17, 19, 2, 3, dtype))
    for value in data[:-1]:
        value.fill_(torch.nan)
    data[-1].fill_(-1)
    actual = evaluate([x.to(backend) for x in data])
    torch.testing.assert_close(actual.cpu(), torch.zeros(3, 17, dtype=dtype), atol=0, rtol=0)
    # A valid zero-weight route still executes the MLP, exactly as CUDA does.
    data[-1][0, 0] = 0
    data[-2][0, 0] = 0
    actual = evaluate([x.to(backend) for x in data]).cpu()
    assert torch.isnan(actual[0]).all()
    assert torch.all(actual[1:] == 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tensor_parallel_intermediate_shards_and_local_experts(backend, dtype):
    x, w1, w2, weights, ids = inputs(9, 17, 34, 4, 3, dtype)
    # TP slices retain gate/up halves; local expert ids include -1 sentinels.
    local_ids = torch.where(ids >= 2, ids - 2, -1)
    for start, end in ((0, 17), (17, 34)):
        gate_up = torch.cat((w1[2:, start:end], w1[2:, 34 + start : 34 + end]), dim=1)
        data = x, gate_up, w2[2:, :, start:end], weights, local_ids
        close(evaluate([upload(t, backend, True) for t in data], scale=0.75), reference(*data, scale=0.75))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_non_default_stream_and_device(backend, dtype):
    api = getattr(torch, backend)
    target = torch.device(backend, min(1, api.device_count() - 1))
    data = inputs(17, 33, 65, 3, 2, dtype)
    tensors = [t.to(target) for t in data]
    stream = api.Stream(device=target)
    stream.wait_stream(api.current_stream(target))
    api.set_device(0)
    with api.stream(stream):
        tensors[0].mul_(2)
        result = evaluate(tensors)
    stream.synchronize()
    assert result.device == target
    close(result, reference(data[0] * 2, *data[1:]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_graph_replay_changes_expert_counts_without_host_sync(backend, dtype):
    api = getattr(torch, backend)
    data = inputs(17, 33, 65, 5, 3, dtype)
    tensors = [t.to(backend) for t in data]
    stream = api.Stream()
    stream.wait_stream(api.current_stream())
    with api.stream(stream):
        for _ in range(3):
            evaluate(tensors)
    api.current_stream().wait_stream(stream)
    graph = api.CUDAGraph() if backend == "cuda" else api.NPUGraph()
    with api.graph(graph):
        actual = evaluate(tensors)
    for expert in (-1, 0, 4):
        tensors[-1].fill_(expert)
        tensors[-2].fill_(0.25)
        graph.replay()
        updated = *data[:3], torch.full_like(data[-2], 0.25), torch.full_like(data[-1], expert)
        close(actual, reference(*updated))


def test_npu_empty_and_input_validation(backend):
    if backend != "npu":
        pytest.skip("Native Ascend validation; CUDA validation is unchanged")
    data = [t.to(backend) for t in inputs(0, 17, 19, 3, 2, torch.bfloat16)]
    assert evaluate(data).shape == (0, 17)
    data = [t.to(backend) for t in inputs(3, 17, 19, 3, 2, torch.bfloat16)]
    with pytest.raises(ValueError, match="activation"):
        evaluate(data, "relu")
    with pytest.raises(RuntimeError, match="dtype"):
        evaluate([data[0], data[1].float(), *data[2:]])
    with pytest.raises(RuntimeError, match="integral"):
        evaluate([*data[:-1], data[-1].float()])
    with pytest.raises(RuntimeError, match="shape"):
        evaluate([*data[:3], data[3][:, :1], data[4]])
