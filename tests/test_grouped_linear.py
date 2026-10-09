"""One contract suite for the compiled common host code and both device GEMMs.

The CPU adapter compiles the production header; it does not emulate CANN or
cuBLAS. Use -k cpu for that subset, -k npu or -k cuda for device validation.
"""

import importlib.util
import itertools
import shutil
from pathlib import Path

import pytest
import torch
from setuptools import Distribution
from torch.utils.cpp_extension import BuildExtension, CppExtension, get_cxx_compiler

from areno.accel._extension import extension
from areno.accel.linear import areno_grouped_linear

ROOT = Path(__file__).resolve().parents[1]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SHAPES = [
    ([], 7, 9),
    ([0, 0, 0], 7, 9),
    ([1], 1, 1),
    ([0, 3, 0, 1, 5, 0], 17, 65),
    ([1, 2, 0, 3], 1025, 7),
    ([2, 0, 1], 7, 1027),
    ([2, 0, 1], 0, 65),
    ([2, 0, 1], 17, 0),
    ([int(i % 17 == 0) for i in range(130)], 7, 9),
]


@pytest.fixture(scope="module")
def cpu_native(tmp_path_factory):
    if shutil.which(get_cxx_compiler()) is None:
        pytest.skip("A C++ compiler is required for the shared host-code test")
    name = "_areno_grouped_linear_common_test"
    build = tmp_path_factory.mktemp("grouped_linear_common")
    ext = CppExtension(
        name,
        sources=[str(ROOT / "tests/csrc/grouped_linear_common.cpp")],
        include_dirs=[str(ROOT / "areno/accel/csrc")],
        extra_compile_args=["-O0", "-g0"],
    )
    command = BuildExtension(Distribution({"ext_modules": [ext]}), use_ninja=False)
    command.ensure_finalized()
    command.build_temp, command.build_lib = str(build / "temp"), str(build / "lib")
    build_failed = False
    try:
        command.run()
    except Exception:
        build_failed = True
    if build_failed:
        pytest.fail("C++ test extension build failed; see compiler output", pytrace=False)
    spec = importlib.util.spec_from_file_location(name, command.get_ext_fullpath(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=["cpu", "cuda", "npu"])
def backend(request):
    device = request.param
    if device == "cpu":
        return device, request.getfixturevalue("cpu_native")
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA hardware and the compiled extension are required")
    else:
        if importlib.util.find_spec("torch_npu") is None:
            pytest.skip("Ascend torch_npu and hardware are required")
        import torch_npu  # noqa: F401

        assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
        torch.npu.set_device(0)
    native = extension(device)
    for suffix in ("forward", "backward", "forward_counts", "backward_counts"):
        assert hasattr(native, f"areno_grouped_linear_{suffix}")
    return device, native


def values(shape, dtype, seed=37):
    return (torch.randint(-8, 9, shape, generator=torch.Generator().manual_seed(seed)) / 64).to(dtype)


def reference(x, w, counts, grad):
    # Expand experts by token and use independent per-token batched products.
    # Double precision autograd avoids low-precision per-token accumulation
    # when constructing the expected grouped weight gradient.
    input = x.double().requires_grad_()
    weight = w.double().requires_grad_()
    expert_ids = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts, dtype=torch.int64))
    output = torch.bmm(weight[expert_ids], input.unsqueeze(-1)).squeeze(-1)
    dx, dw = torch.autograd.grad(output, (input, weight), grad.double())
    return output.to(x.dtype), dx.to(x.dtype), dw.to(w.dtype)


def native_calls(native, x, w, counts, gradient, needs=(True, True)):
    suffix = "_counts" if isinstance(counts, torch.Tensor) else ""
    output = getattr(native, f"areno_grouped_linear_forward{suffix}")(x, w, counts)
    dx, dw = getattr(native, f"areno_grouped_linear_backward{suffix}")(gradient, x, w, counts, *needs)
    return output, dx, dw


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("counts,k,n", SHAPES)
@pytest.mark.parametrize("count_type", ["list", "int32", "int64"])
def test_grouped_native_shapes_and_empty_experts(backend, dtype, counts, k, n, count_type):
    device, native = backend
    x, w = values((sum(counts), k), dtype), values((len(counts), n, k), dtype, seed=41)
    g = values((sum(counts), n), dtype, seed=43)
    c = counts if count_type == "list" else torch.tensor(counts, dtype=getattr(torch, count_type), device=device)
    actual = native_calls(native, x.to(device), w.to(device), c, g.to(device))
    expected = reference(x, w, counts, g)
    for output, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(output.cpu(), target, atol=0, rtol=0)


@pytest.mark.parametrize("needs", list(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize("count_type", ["list", "int32", "int64"])
def test_grouped_selective_gradients_and_storage_offsets(backend, needs, count_type):
    device, native = backend
    counts = [3, 0, 2, 1]
    x, w, g = values((6, 17), torch.float32), values((4, 65, 17), torch.float32), values((6, 65), torch.float32)
    input = torch.cat((torch.zeros(1, 17), x)).to(device)[1:]
    weight = torch.cat((torch.zeros(1, 65, 17), w)).to(device)[1:]
    grad = torch.cat((torch.zeros(1, 65), g)).to(device)[1:]
    c = counts if count_type == "list" else torch.tensor(counts, dtype=getattr(torch, count_type), device=device)
    output, dx, dw = native_calls(native, input, weight, c, grad, needs)
    y_ref, dx_ref, dw_ref = reference(x, w, counts, g)
    torch.testing.assert_close(output.cpu(), y_ref, atol=0, rtol=0)
    for value, target, needed in zip((dx, dw), (dx_ref, dw_ref), needs, strict=True):
        if needed:
            torch.testing.assert_close(value.cpu(), target, atol=0, rtol=0)
        else:
            assert value.numel() == 0


@pytest.mark.parametrize("count_type", ["list", "int32", "int64"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_grouped_shared_autograd_packs_strided_inputs(backend, count_type, dtype):
    device, _ = backend
    if device == "cpu":
        pytest.skip("The public accel wrapper requires a kernel device")
    counts = [2, 0, 3]
    x, w, g = values((5, 34), dtype)[:, ::2], values((3, 65, 34), dtype)[:, :, ::2], values((5, 130), dtype)[:, ::2]
    input = values((5, 34), dtype).to(device)[:, ::2].requires_grad_()
    weight = values((3, 65, 34), dtype).to(device)[:, :, ::2].requires_grad_()
    gradient = values((5, 130), dtype).to(device)[:, ::2]
    c = counts
    if count_type != "list":
        c = torch.tensor([2, -1, 0, -1, 3, -1], dtype=getattr(torch, count_type), device=device)[::2]
    output = areno_grouped_linear(input, weight, c)
    output.backward(gradient)
    expected = reference(x, w, counts, g)
    for actual, target in zip((output, input.grad, weight.grad), expected, strict=True):
        torch.testing.assert_close(actual.cpu(), target, atol=0, rtol=0)


@pytest.mark.parametrize(
    "counts,error",
    [
        ([-1, 7], "non-negative"),
        ([2, 3], "sum"),
        ([2, 5], "sum"),
        ([6], "expert count"),
        ([2**63 - 1, 2**63 - 1], "sum"),
    ],
)
@pytest.mark.parametrize("tensor_counts", [False, True])
def test_grouped_invalid_counts_are_rejected(backend, counts, error, tensor_counts):
    device, native = backend
    x, w, g = torch.ones(6, 17, device=device), torch.ones(2, 65, 17, device=device), torch.ones(6, 65, device=device)
    c = torch.tensor(counts, device=device) if tensor_counts else counts
    suffix = "_counts" if tensor_counts else ""
    with pytest.raises(RuntimeError, match=error):
        getattr(native, f"areno_grouped_linear_forward{suffix}")(x, w, c)
    with pytest.raises(RuntimeError, match=error):
        getattr(native, f"areno_grouped_linear_backward{suffix}")(g, x, w, c, True, True)


def test_grouped_input_contract(backend):
    device, native = backend
    x, w = torch.ones(6, 17, device=device), torch.ones(2, 65, 17, device=device)
    forward = native.areno_grouped_linear_forward
    for a, b, error in (
        (x[0], w, "2D"),
        (x, w[0], "3D"),
        (x, w.half(), "dtype"),
        (x[:, ::2], w[:, :, ::2], "contiguous"),
    ):
        with pytest.raises(RuntimeError, match=error):
            forward(a, b, [3, 3])
    with pytest.raises(RuntimeError, match="dtype|int32 or int64"):
        native.areno_grouped_linear_forward_counts(x, w, torch.tensor([3.0, 3.0], device=device))
    with pytest.raises(RuntimeError, match="1D"):
        native.areno_grouped_linear_forward_counts(x, w, torch.tensor([[3, 3]], device=device))
    with pytest.raises(RuntimeError, match="gradient shape"):
        native.areno_grouped_linear_backward(torch.ones(6, 64, device=device), x, w, [3, 3], True, True)


def test_grouped_nondefault_stream_and_device(backend):
    device, native = backend
    if device == "cpu":
        pytest.skip("Streams and device guards need CUDA or NPU")
    api = getattr(torch, device)
    index = 1 if api.device_count() > 1 else 0
    target = torch.device(device, index)
    input, weight, grad = (
        torch.empty(3, 17, device=target),
        torch.empty(3, 65, 17, device=target),
        torch.empty(3, 65, device=target),
    )
    counts = torch.empty(3, dtype=torch.int64, device=target)
    stream = api.Stream(device=target)
    with api.stream(stream):
        input.fill_(0.125)
        weight.fill_(0.25)
        grad.fill_(1.0)
        counts.fill_(1)
        y, dx, dw = native_calls(native, input, weight, counts, grad)
    stream.synchronize()
    for value in (y, dx, dw):
        assert value.device == target
    torch.testing.assert_close(y.cpu(), torch.full((3, 65), 17 / 32), atol=0, rtol=0)
    torch.testing.assert_close(dx.cpu(), torch.full((3, 17), 65 / 4), atol=0, rtol=0)
    torch.testing.assert_close(dw.cpu(), torch.full((3, 65, 17), 1 / 8), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_grouped_cuda_graph_replays_list_counts(backend, dtype):
    device, native = backend
    if device != "cuda":
        pytest.skip("CUDA graph regression for the shared host-code extraction")
    counts = [2, 0, 3]
    x, w, g = values((5, 17), dtype), values((3, 65, 17), dtype, seed=41), values((5, 65), dtype, seed=43)
    input, weight, grad = x.to(device), w.to(device), g.to(device)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            native_calls(native, input, weight, counts, grad)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        results = native_calls(native, input, weight, counts, grad)
    for sign in (1, -1, 1):
        input.copy_(x * sign)
        graph.replay()
        for actual, expected in zip(results, reference(x * sign, w, counts, g), strict=True):
            torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
