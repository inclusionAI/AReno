"""CPU numerical references for the actual convolution and routing TPC source."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.test_hpu_attention_source_cpu import DTYPES, floats, storage
from tests.test_hpu_quantized_optimizer_source_cpu import pointers


@pytest.fixture(scope="module", params=range(3))
def source(tmp_path_factory, request):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Clang is required for the TPC source interpreter")
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("hpu_conv_topk")
    dtype = request.param
    text = (
        '#include "hpu_tpc_reference.h"\n'
        + f'#define ARENO_DTYPE {dtype}\n#include "tensor_io.h"\n#include "index_io.h"\n'
    )
    for family, kinds in (("conv", (0, 1, 2)), ("topk", (0, 1))):
        for kind in kinds:
            for direction in (0, 1):
                if direction and ((family == "conv" and kind == 2) or (family == "topk" and kind == 1)):
                    continue
                text += (
                    f"namespace {family}_{kind}_{direction} {{\n#define ARENO_KIND {kind}\n#define ARENO_DIRECTION {direction}\n#define main kernel\n"
                    f'#include "{family}.c"\n#undef main\n#undef ARENO_KIND\n#undef ARENO_DIRECTION\n}}\n'
                )
    text += (
        'extern "C" void conv(int kind,int n,int channels,int kernel,int length,int sequences,void** p) {\n'
        f"auto t=[&](int i,int rows) {{ return tensor{{p[i],channels,rows,{4 if dtype == 0 else 2}}}; }};\n"
        "tensor x=t(0,n),weight{p[1],channels,kernel},grad=t(2,n),out=t(3,n),preact{p[4],channels,n},dx=t(5,n),dw{p[6],channels,kernel},cu{p[7],sequences+1,1};\n"
        "index_space_origin=int5{}; index_space_extent=int5{(channels+127)/128,n,0,0,0};\n"
        "#define PARAMS n,channels,kernel,length,sequences\n"
        "if (kind == 0) conv_0_0::kernel(x,weight,out,preact,PARAMS);\n"
        "else if (kind == 1) conv_1_0::kernel(x,weight,cu,out,preact,PARAMS);\n"
        "else { conv_2_0::kernel(x,weight,t(8,n*(kernel-1)),out,preact,PARAMS); return; }\n"
        "index_space_extent=int5{(channels+127)/128,n>kernel ? n : kernel,0,0,0};\n"
        "if (kind == 0) conv_0_1::kernel(x,weight,grad,preact,dx,dw,PARAMS);\n"
        "else conv_1_1::kernel(x,weight,grad,preact,cu,dx,dw,PARAMS);\n}\n"
        'extern "C" void topk(int grouped,int n,int experts,int k,int renorm,int groups,int top_groups,void** p) {\n'
        f"tensor x{{p[0],experts,n,{4 if dtype == 0 else 2}}},dx{{p[5],experts,n,{4 if dtype == 0 else 2}}};\n"
        "tensor bias{p[1],experts,1},ids{p[2],k,n,8},weights{p[3],k,n},grad{p[4],k,n};\n"
        "index_space_origin=int5{}; index_space_extent=int5{n,0,0,0,0};\n"
        "if (grouped) topk_1_0::kernel(x,bias,ids,weights,n,experts,k,renorm,groups,top_groups);\n"
        "else { topk_0_0::kernel(x,ids,weights,n,experts,k,renorm,groups,top_groups); topk_0_1::kernel(x,ids,grad,dx,n,experts,k,renorm,groups,top_groups); }\n}\n"
    )
    path = build / "source.cpp"
    path.write_text(text)
    library = build / "source.so"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Werror",
            "-Wno-psabi",
            "-I",
            str(root / "tests"),
            "-I",
            str(root / "areno/accel/csrc/hpu"),
            str(path),
            "-o",
            str(library),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    native = ctypes.CDLL(str(library))
    native.conv.argtypes = [ctypes.c_int] * 6 + [ctypes.POINTER(ctypes.c_void_p)]
    native.topk.argtypes = [ctypes.c_int] * 7 + [ctypes.POINTER(ctypes.c_void_p)]
    native.conv.restype = native.topk.restype = None
    return native, DTYPES[dtype]


@pytest.mark.parametrize("channels", [1, 65, 129])
@pytest.mark.parametrize("kernel", [1, 4, 9])
@pytest.mark.parametrize("kind", [0, 1, 2])
def test_conv_forward_backward_and_decode(source, channels, kernel, kind):
    native, dtype = source
    rng = torch.Generator().manual_seed(25)
    tokens = 2 if kind == 2 else 14
    x = torch.randn(tokens, channels, generator=rng).to(dtype).float().requires_grad_()
    w = torch.randn(channels, 1, kernel, generator=rng).requires_grad_()
    grad = torch.randn(tokens, channels, generator=rng).to(dtype).float()
    cu = np.array([0, 0, 1, 6, 6, 14], dtype=np.int32)
    history = torch.randn(tokens, channels, kernel - 1, generator=rng).to(dtype).float()
    data = [
        storage(x.detach(), dtype),
        w.detach().squeeze(1).T.contiguous().numpy(),
        storage(grad, dtype),
        storage(torch.empty_like(x), dtype),
        np.empty((tokens, channels), np.float32),
        storage(torch.empty_like(x), dtype),
        np.empty((kernel, channels), np.float32),
        cu,
        storage(history.transpose(1, 2).contiguous(), dtype),
    ]
    native.conv(kind, tokens, channels, kernel, 7, len(cu) - 1, pointers(data))
    if kind == 2:
        preact = (history * w[:, 0, :-1]).sum(-1) + x * w[:, 0, -1]
    else:
        boundaries = [0, 7, 14] if kind == 0 else list(cu)
        parts = []
        for begin, end in zip(boundaries, boundaries[1:]):
            if begin == end:
                continue
            segment = x[begin:end].T.unsqueeze(0)
            parts.append(F.conv1d(segment, w, padding=kernel - 1, groups=channels)[0, :, : end - begin].T)
        preact = torch.cat(parts)
    expected = F.silu(preact)
    expected.backward(grad)
    torch.testing.assert_close(torch.from_numpy(data[4]), preact.detach(), rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(
        floats(data[3], dtype), expected.detach().to(dtype).float(), rtol=max(torch.finfo(dtype).eps, 3e-5), atol=3e-6
    )
    if kind != 2:
        torch.testing.assert_close(
            floats(data[5], dtype), x.grad.to(dtype).float(), rtol=max(torch.finfo(dtype).eps, 3e-5), atol=3e-6
        )
        torch.testing.assert_close(torch.from_numpy(data[6]).T.unsqueeze(1), w.grad, rtol=3e-5, atol=4e-6)


@pytest.mark.parametrize("experts,top_k", [(1, 1), (16, 4), (32, 16), (128, 4), (512, 16)])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("ties", [False, True])
def test_topk_softmax_forward_backward(source, experts, top_k, renormalize, ties):
    native, dtype = source
    rng = torch.Generator().manual_seed(18)
    x = (
        (torch.zeros(3, experts) if ties else torch.randn(3, experts, generator=rng) * 2)
        .to(dtype)
        .float()
        .requires_grad_()
    )
    grad = torch.randn(3, top_k, generator=rng)
    data = [
        storage(x.detach(), dtype),
        np.zeros(experts, np.float32),
        np.empty((3, top_k), np.int64),
        np.empty((3, top_k), np.float32),
        grad.numpy(),
        storage(torch.empty_like(x), dtype),
    ]
    native.topk(0, 3, experts, top_k, int(renormalize), 0, 0, pointers(data))
    p = x.softmax(-1)
    ids = p.argsort(dim=-1, descending=True, stable=True)[:, :top_k]
    expected = p.gather(-1, ids)
    if renormalize:
        expected = expected / expected.sum(-1, keepdim=True)
    expected.backward(grad)
    np.testing.assert_array_equal(data[2], ids.numpy())
    torch.testing.assert_close(torch.from_numpy(data[3]), expected.detach(), rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(
        floats(data[5], dtype), x.grad.to(dtype).float(), rtol=max(torch.finfo(dtype).eps, 3e-5), atol=3e-6
    )


@pytest.mark.parametrize(
    "experts,groups,top_groups,top_k", [(16, 1, 1, 4), (32, 4, 2, 4), (128, 8, 4, 8), (512, 64, 8, 16)]
)
@pytest.mark.parametrize("ties", [False, True])
def test_grouped_router_stable_selection(source, experts, groups, top_groups, top_k, ties):
    native, dtype = source
    rng = torch.Generator().manual_seed(29)
    x = (torch.zeros(3, experts) if ties else torch.randn(3, experts, generator=rng) * 2).to(dtype).float()
    bias = torch.zeros(experts) if ties else torch.randn(experts, generator=rng) * 0.1
    data = [
        storage(x, dtype),
        bias.numpy(),
        np.empty((3, top_k), np.int64),
        np.empty((3, top_k), np.float32),
        np.zeros((3, top_k), np.float32),
        storage(torch.empty_like(x), dtype),
    ]
    native.topk(1, 3, experts, top_k, 1, groups, top_groups, pointers(data))
    scores = x.sigmoid()
    route = scores + bias
    group_scores = (
        route.reshape(3, groups, -1)
        .sort(dim=-1, descending=True, stable=True)
        .values[:, :, : top_k // top_groups]
        .sum(-1)
    )
    selected_groups = group_scores.argsort(dim=-1, descending=True, stable=True)[:, :top_groups]
    mask = (
        torch.zeros((3, groups), dtype=torch.bool)
        .scatter_(1, selected_groups, True)
        .repeat_interleave(experts // groups, dim=1)
    )
    ids = route.masked_fill(~mask, -torch.inf).argsort(dim=-1, descending=True, stable=True)[:, :top_k]
    weights = scores.gather(-1, ids)
    weights /= weights.sum(-1, keepdim=True)
    np.testing.assert_array_equal(data[2], ids.numpy())
    torch.testing.assert_close(torch.from_numpy(data[3]), weights, rtol=3e-5, atol=3e-6)
