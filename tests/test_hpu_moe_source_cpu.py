"""Execute native MoE routing and reduction source with host TPC primitives."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_hpu_attention_source_cpu import DTYPES, floats, storage
from tests.test_hpu_quantized_optimizer_source_cpu import pointers


@pytest.fixture(scope="module", params=range(3))
def source(tmp_path_factory, request):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Clang is required for native source reference tests")
    dtype = request.param
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("hpu_moe")
    text = (
        '#include "hpu_tpc_reference.h"\n'
        + f'#define ARENO_DTYPE {dtype}\n#include "tensor_io.h"\n#include "index_io.h"\n'
    )
    for family, kinds in (("moe", range(5)), ("moe_align", (0,)), ("fused_moe", (0, 1)), ("normalization", (3,))):
        for kind in kinds:
            text += (
                f"namespace {family}_{kind} {{\n#define ARENO_KIND {kind}\n#define ARENO_DIRECTION 0\n#define main kernel\n"
                f'#include "{family}.c"\n#undef main\n#undef ARENO_KIND\n#undef ARENO_DIRECTION\n}}\n'
            )
    text += f"constexpr int bytes={4 if dtype == 0 else 2};\n"
    text += r"""
extern "C" void counts(int topk,int tokens,int experts,int k,int start,void** p) {
    tensor routes{p[0],k,tokens,topk ? 8 : 1},weights{p[1],k,tokens},count{p[2],experts,1,8};
    index_space_origin=int5{}; index_space_extent=int5{experts,0,0,0,0};
    if (topk) moe_1::kernel(routes,weights,count,tokens,0,experts,k,start,0);
    else moe_0::kernel(routes,weights,count,tokens,0,experts,k,start,0);
}
extern "C" void permute(int topk,int tokens,int hidden,int experts,int k,int start,int rows,void** p) {
    tensor x{p[0],hidden,tokens,bytes},routes{p[1],k,tokens,topk ? 8 : 1},weights{p[2],k,tokens},counts{p[3],experts,1,8};
    tensor output{p[4],hidden,rows,bytes},rw{p[5],rows,1},ids{p[6],rows,1,8},positions{p[7],rows,1};
    index_space_origin=int5{}; index_space_extent=int5{(hidden+127)/128,experts,0,0,0};
    if (topk) moe_3::kernel(x,routes,weights,counts,output,rw,ids,positions,tokens,hidden,experts,k,start,rows);
    else moe_2::kernel(x,routes,weights,counts,output,rw,ids,tokens,hidden,experts,k,start,rows);
}
extern "C" void weight_grad(int rows,int tokens,int k,void** p) {
    index_space_origin=int5{}; index_space_extent=int5{rows,0,0,0,0};
    moe_4::kernel(tensor{p[0],rows,1},tensor{p[1],rows,1,8},tensor{p[2],rows,1},tensor{p[3],k,tokens},tokens,0,0,k,0,rows);
}
extern "C" void align(int n,int experts,int block,int cap,int bcap,int scratch,int pad,void** p) {
    index_space_origin=int5{}; index_space_extent=int5{1,0,0,0,0};
    moe_align_0::kernel(tensor{p[0],n,1,8},tensor{p[1],cap,1},tensor{p[2],bcap,1},tensor{p[3],scratch,1},
        tensor{p[4],cap,1},tensor{p[5],bcap,1},tensor{p[6],1,1},tensor{p[7],scratch,1},n,experts,block,cap,bcap,scratch,pad);
}
extern "C" void reduce(int rows,int tokens,int hidden,int k,float scale,void** p) {
    index_space_origin=int5{}; index_space_extent=int5{(hidden+127)/128,rows,0,0,0};
    tensor expanded{p[4],hidden,tokens*k,bytes};
    fused_moe_0::kernel(tensor{p[0],hidden,rows},tensor{p[1],rows,1},tensor{p[2],rows,1,8},tensor{p[3],rows,1},expanded,rows,tokens,hidden,k,scale);
    index_space_extent=int5{(hidden+127)/128,tokens,0,0,0};
    fused_moe_1::kernel(expanded,tensor{p[5],hidden,tokens,bytes},rows,tokens,hidden,k,scale);
}
extern "C" void norm(int rows,int groups,int hidden,float eps,void** p) {
    index_space_origin=int5{}; index_space_extent=int5{rows,0,0,0,0};
    normalization_3::kernel(tensor{p[0],hidden,rows,bytes},tensor{p[1],hidden,groups},tensor{p[2],hidden,rows,bytes},
        tensor{p[3],hidden,rows,bytes},tensor{p[4],1,rows},hidden,rows,eps,groups);
}
"""
    path = build / "source.cpp"
    path.write_text(text)
    lib = build / "source.so"
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
            str(lib),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    native = ctypes.CDLL(str(lib))
    for name, integers, extra in (
        ("counts", 5, []),
        ("permute", 7, []),
        ("weight_grad", 3, []),
        ("align", 7, []),
        ("reduce", 4, [ctypes.c_float]),
        ("norm", 3, [ctypes.c_float]),
    ):
        fn = getattr(native, name)
        fn.argtypes = [ctypes.c_int] * integers + extra + [ctypes.POINTER(ctypes.c_void_p)]
        fn.restype = None
    return native, DTYPES[dtype]


@pytest.mark.parametrize("hidden", [1, 65, 129, 257])
@pytest.mark.parametrize("topk", [False, True])
@pytest.mark.parametrize("start", [0, 3])
def test_counts_permute_and_weight_gradient(source, hidden, topk, start):
    native, dtype = source
    rng = torch.Generator().manual_seed(93)
    tokens, experts, k = 9, 5, 4 if topk else 5
    x = torch.randn(tokens, hidden, generator=rng).to(dtype)
    weights = torch.randn(tokens, k, generator=rng)
    weights[1:3] = 0
    if topk:
        routes = torch.randint(-1, 10, (tokens, k), generator=rng).numpy().astype(np.int64)
        routes[0, 0] = 2**32 + start
    else:
        routes = (torch.rand(tokens, k, generator=rng) > 0.6).numpy().astype(np.uint8)
    counts = np.empty(experts, np.int64)
    native.counts(topk, tokens, experts, k, start, pointers([routes, weights.numpy(), counts]))
    active = [[] for _ in range(experts)]
    for e in range(experts):
        for token in range(tokens):
            for pos in range(k):
                if (topk and routes[token, pos] == e + start and weights[token, pos] != 0) or (
                    not topk and pos == e and routes[token, e]
                ):
                    active[e].append((token, pos))
    np.testing.assert_array_equal(counts, [len(v) for v in active])
    flat = [v for expert in active for v in expert]
    rows = len(flat)
    data = [
        storage(x, dtype),
        routes,
        weights.numpy(),
        counts,
        storage(torch.empty(rows, hidden), dtype),
        np.empty(rows, np.float32),
        np.empty(rows, np.int64),
        np.empty(rows, np.int32),
    ]
    native.permute(topk, tokens, hidden, experts, k, start, rows, pointers(data))
    np.testing.assert_array_equal(data[6], [t for t, _ in flat])
    torch.testing.assert_close(floats(data[4], dtype), x[[t for t, _ in flat]].float(), rtol=0, atol=0)
    torch.testing.assert_close(torch.from_numpy(data[5]), torch.stack([weights[t, p] for t, p in flat]), rtol=0, atol=0)
    if topk:
        np.testing.assert_array_equal(data[7], [p for _, p in flat])
        grad = torch.randn(rows, generator=rng).numpy()
        output = np.zeros((tokens, k), np.float32)
        native.weight_grad(rows, tokens, k, pointers([grad, data[6], data[7], output]))
        expected = np.zeros_like(output)
        for g, (t, p) in zip(grad, flat):
            expected[t, p] = g
        np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("block", [1, 4, 16])
@pytest.mark.parametrize("pad", [False, True])
def test_align_preserves_tail_and_sentinel_expert(source, block, pad):
    native, _ = source
    ids = np.array([-1, 2, 1, 2, 0, 2, -1, 3, 0, 2**32, -3], np.int64)
    experts = 5
    capacity = len(ids) + experts * (block - 1)
    bcap = (capacity + block - 1) // block
    data = [
        ids,
        np.full(capacity, -7, np.int32),
        np.full(bcap, -8, np.int32),
        np.full(experts + 3, -9, np.int32),
        np.empty(capacity, np.int32),
        np.empty(bcap, np.int32),
        np.empty(1, np.int32),
        np.empty(experts + 3, np.int32),
    ]
    native.align(len(ids), experts, block, capacity, bcap, experts + 3, pad, pointers(data))
    sorted_ids = np.full(capacity, len(ids) if pad else -7, np.int32)
    block_experts = data[2].copy()
    total = 0
    for e in range(-1, experts - 1):
        routes = np.flatnonzero(ids == e)
        sorted_ids[total : total + len(routes)] = routes
        size = (len(routes) + block - 1) // block * block
        block_experts[total // block : (total + size) // block] = e
        total += size
    np.testing.assert_array_equal(data[4], sorted_ids)
    np.testing.assert_array_equal(data[5], block_experts)
    assert data[6][0] == total
    np.testing.assert_array_equal(data[7][-2:], [-9, -9])


@pytest.mark.parametrize("hidden", [1, 65, 129])
@pytest.mark.parametrize("scale", [0.3, 1.0, 2.5])
def test_fused_reduction_rounds_weighted_routes_before_sum(source, hidden, scale):
    native, dtype = source
    rng = torch.Generator().manual_seed(12)
    tokens, k = 5, 3
    indices = torch.tensor([8, 0, 12, 4, 2, 6, 1, 13])
    down = torch.randn(len(indices), hidden, generator=rng)
    weights = torch.randn(len(indices), generator=rng)
    data = [
        down.numpy(),
        weights.numpy(),
        (indices // k).numpy(),
        (indices % k).int().numpy(),
        storage(torch.zeros(tokens * k, hidden), dtype),
        storage(torch.empty(tokens, hidden), dtype),
    ]
    native.reduce(len(indices), tokens, hidden, k, scale, pointers(data))
    expected = torch.zeros(tokens * k, hidden, dtype=dtype)
    expected[indices] = (down * weights[:, None]).to(dtype)
    torch.testing.assert_close(floats(data[4], dtype), expected.float(), rtol=0, atol=0)
    expected = (expected.float().view(tokens, k, hidden).sum(1) * scale).to(dtype).float()
    torch.testing.assert_close(floats(data[5], dtype), expected, rtol=0, atol=0)


@pytest.mark.parametrize("hidden", [1, 65, 129, 257])
@pytest.mark.parametrize("groups", [1, 3])
def test_grouped_norm_uses_sigmoid_gate(source, hidden, groups):
    native, dtype = source
    rng = torch.Generator().manual_seed(6)
    x = torch.randn(4, groups, hidden, generator=rng).to(dtype).float()
    gate = torch.randn(x.shape, generator=rng).to(dtype).float()
    w = torch.randn(groups, hidden, generator=rng).to(dtype).float()
    data = [
        storage(x, dtype),
        w.numpy(),
        storage(gate, dtype),
        storage(torch.empty_like(x), dtype),
        np.empty((4, groups), np.float32),
    ]
    native.norm(4 * groups, groups, hidden, 1e-5, pointers(data))
    inv = (x.square().mean(-1) + 1e-5).rsqrt()
    expected = (x * inv[..., None] * w * gate.sigmoid()).to(dtype).float()
    torch.testing.assert_close(torch.from_numpy(data[4]), inv, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(floats(data[3], dtype), expected, rtol=max(torch.finfo(dtype).eps, 2e-5), atol=2e-6)
