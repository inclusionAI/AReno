"""Host interpretation of attention/embedding TPC source, independent of HPU ISA."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_hpu_quantized_optimizer_source_cpu import pointers

DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def storage(tensor, dtype):
    value = tensor.to(dtype).contiguous()
    return value.view(torch.uint16).numpy().copy() if dtype == torch.bfloat16 else value.numpy().copy()


def floats(array, dtype):
    value = torch.from_numpy(array.copy())
    return value.view(torch.bfloat16).float() if dtype == torch.bfloat16 else value.float()


@pytest.fixture(scope="module", params=range(3))
def attention_source(tmp_path_factory, request):
    dtype = request.param
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Clang is required for the TPC source interpreter")
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("hpu_attention")
    text = (
        '#include "hpu_tpc_reference.h"\n'
        + f'#define ARENO_DTYPE {dtype}\n#include "tensor_io.h"\n#include "index_io.h"\n'
    )
    for kind in (0, 1, 2):
        for direction in (0, 1):
            if kind == 2 and direction == 1:
                continue
            text += (
                f"namespace attention_{kind}_{direction} {{\n#define ARENO_KIND {kind}\n#define ARENO_DIRECTION {direction}\n#define main kernel\n"
                '#include "attention.c"\n#undef main\n#undef ARENO_DIRECTION\n#undef ARENO_KIND\n}\n'
            )
    for direction in (0, 1):
        text += (
            f"namespace embedding_{direction} {{\n#define ARENO_DIRECTION {direction}\n#define main kernel\n"
            '#include "embedding.c"\n#undef main\n#undef ARENO_DIRECTION\n}\n'
        )
    for kind in (0, 1):
        text += (
            f"namespace cache_{kind} {{\n#define ARENO_KIND {kind}\n#define main kernel\n"
            '#include "paged_cache.c"\n#undef main\n#undef ARENO_KIND\n}\n'
        )
    text += (
        'extern "C" void attention(int packed,int qr,int kr,int hidden,int qh,int kh,int ql,int kl,int start,int window,int seqs,float scale,void** p) {\n'
        f"auto t=[&](int i,int rows) {{ return tensor{{p[i],hidden,rows,{4 if dtype == 0 else 2}}}; }};\n"
        "tensor q=t(0,qr), k=t(1,kr), v=t(2,kr), grad=t(3,qr), out=t(4,qr);\n"
        "tensor dq{p[5],hidden,qr}, dk{p[6],hidden,kr}, dv{p[7],hidden,kr}, cu{p[8],seqs+1,1};\n"
        "index_space_origin=int5{}; index_space_extent=int5{(hidden+127)/128,qr,0,0,0};\n"
        "#define PARAMS qr,kr,hidden,qh,kh,ql,kl,start,window,seqs,scale,0,0\n"
        "if (packed) { attention_1_0::kernel(q,k,v,cu,out,PARAMS); attention_1_1::kernel(q,k,v,grad,out,cu,dq,dk,dv,PARAMS); }\n"
        "else { attention_0_0::kernel(q,k,v,out,PARAMS); attention_0_1::kernel(q,k,v,grad,out,dq,dk,dv,PARAMS); }\n}\n"
        'extern "C" void embedding(int tokens,int hidden,int start,int end,void** p) {\n'
        f"auto t=[&](int i,int rows) {{ return tensor{{p[i],hidden,rows,{4 if dtype == 0 else 2}}}; }};\n"
        "tensor ids{p[0],tokens,1,8};\n"
        "index_space_origin=int5{}; index_space_extent=int5{(hidden+127)/128,tokens,0,0,0};\n"
        "embedding_0::kernel(ids,t(1,end-start),t(3,tokens),tokens,hidden,start,end);\n"
        "embedding_1::kernel(ids,t(2,tokens),t(4,end-start),tokens,hidden,start,end);\n}\n"
        'extern "C" void paged(int batch,int slots,int block,int blocks,int qh,int kh,int hidden,int window,void** p) {\n'
        f"auto t=[&](int i,int rows) {{ return tensor{{p[i],hidden,rows,{4 if dtype == 0 else 2}}}; }};\n"
        "tensor table{p[5],blocks,batch}, lengths{p[6],batch,1}, owners{p[10],slots,1};\n"
        "index_space_origin=int5{}; index_space_extent=int5{batch,0,0,0,0};\n"
        "cache_0::kernel(table,lengths,owners,batch,slots,block,blocks,kh,hidden);\n"
        "index_space_extent=int5{(hidden+127)/128,slots*kh,0,0,0};\n"
        "cache_1::kernel(t(1,slots*kh),t(2,slots*kh),t(3,batch*kh),t(4,batch*kh),owners,t(8,slots*kh),t(9,slots*kh),batch,slots,block,blocks,kh,hidden);\n"
        "index_space_extent=int5{(hidden+127)/128,batch*qh,0,0,0};\n"
        "attention_2_0::kernel(t(0,batch*qh),t(8,slots*kh),t(9,slots*kh),table,lengths,t(7,batch*qh),"
        "batch*qh,slots*kh,hidden,qh,kh,1,slots,0,window,0,1.0f/std::sqrt(float(hidden)),block,blocks);\n}\n"
    )
    source = build / "attention.cpp"
    source.write_text(text)
    library = build / "attention.so"
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
            str(source),
            "-o",
            str(library),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    native = ctypes.CDLL(str(library))
    native.attention.argtypes = [ctypes.c_int] * 11 + [ctypes.c_float, ctypes.POINTER(ctypes.c_void_p)]
    native.embedding.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_void_p)]
    native.paged.argtypes = [ctypes.c_int] * 8 + [ctypes.POINTER(ctypes.c_void_p)]
    native.attention.restype = native.embedding.restype = native.paged.restype = None
    return native, DTYPES[dtype]


def reference(q, k, v, grad, start, window, dtype, saved=None):
    scale = q.shape[-1] ** -0.5
    scores = q @ k.transpose(-1, -2) * scale
    position = torch.arange(start, start + q.shape[-2]).view(-1, 1)
    key = torch.arange(k.shape[-2]).view(1, -1)
    mask = key <= position
    if window >= 0:
        mask &= key >= position - window
    probabilities = scores.masked_fill(~mask, -torch.inf).softmax(-1)
    out = (probabilities @ v).to(dtype).float()
    # CUDA backward consumes the saved output in its storage dtype.
    saved = out if saved is None else saved
    ds = probabilities * ((grad @ v.transpose(-1, -2)) - (grad * saved).sum(-1, keepdim=True)) * scale
    return out, ds @ k, ds.transpose(-1, -2) @ q, probabilities.transpose(-1, -2) @ grad


@pytest.mark.parametrize("hidden", [1, 63, 64, 65, 127, 128, 129, 257])
@pytest.mark.parametrize("window", [-1, 0, 2])
def test_dense_attention_forward_backward(attention_source, hidden, window):
    native, dtype = attention_source
    rng = torch.Generator().manual_seed(62)
    values = [torch.randn(2, 2, length, hidden, generator=rng).to(dtype).float() for length in (3, 5, 5, 3)]
    q, k, v, grad = values
    data = [storage(x, dtype) for x in values]
    data += [np.empty_like(data[0])] + [np.zeros(x.shape, dtype=np.float32) for x in (q, k, v)]
    data += [np.array([0], dtype=np.int32)]
    native.attention(0, 12, 20, hidden, 2, 2, 3, 5, 2, window, 0, hidden**-0.5, pointers(data))
    expected = reference(q, k, v, grad, 2, window, dtype, saved=floats(data[4], dtype))
    actual = [floats(data[4], dtype)] + [torch.from_numpy(x) for x in data[5:8]]
    for i, (result, ref) in enumerate(zip(actual, expected)):
        tolerance = torch.finfo(dtype).eps if i == 0 else 3e-4
        torch.testing.assert_close(result, ref, rtol=max(tolerance, 3e-4), atol=2e-5)


@pytest.mark.parametrize("hidden", [1, 65, 129])
@pytest.mark.parametrize("heads", [(1, 1), (4, 2), (4, 1)])
@pytest.mark.parametrize("window", [-1, 0, 2])
def test_packed_attention_gqa_empty_sequences(attention_source, hidden, heads, window):
    native, dtype = attention_source
    qh, kh = heads
    boundaries = [0, 0, 2, 2, 5, 9]
    rng = torch.Generator().manual_seed(77)
    values = [torch.randn(9, h, hidden, generator=rng).to(dtype).float() for h in (qh, kh, kh, qh)]
    q, k, v, grad = values
    data = [storage(x, dtype) for x in values]
    data += [np.empty_like(data[0])] + [np.zeros(x.shape, dtype=np.float32) for x in (q, k, v)]
    data += [np.array(boundaries, dtype=np.int32)]
    native.attention(
        1, 9 * qh, 9 * kh, hidden, qh, kh, 9, 9, 0, window, len(boundaries) - 1, hidden**-0.5, pointers(data)
    )
    expected = [torch.zeros_like(x) for x in (q, q, k, v)]
    for begin, end in zip(boundaries, boundaries[1:]):
        if begin == end:
            continue
        qr, kr, vr, gr = [x[begin:end].transpose(0, 1) for x in values]
        kr, vr = [x.repeat_interleave(qh // kh, dim=0) for x in (kr, vr)]
        saved = floats(data[4], dtype)[begin:end].transpose(0, 1)
        ref = reference(qr, kr, vr, gr, 0, window, dtype, saved=saved)
        for i, result in enumerate(ref):
            if i >= 2:
                result = result.reshape(kh, qh // kh, end - begin, hidden).sum(1)
            expected[i][begin:end] = result.transpose(0, 1)
    actual = [floats(data[4], dtype)] + [torch.from_numpy(x) for x in data[5:8]]
    for i, (result, ref) in enumerate(zip(actual, expected)):
        tolerance = torch.finfo(dtype).eps if i == 0 else 3e-4
        torch.testing.assert_close(result, ref, rtol=max(tolerance, 3e-4), atol=2e-5)


@pytest.mark.parametrize("hidden", [1, 65, 129])
@pytest.mark.parametrize("window", [-1, 0, 2])
def test_paged_decode_updates_only_selected_slots(attention_source, hidden, window):
    native, dtype = attention_source
    rng = torch.Generator().manual_seed(39)
    values = [
        torch.randn(shape, generator=rng).to(dtype).float()
        for shape in ((2, 4, hidden), (6, 4, 2, hidden), (6, 4, 2, hidden), (2, 2, hidden), (2, 2, hidden))
    ]
    table = np.array([[2, 0, 1], [3, 4, 5]], dtype=np.int32)
    lengths = np.array([5, 0], dtype=np.int32)
    data = [storage(x, dtype) for x in values] + [table, lengths]
    data += [np.empty_like(data[0]), np.empty_like(data[1]), np.empty_like(data[2]), np.full(24, -1, dtype=np.int32)]
    native.paged(2, 24, 4, 3, 4, 2, hidden, window, pointers(data))
    q, kc, vc, ku, vu = values
    expected_k, expected_v = kc.clone(), vc.clone()
    expected = []
    for b, length in enumerate(lengths):
        page, offset = table[b, length // 4], length % 4
        expected_k[page, offset] = ku[b]
        expected_v[page, offset] = vu[b]
        positions = torch.arange(int(length) + 1)
        pages = torch.from_numpy(table[b])[positions // 4]
        kr, vr = [x[pages, positions % 4].repeat_interleave(2, dim=1).transpose(0, 1) for x in (expected_k, expected_v)]
        query = q[b].unsqueeze(1)
        expected.append(reference(query, kr, vr, torch.zeros_like(query), int(length), window, dtype)[0].squeeze(1))
    torch.testing.assert_close(
        floats(data[7], dtype), torch.stack(expected), rtol=max(3e-4, torch.finfo(dtype).eps), atol=2e-5
    )
    torch.testing.assert_close(floats(data[8], dtype), expected_k, rtol=0, atol=0)
    torch.testing.assert_close(floats(data[9], dtype), expected_v, rtol=0, atol=0)


@pytest.mark.parametrize("hidden", [1, 63, 64, 65, 127, 128, 129, 257])
def test_vocab_embedding_tp_range_and_repeated_ids(attention_source, hidden):
    native, dtype = attention_source
    ids = np.array([7, 8, 8, -1, 11, 12, 7, 2**32 + 8, -(2**32) + 8, 2**63 - 1], dtype=np.int64)
    weight = torch.linspace(-1, 1, 5 * hidden).reshape(5, hidden).to(dtype)
    grad = torch.linspace(-0.5, 0.5, len(ids) * hidden).reshape(len(ids), hidden).to(dtype)
    data = [
        ids,
        storage(weight, dtype),
        storage(grad, dtype),
        storage(torch.empty_like(grad), dtype),
        storage(torch.zeros_like(weight), dtype),
    ]
    native.embedding(len(ids), hidden, 7, 12, pointers(data))
    expected = torch.zeros_like(grad)
    expected_grad = torch.zeros_like(weight)
    for i, index in enumerate(ids):
        if 7 <= index < 12:
            expected[i] = weight[index - 7]
            expected_grad[index - 7] += grad[i]
    torch.testing.assert_close(floats(data[3], dtype), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(floats(data[4], dtype), expected_grad.float(), rtol=0, atol=0)
