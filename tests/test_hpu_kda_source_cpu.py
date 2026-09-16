"""KDA TPC source forward/backward against an independent differentiable recurrence."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.test_hpu_quantized_optimizer_source_cpu import pointers


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Clang is required for native source reference tests")
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("hpu_kda")
    text = '#include "hpu_tpc_reference.h"\n#define ARENO_DTYPE 0\n#include "recurrent_math.h"\n#include "index_io.h"\n'
    for family in ("kda_prepare", "kda"):
        for direction in (0, 1):
            text += (
                f"namespace {family}_{direction} {{\n#define ARENO_DIRECTION {direction}\n#define main kernel\n"
                f'#include "{family}.c"\n#undef main\n#undef ARENO_DIRECTION\n}}\n'
            )
    text += """
extern "C" void prepare(int direction,int tokens,int qh,int heads,int key,int normalize,int dtype,int gdtype,int recurrent,int bounded,float bound,void** p) {
    tensor q{p[0],key,tokens*qh},k{p[1],key,tokens*qh},g{p[2],key,tokens*heads},a{p[3],heads,1},bias{p[4],key,heads},beta{p[5],1,tokens*heads};
    tensor prepared{p[6],3*key+1,tokens*heads},gp{p[7],3*key+1,tokens*heads};
    index_space_origin=int5{}; index_space_extent=int5{tokens*heads,0,0,0,0};
    if (!direction) kda_prepare_0::kernel(q,k,g,a,bias,beta,prepared,tokens,qh,heads,key,normalize,dtype,gdtype,recurrent,bounded,bound,1,20);
    else kda_prepare_1::kernel(q,k,g,a,bias,gp,tensor{p[8],key,tokens*qh},tensor{p[9],key,tokens*qh},tensor{p[10],key,tokens*heads},
        tensor{p[11],heads,1},tensor{p[12],key,heads},tensor{p[13],1,tokens*heads},tokens,qh,heads,key,normalize,dtype,gdtype,recurrent,bounded,bound,1,20);
}
extern "C" void recurrence(int direction,int tokens,int heads,int key,int value,int sequences,int slots,float scale,void** p) {
    tensor prepared{p[0],3*key+1,tokens*heads},v{p[1],value,tokens*heads},initial{p[2],key,slots*heads*value},cu{p[3],sequences+1,1},ids{p[4],sequences,1,8};
    tensor output{p[5],value,tokens*heads},final_state{p[6],key,sequences*heads*value},history{p[7],key,tokens*heads*value};
    index_space_origin=int5{}; index_space_extent=int5{sequences*heads*value,0,0,0,0};
    if (!direction) kda_0::kernel(prepared,v,initial,cu,ids,output,final_state,history,tokens,heads,key,value,sequences,slots,1,scale);
    else kda_1::kernel(prepared,v,history,tensor{p[8],value,tokens*heads},tensor{p[9],key,sequences*heads*value},cu,ids,
        tensor{p[10],3*key+1,tokens*heads},tensor{p[11],value,tokens*heads},tensor{p[12],key,slots*heads*value},tokens,heads,key,value,sequences,slots,1,scale);
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
    native.prepare.argtypes = [ctypes.c_int] * 10 + [ctypes.c_float, ctypes.POINTER(ctypes.c_void_p)]
    native.recurrence.argtypes = [ctypes.c_int] * 7 + [ctypes.c_float, ctypes.POINTER(ctypes.c_void_p)]
    native.prepare.restype = native.recurrence.restype = None
    return native


def reference(q, k, v, g, a, bias, beta, initial, cu, indices, *, dtype, normalize, recurrent, bound, scale):
    heads = v.shape[1]
    if normalize:
        q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        if not recurrent:
            q, k = q.to(dtype).float(), k.to(dtype).float()
    q = q.repeat_interleave(heads // q.shape[1], dim=1)
    k = k.repeat_interleave(heads // k.shape[1], dim=1)
    rate = a.exp()[None, :, None]
    x = g + bias
    gate = bound * (rate * x).sigmoid() if bound is not None else -rate * F.softplus(x)
    if not recurrent:
        gate = gate.to(dtype).float()
    beta = beta.sigmoid() if recurrent else beta
    outputs, states = [], []
    for seq, (first, last) in enumerate(zip(cu, cu[1:])):
        state = initial[indices[seq]] if indices[seq] >= 0 else torch.zeros_like(initial[0])
        for token in range(first, last):
            state = state * gate[token].exp()[:, None, :]
            update = (v[token] - (state * k[token][:, None, :]).sum(-1)) * beta[token][:, None]
            state = state + k[token][:, None, :] * update[:, :, None]
            outputs.append((state * q[token][:, None, :]).sum(-1) * scale)
        states.append(state)
    return torch.stack(outputs), torch.stack(states)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("key", [1, 3, 65, 129])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("bound", [None, -5.0])
@pytest.mark.parametrize("recurrent", [False, True])
def test_kda_forward_and_all_training_gradients(source, dtype, key, normalize, bound, recurrent):
    rng = torch.Generator().manual_seed(23)
    tokens, qh, heads, value, slots = 8, 1, 2, 3, 3
    cu = np.array([0, 0, 2, 5, 8], np.int32)
    indices = np.array([0, 1, 1, -1], np.int64)

    def leaf(shape, factor=1, quantize=True):
        x = torch.randn(shape, generator=rng) * factor
        return (x.to(dtype).float() if quantize else x).requires_grad_()

    q, k = leaf((tokens, qh, key), 0.2), leaf((tokens, qh, key), 0.2)
    v, g = leaf((tokens, heads, value)), leaf((tokens, heads, key), 0.2)
    a, bias = leaf((heads,), 0.2, False), leaf((heads, key), 0.1, False)
    beta, initial = leaf((tokens, heads), 0.2), leaf((slots, heads, value, key), 0.1, False)
    prepared = np.empty((tokens * heads, 3 * key + 1), np.float32)
    grad_prepared = np.zeros_like(prepared)
    leaves = [q, k, g, a, bias, beta]
    data = (
        [x.detach().numpy() for x in leaves]
        + [prepared, grad_prepared]
        + [np.zeros_like(x.detach().numpy()) for x in leaves]
    )
    typecode = [torch.float32, torch.bfloat16, torch.float16].index(dtype)
    source.prepare(
        0,
        tokens,
        qh,
        heads,
        key,
        normalize,
        typecode,
        typecode,
        recurrent,
        bound is not None,
        bound or 0,
        pointers(data),
    )
    grad_out = torch.randn(v.shape, generator=rng)
    grad_final = torch.randn(len(indices), heads, value, key, generator=rng)
    recurrent_data = [
        prepared,
        v.detach().numpy(),
        initial.detach().numpy(),
        cu,
        indices,
        np.empty(v.shape, np.float32),
        np.empty(grad_final.shape, np.float32),
        np.empty((tokens, heads, value, key), np.float32),
        grad_out.numpy(),
        grad_final.numpy(),
        grad_prepared,
        np.zeros(v.shape, np.float32),
        np.zeros(initial.shape, np.float32),
    ]
    scale = 0.37
    source.recurrence(0, tokens, heads, key, value, len(indices), slots, scale, pointers(recurrent_data))
    expected, final = reference(
        q,
        k,
        v,
        g,
        a,
        bias,
        beta,
        initial,
        cu,
        indices,
        dtype=dtype,
        normalize=normalize,
        recurrent=recurrent,
        bound=bound,
        scale=scale,
    )
    torch.testing.assert_close(torch.from_numpy(recurrent_data[5]), expected, rtol=3e-4, atol=2e-6)
    torch.testing.assert_close(torch.from_numpy(recurrent_data[6]), final, rtol=3e-4, atol=3e-6)
    if not recurrent:
        source.recurrence(1, tokens, heads, key, value, len(indices), slots, scale, pointers(recurrent_data))
        source.prepare(
            1,
            tokens,
            qh,
            heads,
            key,
            normalize,
            typecode,
            typecode,
            recurrent,
            bound is not None,
            bound or 0,
            pointers(data),
        )
        ((expected * grad_out).sum() + (final * grad_final).sum()).backward()
        # CUDA's Python gate/normalization casts also round their backward signal.
        # Source gradients are FP32 before the public wrapper's dtype conversions;
        # compare float32 exactly, lower precisions with their cast error budget.
        tolerance = max(5e-4, 5 * torch.finfo(dtype).eps)
        for actual, leaf in zip(data[8:] + [recurrent_data[11], recurrent_data[12]], leaves + [v, initial]):
            torch.testing.assert_close(
                torch.from_numpy(actual), leaf.grad, rtol=tolerance, atol=max(1e-5, 2 * torch.finfo(dtype).eps)
            )
