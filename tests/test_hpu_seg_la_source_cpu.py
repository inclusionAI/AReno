"""Native segmented attention: prefill, decode, MTP snapshots and tree masks."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_hpu_quantized_optimizer_source_cpu import pointers


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Clang is required for native source reference tests")
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("hpu_seg_la")
    text = '#include "hpu_tpc_reference.h"\n#define ARENO_DTYPE 0\n#include "recurrent_math.h"\n#include "index_io.h"\n'
    for kind in range(4):
        text += f'namespace seg_{kind} {{\n#define ARENO_KIND {kind}\n#define main kernel\n#include "seg_la.c"\n#undef main\n#undef ARENO_KIND\n}}\n'
    text += 'namespace update {\n#define main kernel\n#include "state_update.c"\n#undef main\n}\n'
    text += """
extern "C" void run(int kind,int n,int h,int d,int seq,int slots,int steps,int masksize,int dtype,float scale,void** p) {
    tensor q{p[0],d,n*h},k{p[1],d,n*h},v{p[2],d,n*h},state{p[3],d,slots*h*d},decay{p[4],h,1};
    tensor offsets{p[5],seq+1,1},lengths{p[6],seq,1},indices{p[7],seq,1,8},scales{p[8],seq,1},mask{p[9],masksize,seq*masksize,1};
    tensor out{p[10],d,n*h},final_state{p[11],d,seq*h*d},cache{p[12],d,seq*steps*h*d};
    index_space_origin=int5{}; index_space_extent=int5{seq*h*d,0,0,0,0};
    #define INPUTS q,k,v,state,decay,offsets,lengths,indices,scales
    #define PARAMS n,h,d,seq,slots,steps,masksize,dtype,scale
    if (kind == 0) seg_0::kernel(INPUTS,out,final_state,PARAMS);
    if (kind == 1) seg_1::kernel(INPUTS,out,final_state,PARAMS);
    if (kind == 2) seg_2::kernel(INPUTS,out,final_state,cache,PARAMS);
    if (kind == 3) seg_3::kernel(INPUTS,mask,out,final_state,PARAMS);
}
extern "C" void state_update(int slots,int sequences,int width,void** p) {
    index_space_origin=int5{}; index_space_extent=int5{(width+127)/128,slots,0,0,0};
    update::kernel(tensor{p[0],width,slots},tensor{p[1],width,sequences},tensor{p[2],sequences,1,8},tensor{p[3],width,slots},slots,sequences,width);
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
    native.run.argtypes = [ctypes.c_int] * 9 + [ctypes.c_float, ctypes.POINTER(ctypes.c_void_p)]
    native.state_update.argtypes = [ctypes.c_int] * 3 + [ctypes.POINTER(ctypes.c_void_p)]
    native.run.restype = native.state_update.restype = None
    return native


@pytest.mark.parametrize("kind", range(4))
@pytest.mark.parametrize("hidden", [4, 33, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_all_segmented_attention_modes(source, kind, hidden, dtype):
    rng = torch.Generator().manual_seed(16)
    sequences, slots, heads = 3, 4, 2
    tokens = 3 if kind == 1 else 6 if kind == 2 else 9 if kind == 3 else 8
    cu = (
        np.array([0, 2, 5, 8], np.int32)
        if kind == 0
        else np.arange(sequences + 1, dtype=np.int32) * (tokens // sequences)
    )
    lengths = np.diff(cu).copy()
    indices = np.array([2, -1, 0], np.int64)
    scales = np.array([0, 1, 1], np.float32)
    q, k, v = [(torch.randn(tokens, heads, hidden, generator=rng) * 0.2).to(dtype).float() for _ in range(3)]
    initial = (torch.randn(slots, heads, hidden, hidden, generator=rng) * 0.1).to(dtype).float()
    decay = torch.tensor([0.03, 0.2])
    mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=torch.uint8).repeat(sequences, 1, 1)
    steps = tokens // sequences if kind == 2 else 0
    scale = 0.3
    data = [
        q.numpy(),
        k.numpy(),
        v.numpy(),
        initial.transpose(-1, -2).contiguous().numpy(),
        decay.numpy(),
        cu,
        lengths,
        indices,
        scales,
        mask.numpy(),
        np.zeros(q.shape, np.float32),
        np.zeros((sequences, heads, hidden, hidden), np.float32),
        np.zeros((sequences, steps, heads, hidden, hidden), np.float32),
    ]
    typecode = [torch.float32, torch.bfloat16, torch.float16].index(dtype)
    source.run(kind, tokens, heads, hidden, sequences, slots, steps, 3, typecode, scale, pointers(data))
    expected = torch.zeros_like(q)
    error_budget = torch.zeros_like(q)
    final = torch.zeros_like(torch.from_numpy(data[11]))
    snapshots = torch.zeros_like(torch.from_numpy(data[12]))
    split = 128 if kind == 1 else 32
    for seq, (first, last) in enumerate(zip(cu, cu[1:])):
        if indices[seq] < 0:
            continue
        state = initial[indices[seq]].clone() if kind in (1, 2) or scales[seq] else torch.zeros_like(initial[0])
        for t, token in enumerate(range(first, last)):
            if kind == 3:
                depth = mask[seq, t].sum() - 1
                candidate = state * torch.exp(-decay * (depth + 1))[:, None, None]
                for p in range(last - first):
                    if mask[seq, t, p]:
                        prev_depth = mask[seq, p].sum() - 1
                        candidate = (
                            candidate
                            + k[first + p][:, :, None]
                            * v[first + p][:, None, :]
                            * torch.exp(-decay * (depth - prev_depth))[:, None, None]
                        )
            else:
                state = state * torch.exp(-decay)[:, None, None] + k[token][:, :, None] * v[token][:, None, :]
                candidate = state
            partials = []
            for start in range(0, hidden, split):
                partial = (q[token, :, start : start + split, None] * candidate[:, start : start + split, :]).sum(
                    1
                ) * scale
                partials.append(partial.to(dtype).float())
            expected[token] = torch.stack(partials).sum(0)
            # A rounded partial can cross a dtype boundary before cancellation
            # in the final sum. Bound its error before summing the partials.
            error_budget[token] = torch.stack(partials).abs().sum(0) * max(3e-4, torch.finfo(dtype).eps) + 3e-6
            if kind == 2:
                snapshots[seq, t] = state.transpose(-1, -2)
        final[seq] = state.transpose(-1, -2)
    assert torch.all((torch.from_numpy(data[10]) - expected).abs() <= error_budget)
    torch.testing.assert_close(torch.from_numpy(data[11]), final, rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(torch.from_numpy(data[12]), snapshots, rtol=3e-4, atol=3e-6)


@pytest.mark.parametrize("width", [1, 65, 129, 513])
def test_state_update_preserves_unselected_slots_and_nan_bits(source, width):
    old = np.arange(5 * width, dtype=np.float32).reshape(5, width)
    old.view(np.uint32)[0, 0] = 0x7FC0AABB
    new = np.arange(3 * width, dtype=np.float32).reshape(3, width) + 0.25
    indices = np.array([3, -1, 1], np.int64)
    result = np.empty_like(old)
    source.state_update(5, 3, width, pointers([old, new, indices, result]))
    expected = old.copy()
    expected[3] = new[0]
    expected[1] = new[2]
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))
