"""Run the TPC source with host FP32 primitives; no Gaudi ISA is emulated."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def normalization_source(tmp_path_factory):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Clang vector extensions are required for TPC source reference tests")
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("hpu_normalization")
    text = '#include "hpu_tpc_reference.h"\n#define ARENO_DTYPE 0\n#include "tensor_io.h"\n'
    calls = []
    for kind in range(3):
        for direction in range(3):
            if kind == 0 and direction == 2:
                continue
            text += (
                f"namespace norm_{kind}_{direction} {{\n"
                f"#define ARENO_KIND {kind}\n#define ARENO_DIRECTION {direction}\n"
                '#define main kernel\n#include "normalization.c"\n'
                "#undef main\n#undef ARENO_KIND\n#undef ARENO_DIRECTION\n}\n"
            )
            args = ["x"]
            if direction != 0:
                args += ["dy", "inv"]
            if kind >= 1 and direction != 2:
                args += ["w"]
            if kind == 2:
                args += ["g"]
            args += ["y", "inv"] if direction == 0 else ["dx", "dg"] if direction == 1 and kind == 2 else ["dx"] if direction == 1 else ["dw"]
            args += ["hidden", "rows", "eps"]
            calls.append(f"case {kind * 3 + direction}: norm_{kind}_{direction}::kernel({', '.join(args)}); break;")
    text += (
        'extern "C" void run(int kind, int direction, int rows, int hidden, float eps, '
        'float* xp, float* wp, float* gp, float* dyp, float* yp, float* invp, float* dxp, float* dgp, float* dwp) {\n'
        'tensor x{xp,hidden,rows}, w{wp,hidden,1}, g{gp,hidden,rows}, dy{dyp,hidden,rows};\n'
        'tensor y{yp,hidden,rows}, inv{invp,1,rows}, dx{dxp,hidden,rows}, dg{dgp,hidden,rows}, dw{dwp,hidden,1};\n'
        'index_space_extent = int5{direction == 2 ? (hidden + 127) / 128 : rows, 0, 0, 0, 0};\n'
        'switch (kind * 3 + direction) {\n' + '\n'.join(calls) + '\n}\n}\n'
    )
    source = build / "normalization.cpp"
    source.write_text(text)
    library = build / "normalization.so"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-Wno-psabi", "-I", str(root / "tests"),
         "-I", str(root / "areno/accel/csrc/hpu"), str(source), "-o", str(library)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    native = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    native.run.argtypes = [ctypes.c_int] * 4 + [ctypes.c_float] + [array] * 9
    native.run.restype = None
    return native


@pytest.mark.parametrize("kind", range(3), ids=["unscaled", "scaled", "silu_gate"])
@pytest.mark.parametrize("rows,hidden", [(1, 1), (3, 63), (4, 64), (7, 65), (3, 127), (5, 128), (2, 129), (9, 1025)])
@pytest.mark.parametrize("zero", [False, True])
def test_normalization_source_forward_backward(normalization_source, kind, rows, hidden, zero):
    generator = torch.Generator().manual_seed(12)
    x = torch.randn(rows, hidden, generator=generator)
    if zero:
        x.zero_()
    x.requires_grad_()
    w = torch.randn(hidden, generator=generator, requires_grad=True)
    g = torch.randn(rows, hidden, generator=generator, requires_grad=True)
    dy = torch.randn(rows, hidden, generator=generator)
    eps = 1e-5
    expected_inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    expected = x * expected_inv
    if kind >= 1:
        expected = expected * w
    if kind == 2:
        expected = expected * F.silu(g)
    expected.backward(dy)
    arrays = [t.detach().contiguous().numpy() for t in (x, w, g, dy)]
    arrays += [np.full(shape, np.nan, dtype=np.float32) for shape in
               ((rows, hidden), (rows, 1), (rows, hidden), (rows, hidden), (hidden,))]
    for direction in range(3 if kind >= 1 else 2):
        normalization_source.run(kind, direction, rows, hidden, eps, *arrays)
    torch.testing.assert_close(torch.from_numpy(arrays[4]), expected.detach(), atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(torch.from_numpy(arrays[5]), expected_inv.detach(), atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(torch.from_numpy(arrays[6]), x.grad, atol=3e-5, rtol=3e-5)
    if kind == 2:
        torch.testing.assert_close(torch.from_numpy(arrays[7]), g.grad, atol=3e-6, rtol=3e-5)
    if kind >= 1:
        torch.testing.assert_close(torch.from_numpy(arrays[8]), w.grad, atol=3e-6, rtol=3e-5)
