"""Interpret the linear TPC source on the host; MME GEMM needs HPU validation."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def linear_source(tmp_path_factory):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Clang vector extensions are required for TPC source reference tests")
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("hpu_linear")
    source = build / "linear.cpp"
    source.write_text(
        '#include "hpu_tpc_reference.h"\n#define ARENO_DTYPE 0\n#include "tensor_io.h"\n'
        'namespace forward {\n#define ARENO_DIRECTION 0\n#define main kernel\n#include "linear.c"\n'
        '#undef main\n#undef ARENO_DIRECTION\n}\n'
        'namespace backward {\n#define ARENO_DIRECTION 1\n#define main kernel\n#include "linear.c"\n'
        '#undef main\n#undef ARENO_DIRECTION\n}\n'
        'extern "C" void run(int rows, int hidden, float* xp, float* bp, float* yp, float* dbp) {\n'
        'tensor x{xp,hidden,rows}, b{bp,hidden,1}, y{yp,hidden,rows}, db{dbp,hidden,1};\n'
        'index_space_extent = int5{(hidden+127)/128,rows,0,0,0};\n'
        'forward::kernel(x,b,y,hidden,rows,0);\nbackward::kernel(x,db,hidden,rows,0);\n}\n'
    )
    library = build / "linear.so"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-Wno-psabi", "-I", str(root / "tests"),
         "-I", str(root / "areno/accel/csrc/hpu"), str(source), "-o", str(library)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    native = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    native.run.argtypes = [ctypes.c_int] * 2 + [array] * 4
    native.run.restype = None
    return native


@pytest.mark.parametrize("rows,hidden", [(1, 1), (7, 63), (3, 64), (5, 65), (7, 127), (3, 128), (5, 129), (17, 1025)])
def test_bias_source_add_and_reduce(linear_source, rows, hidden):
    generator = torch.Generator().manual_seed(18)
    x = torch.randn(rows, hidden, generator=generator)
    bias = torch.randn(hidden, generator=generator)
    y = np.full((rows, hidden), np.nan, dtype=np.float32)
    db = np.full(hidden, np.nan, dtype=np.float32)
    linear_source.run(rows, hidden, x.numpy(), bias.numpy(), y, db)
    torch.testing.assert_close(torch.from_numpy(y), x + bias)
    torch.testing.assert_close(torch.from_numpy(db), x.sum(0), atol=2e-6, rtol=2e-5)
