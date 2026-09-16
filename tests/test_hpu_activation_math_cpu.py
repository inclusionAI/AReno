"""Numerical checks for the FP32 formulas; this does not emulate the Gaudi ISA."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def activation_math(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is required to check native activation formulas")
    root = Path(__file__).resolve().parents[1] / "areno/accel/csrc/hpu"
    build = tmp_path_factory.mktemp("hpu_activation_math")
    source = build / "math.cpp"
    source.write_text(
        "#include <cmath>\n"
        "#define ARENO_FLOAT float\n"
        "#define ARENO_EXP(x) std::exp(x)\n"
        "#define ARENO_LOG(x) std::log(x)\n"
        "#define ARENO_RECIP(x) (1.0f / (x))\n"
        "#define ARENO_TANH(x) std::tanh(x)\n"
        "#define ARENO_SELECT_GT(a,b,x,y) ((a) > (b) ? (x) : (y))\n"
        "#define ARENO_SELECT_EQ(a,b,x,y) ((a) == (b) ? (x) : (y))\n"
        '#include "activation_math.h"\n'
        'extern "C" void run(int kind, int n, const float* x, const float* u, const float* g, '
        "float* y, float* dx, float* du) {\n"
        "  for (int i=0; i<n; ++i) {\n"
        "    y[i] = areno_activation_value(kind, x[i], u[i]);\n"
        "    float saved = kind == 1 ? y[i] : x[i];\n"
        "    dx[i] = areno_activation_grad_x(kind, saved, u[i], g[i]);\n"
        "    du[i] = areno_activation_grad_up(kind, x[i], g[i]);\n"
        "  }\n}\n"
    )
    library = build / "math.so"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-I", str(root), str(source), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    native = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    native.run.argtypes = [ctypes.c_int, ctypes.c_int, *([array] * 6)]
    native.run.restype = None
    return native


@pytest.mark.parametrize("kind", range(5), ids=["silu", "sigmoid", "softplus", "silu_mul", "gelu_tanh_mul"])
def test_activation_formulas_and_derivatives(activation_math, kind):
    generator = torch.Generator().manual_seed(17)
    x = torch.cat(
        [
            torch.randn(2048, generator=generator) * 6,
            torch.tensor([-80.0, -30.0, -20.0, -10.0, -1e-6, 0.0, 1e-6, 19.999, 20.0, 20.001, 30.0, 80.0]),
        ]
    )
    up = torch.randn(x.shape, generator=generator)
    grad = torch.randn(x.shape, generator=generator)
    x.requires_grad_()
    up.requires_grad_()
    references = [
        lambda: F.silu(x),
        lambda: torch.sigmoid(x),
        lambda: F.softplus(x),
        lambda: F.silu(x) * up,
        lambda: F.gelu(x, approximate="tanh") * up,
    ]
    expected = references[kind]()
    expected.backward(grad)
    result, dx, du = (np.empty(x.numel(), dtype=np.float32) for _ in range(3))
    activation_math.run(kind, x.numel(), x.detach().numpy(), up.detach().numpy(), grad.numpy(), result, dx, du)
    torch.testing.assert_close(torch.from_numpy(result), expected.detach(), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(torch.from_numpy(dx), x.grad, atol=2e-6, rtol=3e-5)
    if kind in {3, 4}:
        torch.testing.assert_close(torch.from_numpy(du), up.grad, atol=2e-6, rtol=2e-5)
