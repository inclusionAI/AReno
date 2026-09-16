"""Execute optimizer TPC source through host primitives, independent of HPU ISA."""

import ctypes
import itertools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module", params=list(itertools.product((0, 1), repeat=2)))
def optimizer_source(tmp_path_factory, request):
    model_dtype, grad_dtype = request.param
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Clang is required for the TPC source interpreter")
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("hpu_optimizer")
    text = (
        '#include "hpu_tpc_reference.h"\n'
        f'#define ARENO_DTYPE {model_dtype}\n#define ARENO_GRAD_DTYPE {grad_dtype}\n#include "tensor_io.h"\n'
    )
    for kind in (0, 1):
        text += (
            f"namespace kind_{kind} {{\n#define ARENO_KIND {kind}\n#define main kernel\n"
            '#include "optimizer.c"\n#undef main\n#undef ARENO_KIND\n}\n'
        )
    text += (
        'extern "C" void run(int kind, int n, int offset, float* scalars, void* modelp, void* gradp, '
        "void* lowp, void* carryp, float* mp, float* vp, void* outp, void* lowoutp, void* carryoutp, float* moutp, float* voutp) {\n"
        f"tensor model{{modelp,n,1,{2 if model_dtype else 4}}}, grad{{gradp,n,1,{2 if grad_dtype else 4}}};\n"
        "tensor low{lowp,n,1,2}, carry{carryp,(offset+n+7)/8,1,1}, m{mp,n,1}, v{vp,n,1};\n"
        f"tensor output{{outp,n,1,{2 if model_dtype else 4}}}, lo{{lowoutp,n,1,2}}, co{{carryoutp,(offset+n+7)/8,1,1}};\n"
        "tensor mo{moutp,n,1}, vo{voutp,n,1};\n"
        "index_space_extent = int5{kind ? (offset+n+7)/8 : (n+127)/128,0,0,0,0};\n"
        "if (kind) kind_1::kernel(model,grad,low,carry,m,v,output,lo,co,mo,vo,n,offset,scalars[0],scalars[1],scalars[2],scalars[3],scalars[4],scalars[5],scalars[6]);\n"
        "else kind_0::kernel(model,grad,m,v,output,mo,vo,n,offset,scalars[0],scalars[1],scalars[2],scalars[3],scalars[4],scalars[5],scalars[6]);\n}\n"
    )
    source = build / "optimizer.cpp"
    source.write_text(text)
    library = build / "optimizer.so"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
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
    array = np.ctypeslib.ndpointer(flags="C_CONTIGUOUS")
    native.run.argtypes = [ctypes.c_int] * 3 + [array] * 12
    native.run.restype = None
    return native, model_dtype, grad_dtype


def storage(x, bf16):
    return x.to(torch.bfloat16).view(torch.uint16).numpy().copy() if bf16 else x.numpy().copy()


def floats(x, bf16):
    tensor = torch.from_numpy(x.copy())
    return tensor.view(torch.bfloat16).float() if bf16 else tensor


@pytest.mark.parametrize("count", [1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 257])
def test_fp32_state_updates(optimizer_source, count):
    native, model_dtype, grad_dtype = optimizer_source
    rng = torch.Generator().manual_seed(43)
    model = storage(torch.randn(count, generator=rng), model_dtype)
    grad = storage(torch.randn(count, generator=rng), grad_dtype)
    moment = np.zeros(count, dtype=np.float32)
    variance = np.zeros(count, dtype=np.float32)
    unused_low = np.zeros(count, dtype=np.uint16)
    unused_carry = np.zeros((count + 7) // 8, dtype=np.uint8)
    expected_model = floats(model, model_dtype)
    expected_m = torch.zeros(count)
    expected_v = torch.zeros(count)
    gradient = floats(grad, grad_dtype)
    for step in range(1, 6):
        scalars = np.array(
            [0.9, 0.99, 0.001, 0.1, 1e-8, 0.001 / (1 - 0.9**step), (1 - 0.99**step) ** 0.5], dtype=np.float32
        )
        beta1, beta2, lr, wd, eps, step_size, correction = map(float, scalars)
        expected_m = beta1 * expected_m + (1 - beta1) * gradient
        expected_v = beta2 * expected_v + (1 - beta2) * gradient.square()
        expected_model = expected_model * (1 - lr * wd) - step_size * expected_m / (
            expected_v.sqrt() / correction + eps
        )
        if model_dtype:
            expected_model = expected_model.bfloat16().float()
        output, mo, vo = np.empty_like(model), np.empty_like(moment), np.empty_like(variance)
        native.run(
            0,
            count,
            0,
            scalars,
            model,
            grad,
            unused_low,
            unused_carry,
            moment,
            variance,
            output,
            unused_low,
            unused_carry,
            mo,
            vo,
        )
        torch.testing.assert_close(floats(output, model_dtype), expected_model, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(torch.from_numpy(mo), expected_m, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(torch.from_numpy(vo), expected_v, atol=2e-7, rtol=2e-5)
        model, moment, variance = output, mo, vo


@pytest.mark.parametrize("offset", range(8))
@pytest.mark.parametrize("count", [1, 7, 8, 9, 129])
@pytest.mark.parametrize("optimizer_source", [(1, 0), (1, 1)], indirect=True)
def test_compact_master_preserves_carries_and_fp32_precision(optimizer_source, offset, count):
    native, model_dtype, grad_dtype = optimizer_source
    rng = torch.Generator().manual_seed(91)
    original = torch.randn(count, generator=rng)
    model = storage(original, True)
    words = original.numpy().view(np.uint32)
    low = (words & 0xFFFF).astype(np.uint16)
    carries = np.full((offset + count + 7) // 8, 0xA5, dtype=np.uint8)
    for index in range(count):
        bit_index = offset + index
        mask = 1 << (bit_index % 8)
        rounded = int(model[index]) != int(words[index] >> 16)
        carries[bit_index // 8] = (int(carries[bit_index // 8]) & ~mask) | (mask if rounded else 0)
    grad = storage(torch.randn(count, generator=rng), grad_dtype)
    moment, variance = (np.zeros(count, dtype=np.float32) for _ in range(2))
    scalars = np.array([0.9, 0.99, 0.001, 0.1, 1e-8, 0.01, 0.1], dtype=np.float32)
    output, lo, co, mo, vo = (np.empty_like(x) for x in (model, low, carries, moment, variance))
    native.run(1, count, offset, scalars, model, grad, low, carries, moment, variance, output, lo, co, mo, vo)
    gradient = floats(grad, grad_dtype)
    expected_m = (1 - float(scalars[0])) * gradient
    expected_v = (1 - float(scalars[1])) * gradient.square()
    expected = original * (1 - float(scalars[2]) * float(scalars[3])) - float(scalars[5]) * expected_m / (
        expected_v.sqrt() / float(scalars[6]) + float(scalars[4])
    )
    restored = np.empty(count, dtype=np.uint32)
    for index in range(count):
        bit_index = offset + index
        carry = (int(co[bit_index // 8]) >> (bit_index % 8)) & 1
        restored[index] = (((int(output[index]) - carry) & 0xFFFF) << 16) | int(lo[index])
    torch.testing.assert_close(torch.from_numpy(restored.view(np.float32)), expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(torch.from_numpy(mo), expected_m)
    torch.testing.assert_close(torch.from_numpy(vo), expected_v)
    for index in range(carries.size * 8):
        if not offset <= index < offset + count:
            assert ((int(co[index // 8]) ^ int(carries[index // 8])) & (1 << (index % 8))) == 0
