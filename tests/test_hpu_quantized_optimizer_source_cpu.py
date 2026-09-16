"""Interpret the actual TPC source on CPU; this does not compile or emulate HPU ISA."""

import ctypes
import itertools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from areno.engine.optim.adamw_4bit import _SIGNED_DE_MAP
from areno.engine.optim.dynamic_quant import SIGNED_DYNAMIC_MAP, UNSIGNED_DYNAMIC_MAP
from tests.test_hpu_optimizer_source_cpu import floats, storage


@pytest.fixture(scope="module", params=list(itertools.product((0, 1), repeat=2)))
def quantized_source(tmp_path_factory, request):
    model_dtype, grad_dtype = request.param
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Clang is required for the TPC source interpreter")
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("hpu_quantized_optimizer")
    text = (
        '#include "hpu_tpc_reference.h"\n'
        f'#define ARENO_DTYPE {model_dtype}\n#define ARENO_GRAD_DTYPE {grad_dtype}\n#include "tensor_io.h"\n'
    )
    for kind in (3, 4, 8):
        text += (
            f"namespace kind_{kind} {{\n#define ARENO_KIND {kind}\n#define main kernel\n"
            '#include "quantized_optimizer.c"\n#undef main\n#undef ARENO_KIND\n}\n'
        )
    text += (
        'extern "C" void run(int kind, int n, int block, int start, int rows, int cols, float* s, void** p) {\n'
        "int codes = kind == 8 ? n : (n+1)/2, blocks = (n+block-1)/block;\n"
        "auto t = [&](int i, int size, int bytes=4) { return tensor{p[i],size,1,bytes}; };\n"
        f"tensor w=t(0,n,{2 if model_dtype else 4}), g=t(1,n,{2 if grad_dtype else 4}), out=t(11,n,{2 if model_dtype else 4});\n"
        "// Reverse scheduling exercises program ownership independently of launch order.\n"
        "for (int b=blocks-1; b>=0; --b) {\n"
        "index_space_origin=int5{b,0,0,0,0}; index_space_extent=int5{1,0,0,0,0};\n"
        "#define SCALARS n,block,s[0],s[1],s[2],s[3],s[4],s[5],s[6],start,rows,cols\n"
        "#define COMMON w,g,t(2,codes,1),t(3,blocks)\n"
        "#define OUTPUTS out,t(12,codes,1),t(13,blocks)\n"
        "if (kind == 3) kind_3::kernel(COMMON,t(8,rows+cols),t(9,1),t(10,1),OUTPUTS,SCALARS);\n"
        "else if (kind == 4) kind_4::kernel(COMMON,t(4,codes,1),t(5,blocks),OUTPUTS,t(14,codes,1),t(15,blocks),SCALARS);\n"
        "else kind_8::kernel(COMMON,t(4,codes,1),t(5,blocks),t(6,256),t(7,256),OUTPUTS,t(14,codes,1),t(15,blocks),SCALARS);\n"
        "}\n}\n"
    )
    # Stats consumes only the gradient dtype, independent of model dtype.
    text += f"#undef ARENO_DTYPE\n#define ARENO_DTYPE {grad_dtype}\n"
    for kind in (0, 1):
        text += (
            f"namespace stats_{kind} {{\n#define ARENO_KIND {kind}\n#define main kernel\n"
            '#include "factored_stats.c"\n#undef main\n#undef ARENO_KIND\n}\n'
        )
    text += (
        'extern "C" void stats(int n, int start, int rows, int cols, void** p) {\n'
        "auto t = [&](int i, int size, int bytes=4) { return tensor{p[i],size,1,bytes}; };\n"
        "for (int f=rows+cols-1; f>=0; --f) {\n"
        "index_space_origin=int5{f,0,0,0,0}; index_space_extent=int5{1,0,0,0,0};\n"
        f"stats_0::kernel(t(0,n,{2 if grad_dtype else 4}),t(1,rows+cols),t(3,rows+cols),t(2,rows+cols),n,start,rows,cols);\n"
        "}\nindex_space_origin=int5{}; index_space_extent=int5{1,0,0,0,0};\n"
        "stats_1::kernel(t(2,rows+cols),t(4,1),t(5,1),n,start,rows,cols);\n}\n"
    )
    source = build / "quantized.cpp"
    source.write_text(text)
    library = build / "quantized.so"
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
    pointers = ctypes.POINTER(ctypes.c_void_p)
    native.run.argtypes = [ctypes.c_int] * 6 + [
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        pointers,
    ]
    native.stats.argtypes = [ctypes.c_int] * 4 + [pointers]
    native.run.restype = native.stats.restype = None
    return native, model_dtype, grad_dtype


def pointers(arrays):
    assert all(x.flags.c_contiguous for x in arrays)
    return (ctypes.c_void_p * len(arrays))(*(x.ctypes.data for x in arrays))


def unpack(q, count, bits):
    if bits == 8:
        return torch.from_numpy(q.copy()).long()
    codes = np.stack((q & 15, q >> 4), axis=1).reshape(-1)[:count]
    return torch.from_numpy(codes.copy()).long()


def pack(codes, bits, padding):
    codes = codes.numpy().astype(np.uint8)
    if bits == 8:
        return codes
    if codes.size % 2:
        codes = np.append(codes, np.uint8(padding))
    return codes[::2] | (codes[1::2] << 4)


def initial_state(count, block, bits, model_dtype, grad_dtype):
    rng = torch.Generator().manual_seed(76)
    codes = count if bits == 8 else (count + 1) // 2
    blocks = (count + block - 1) // block
    signed_zero = SIGNED_DYNAMIC_MAP.index(0.0) if bits == 8 else 0x77
    return [
        storage(torch.randn(count, generator=rng), model_dtype),
        storage(torch.randn(count, generator=rng), grad_dtype),
        np.full(codes, signed_zero, dtype=np.uint8),
        np.zeros(blocks, dtype=np.float32),
        np.zeros(codes, dtype=np.uint8),
        np.zeros(blocks, dtype=np.float32),
        np.array(SIGNED_DYNAMIC_MAP, dtype=np.float32),
        np.array(UNSIGNED_DYNAMIC_MAP, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        np.zeros(1, dtype=np.int32),
    ]


def reference_step(state, kind, block, scalars, model_dtype, grad_dtype, start=0, rows=1, columns=1):
    bits = 8 if kind == 8 else 4
    model, grad, mq, ms, vq, vs, signed, unsigned, factors, mean, invalid = state
    w, g = floats(model, model_dtype), floats(grad, grad_dtype)
    m_codes, v_codes = unpack(mq, w.numel(), bits), unpack(vq, w.numel(), bits)
    signed_book = torch.tensor(_SIGNED_DE_MAP) if bits == 4 else torch.from_numpy(signed)
    unsigned_book = torch.arange(1, 17).float() / 16 if bits == 4 else torch.from_numpy(unsigned)
    result = [x.copy() for x in (model, mq, ms, vq, vs)]
    b1, b2, lr, decay, eps, step_size, correction = map(float, scalars)
    for b, begin in enumerate(range(0, w.numel(), block)):
        end = min(begin + block, w.numel())
        if kind == 3 and invalid[0] != 0:
            continue
        m = signed_book[m_codes[begin:end]] * float(ms[b])
        gradient = g[begin:end]
        m = b1 * m + float(np.float32(1 - b1)) * gradient
        if kind == 3:
            indices = torch.arange(start + begin, start + end)
            f = torch.from_numpy(factors)
            v = f[indices // columns] * f[rows + indices % columns] / max(float(mean[0]), 1e-30)
        else:
            v = unsigned_book[v_codes[begin:end]] * float(vs[b])
            v = b2 * v + float(np.float32(1 - b2)) * gradient * gradient
        updated = w[begin:end] * float(np.float32(1 - np.float32(lr) * np.float32(decay)))
        updated -= step_size * m / (v.sqrt() / correction + eps)
        if not all(torch.isfinite(x).all() for x in (gradient, m, v, updated)):
            continue
        new_ms, new_vs = m.abs().max(), v.max()
        nm = m / new_ms.clamp_min(1e-30)
        mc = (nm[:, None] - signed_book).abs().argmin(-1)
        result[0][begin:end] = storage(updated, model_dtype)
        qbegin, qend = (begin, end) if bits == 8 else (begin // 2, (end + 1) // 2)
        result[1][qbegin:qend] = pack(mc, bits, 7)
        result[2][b] = float(new_ms)
        if kind != 3:
            nv = v / new_vs.clamp_min(1e-30)
            vc = (
                (nv * 16 - 1).round().clamp(0, 15).long()
                if bits == 4
                else (nv[:, None] - unsigned_book).abs().argmin(-1)
            )
            result[3][qbegin:qend] = pack(vc, bits, 0)
            result[4][b] = float(new_vs)
    return result


def run_step(fixture, state, kind, block, scalars, start=0, rows=1, columns=1):
    native, model_dtype, grad_dtype = fixture
    expected = reference_step(state, kind, block, scalars, model_dtype, grad_dtype, start, rows, columns)
    outputs = [np.full_like(x, 0xA5) for x in (state[0], state[2], state[3], state[4], state[5])]
    original = [x.copy() for x in state]
    native.run(kind, state[0].size, block, start, rows, columns, scalars, pointers(state + outputs))
    for actual, previous in zip(state, original):
        np.testing.assert_array_equal(actual.view(np.uint8), previous.view(np.uint8))
    for i, (actual, ref) in enumerate(zip(outputs, expected)):
        if kind == 3 and i >= 3:
            continue
        if i == 0:
            torch.testing.assert_close(
                floats(actual, model_dtype), floats(ref, model_dtype), rtol=2e-5, atol=2e-6, equal_nan=True
            )
        elif i in (1, 3):
            np.testing.assert_array_equal(actual, ref)
        else:
            np.testing.assert_allclose(actual, ref, rtol=3e-6, atol=2e-7)
    return outputs


@pytest.mark.parametrize("kind,block", [(4, 32), (4, 128), (4, 1024), (8, 1), (8, 7), (8, 256), (8, 4096)])
@pytest.mark.parametrize("count", [0, 1, 31, 32, 33, 129, 257])
def test_quantized_multiple_steps(quantized_source, kind, block, count):
    _, model_dtype, grad_dtype = quantized_source
    state = initial_state(count, block, kind, model_dtype, grad_dtype)
    for step in range(1, 6):
        scalars = np.array(
            [0.9, 0.99, 0.001, 0.1, 1e-8, 0.001 / (1 - 0.9**step), (1 - 0.99**step) ** 0.5], dtype=np.float32
        )
        output = run_step(quantized_source, state, kind, block, scalars)
        for index, new in zip((0, 2, 3, 4, 5), output):
            state[index] = new


@pytest.mark.parametrize("kind", [3, 4, 8])
@pytest.mark.parametrize("cause", ["gradient_nan", "gradient_inf", "gradient_overflow", "model_payload", "scale_nan"])
def test_invalid_block_preserves_storage(quantized_source, kind, cause):
    _, model_dtype, grad_dtype = quantized_source
    state = initial_state(65, 32, 8 if kind == 8 else 4, model_dtype, grad_dtype)
    state[8] = np.ones(18, dtype=np.float32)
    if cause.startswith("gradient"):
        value = {"gradient_nan": float("nan"), "gradient_inf": float("inf"), "gradient_overflow": 1e30}[cause]
        state[1][33:34] = storage(torch.tensor([value]), grad_dtype)
        if kind == 3 and cause == "gradient_overflow":
            state[8][3] = float("inf")
    elif cause == "model_payload":
        state[0].view(np.uint16 if model_dtype else np.uint32)[33] = 0x7FA1 if model_dtype else 0x7FA12345
    else:
        state[3][1] = float("nan")
    scalars = np.array([0.9, 0.99, 0.001, 0.1, 1e-8, 0.01, 0.1], dtype=np.float32)
    out = run_step(quantized_source, state, kind, 32, scalars, rows=9, columns=9)
    slices = [(32, 64), (32, 64) if kind == 8 else (16, 32), (1, 2)]
    for actual, previous, (begin, end) in zip(out[:3], (state[0], state[2], state[3]), slices):
        np.testing.assert_array_equal(actual[begin:end].view(np.uint8), previous[begin:end].view(np.uint8))


@pytest.mark.parametrize("kind", [3, 4, 8])
def test_zero_scale_and_zero_gradients(quantized_source, kind):
    _, model_dtype, grad_dtype = quantized_source
    state = initial_state(65, 32, 8 if kind == 8 else 4, model_dtype, grad_dtype)
    state[1].fill(0)
    state[8] = np.zeros(18, dtype=np.float32)
    state[9].fill(0)
    scalars = np.array([0.9, 0.99, 0.001, 0.0, 1e-8, 0.01, 0.1], dtype=np.float32)
    output = run_step(quantized_source, state, kind, 32, scalars, rows=9, columns=9)
    np.testing.assert_array_equal(output[0], state[0])
    assert not np.any(output[2])


def test_quantization_ties_choose_lower_code(quantized_source):
    _, model_dtype, grad_dtype = quantized_source
    # Binary fractions make exact midpoint ties representable in both dtypes.
    state = initial_state(33, 32, 8, model_dtype, grad_dtype)
    state[6] = np.linspace(-1, 127 / 128, 256, dtype=np.float32)
    state[7] = np.linspace(0, 255 / 256, 256, dtype=np.float32)
    values = torch.tensor([-0.25, -0.125, 0, 0.125, 0.25, 1]).repeat(6)[:33]
    state[1] = storage(values, grad_dtype)
    state[6][128] = -1 / 256
    state[6][129] = 1 / 256
    scalars = np.array([0, 0, 0, 0, 1e-8, 0, 1], dtype=np.float32)
    output = run_step(quantized_source, state, 8, 32, scalars)
    assert output[1][2] == 128


@pytest.mark.parametrize("start,count", [(0, 1), (1, 31), (7, 65), (12, 129), (155, 15)])
@pytest.mark.parametrize("block", [32, 128])
def test_factored_parameter_shards(quantized_source, start, count, block):
    _, model_dtype, grad_dtype = quantized_source
    state = initial_state(count, block, 4, model_dtype, grad_dtype)
    state[8] = np.linspace(0.1, 2, 27, dtype=np.float32)
    state[9][0] = state[8][:10].mean()
    for step in range(1, 4):
        scalars = np.array([0.9, 0.0, 0.001, 0.1, 1e-8, 0.001 / (1 - 0.9**step), 0.7], dtype=np.float32)
        output = run_step(quantized_source, state, 3, block, scalars, start, 10, 17)
        for index, new in zip((0, 2, 3), output):
            state[index] = new


@pytest.mark.parametrize("flag", [1, 2, -1])
def test_factored_invalid_flag_preserves_every_bit(quantized_source, flag):
    _, model_dtype, grad_dtype = quantized_source
    state = initial_state(65, 32, 4, model_dtype, grad_dtype)
    state[8] = np.ones(18, dtype=np.float32)
    state[10][0] = flag
    state[2][-1] = 0xE3  # Preserve the inactive high nibble as well.
    scalars = np.array([0.9, 0.0, 0.001, 0.1, 1e-8, 0.01, 0.1], dtype=np.float32)
    out = run_step(quantized_source, state, 3, 32, scalars, rows=9, columns=9)
    for actual, previous in zip(out, (state[0], state[2], state[3])):
        np.testing.assert_array_equal(actual.view(np.uint8), previous.view(np.uint8))


@pytest.mark.parametrize("shape", [(1, 37), (37, 1), (10, 17)])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), 1e30])
def test_factored_stats_accumulate_shards(quantized_source, shape, bad):
    native, _, grad_dtype = quantized_source
    rows, columns = shape
    original = torch.linspace(-2, 3, rows * columns)
    if bad is not None:
        original[original.numel() // 2] = bad
    grad = storage(original, grad_dtype)
    sums = np.linspace(0, 1, rows + columns, dtype=np.float32)
    invalid = np.zeros(1, dtype=np.int32)
    expected_sums = torch.from_numpy(sums.copy())
    expected_flag = 0
    boundaries = sorted({0, 1, original.numel() // 2, original.numel() - 1, original.numel()})
    for start, end in zip(boundaries, boundaries[1:]):
        piece = grad[start:end].copy()
        output = np.empty_like(sums)
        mask = np.empty(rows + columns, dtype=np.int32)
        updated_flag = np.empty_like(invalid)
        native.stats(piece.size, start, rows, columns, pointers([piece, sums, mask, output, invalid, updated_flag]))
        square = floats(piece, grad_dtype).square()
        indices = torch.arange(start, end)
        valid = torch.isfinite(square)
        expected_flag |= int(not valid.all())
        expected_sums.scatter_add_(0, indices[valid] // columns, square[valid])
        expected_sums.scatter_add_(0, rows + indices[valid] % columns, square[valid])
        np.testing.assert_allclose(output, expected_sums.numpy(), rtol=2e-6, atol=2e-6)
        assert int(updated_flag[0]) == expected_flag
        sums, invalid = output, updated_flag
