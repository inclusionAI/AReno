"""Compile and execute the production allocation helper shared by CUDA/NPU."""

import importlib.util
import shutil
from pathlib import Path

import pytest
import torch
from setuptools import Distribution
from torch.utils.cpp_extension import BuildExtension, CppExtension, get_cxx_compiler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    if shutil.which(get_cxx_compiler()) is None:
        pytest.skip("A C++ compiler is required for the common allocation test")
    name = "_areno_moe_permute_common_test"
    build = tmp_path_factory.mktemp("moe_permute_common")
    ext = CppExtension(
        name,
        sources=[str(ROOT / "tests/csrc/moe_permute_common.cpp")],
        include_dirs=[str(ROOT / "areno/accel/csrc")],
        extra_compile_args=["-O0", "-g0"],
    )
    command = BuildExtension(Distribution({"ext_modules": [ext]}), use_ninja=False)
    command.ensure_finalized()
    command.build_temp, command.build_lib = str(build / "temp"), str(build / "lib")
    failed = False
    try:
        command.run()
    except Exception:
        failed = True
    if failed:
        pytest.fail("C++ test extension build failed; see compiler output", pytrace=False)
    spec = importlib.util.spec_from_file_location(name, command.get_ext_fullpath(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden", [0, 17])
@pytest.mark.parametrize("counts", [[], [0], [0, 0, 0], [1, 0, 2, 0, 3], [int(i % 31 == 0) for i in range(257)]])
def test_moe_exact_shapes_counts_and_prefix_offsets(native, dtype, hidden, counts):
    input = torch.empty(5, hidden, dtype=dtype)
    # A non-contiguous count vector must be read by logical index.
    physical = torch.full((2 * len(counts),), -777, dtype=torch.int32)
    physical[::2] = torch.tensor(counts, dtype=torch.int32)
    snapshot = physical.clone()
    out, weight, ids, pos, actual_counts, offsets = native.allocate_topk(input, physical[::2])
    assert out.shape == (sum(counts), hidden) and out.dtype == dtype
    for value, storage in ((weight, torch.float32), (ids, torch.int64), (pos, torch.int32)):
        assert value.shape == (sum(counts),) and value.dtype == storage
    expected = torch.tensor(counts, dtype=torch.int64)
    torch.testing.assert_close(actual_counts, expected)
    torch.testing.assert_close(offsets, torch.cat((torch.zeros(1, dtype=torch.int64), expected.cumsum(0))))
    torch.testing.assert_close(physical, snapshot)


def test_moe_invalid_counts_are_rejected(native):
    input = torch.empty(5, 17)
    with pytest.raises(RuntimeError, match="non-negative"):
        native.allocate_topk(input, torch.tensor([2, -1], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="int32"):
        native.allocate_topk(input, torch.tensor([2, 1], dtype=torch.int64))
    with pytest.raises(RuntimeError, match="matrix"):
        native.allocate_topk(input[0], torch.tensor([2, 1], dtype=torch.int32))
