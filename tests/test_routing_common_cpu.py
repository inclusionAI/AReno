"""Execute the exact C++ selection helper used by CUDA and Ascend."""

import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def select(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the shared routing helper")
    executable = tmp_path_factory.mktemp("routing_common") / "select"
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(ROOT / "areno/accel/csrc"),
            str(ROOT / "tests/csrc/routing_common.cpp"),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    def run(values, k, initial_index=0):
        result = subprocess.run(
            [str(executable), str(k), str(initial_index)],
            input=" ".join(map(str, values)),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return list(map(int, result.stdout.split()))

    return run


@pytest.mark.parametrize("experts", [1, 7, 16, 17, 65, 255, 256, 512])
@pytest.mark.parametrize("k", [1, 4, 16])
def test_routing_selection_matches_stable_descending_sort(select, experts, k):
    if k > experts:
        pytest.skip("top_k exceeds the expert count")
    rng = random.Random(37)
    values = [rng.randrange(-8, 9) / 8 for _ in range(experts)]
    expected = sorted(range(experts), key=lambda i: (-values[i], i))[:k]
    assert select(values, k) == expected
    assert select(values, k, initial_index=experts) == expected


def test_routing_selection_ties_signed_zero_and_masked_scores(select):
    values = [0.0, -0.0, -math.inf, 1.0, 1.0, math.nan, math.inf, 0.0]
    assert select(values, 6) == [6, 3, 4, 0, 1, 7]
    assert select([0.0] * 512, 16) == list(range(16))
    # Preserve both initial-index conventions used in the CUDA kernels.
    assert select([-math.inf] * 4, 4) == [0] * 4
    assert select([-math.inf] * 4, 4, initial_index=4) == list(range(4))
