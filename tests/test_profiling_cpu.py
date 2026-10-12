import time
import weakref

import pytest
import torch

from areno.engine.profiling import benchmark_callable, profile_callable, profile_region


def test_profile_region_reports_cpu_wall_time() -> None:
    with profile_region("cpu-test") as result:
        time.sleep(0.001)

    assert result.name == "cpu-test"
    assert result.elapsed_ms >= 0.5
    assert result.peak_memory_bytes is None
    assert result.memory_allocated_bytes is None


def test_profile_callable_returns_value_and_measurement() -> None:
    value, result = profile_callable("operation", lambda: 21 * 2)
    assert value == 42
    assert result.name == "operation"
    assert result.elapsed_ms >= 0.0


def test_benchmark_callable_warms_up_and_summarizes_iterations() -> None:
    calls = []

    result = benchmark_callable("repeat", lambda: calls.append(1), warmup=2, iterations=5)

    assert len(calls) == 7
    assert result.name == "repeat"
    assert result.iterations == 5
    assert result.min_ms <= result.median_ms <= result.p95_ms <= result.max_ms
    assert result.peak_memory_bytes is None


def test_benchmark_callable_releases_each_iteration_output() -> None:
    class Output:
        pass

    outputs = []

    def operation():
        assert all(reference() is None for reference in outputs)
        value = Output()
        outputs.append(weakref.ref(value))
        return value

    benchmark_callable("release", operation, warmup=0, iterations=3)

    assert all(reference() is None for reference in outputs)


@pytest.mark.parametrize(("warmup", "iterations"), [(-1, 1), (0, 0)])
def test_benchmark_callable_validates_counts(warmup, iterations) -> None:
    with pytest.raises(ValueError, match="benchmark"):
        benchmark_callable("invalid", lambda: None, warmup=warmup, iterations=iterations)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_profile_region_reports_cuda_memory() -> None:
    device = torch.device("cuda")
    with profile_region("cuda-test", device=device) as result:
        value = torch.ones(1024, 1024, device=device)
        value = value + 1

    assert result.device == "cuda"
    assert result.peak_memory_bytes is not None and result.peak_memory_bytes > 0
    assert result.memory_allocated_bytes is not None and result.memory_allocated_bytes > 0
