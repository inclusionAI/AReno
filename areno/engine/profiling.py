"""Small, dependency-free profiling helpers for model hot paths.

The profiler is deliberately usable on CPU-only CI.  CUDA/NPU timing and
memory counters are collected when the profiled device supports them; on CPU
the same context manager still reports wall-clock latency.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from typing import TypeVar

import torch


@dataclass(slots=True)
class ProfileResult:
    """Measurements collected for one profiled region."""

    name: str
    elapsed_ms: float = 0.0
    peak_memory_bytes: int | None = None
    memory_allocated_bytes: int | None = None
    device: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregate measurements from repeated profiled iterations."""

    name: str
    iterations: int
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    peak_memory_bytes: int | None
    device: str | None


def _device_type(device: torch.device | str | None) -> str | None:
    if device is None:
        return None
    return torch.device(device).type


def _synchronize(device: torch.device | str | None) -> None:
    kind = _device_type(device)
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _memory(device: torch.device | str | None, *, peak: bool) -> int | None:
    kind = _device_type(device)
    if kind == "cuda":
        return int(torch.cuda.max_memory_allocated(device) if peak else torch.cuda.memory_allocated(device))
    if kind == "npu" and hasattr(torch, "npu"):
        npu = torch.npu
        getter = getattr(npu, "max_memory_allocated" if peak else "memory_allocated", None)
        return int(getter(device)) if getter is not None else None
    return None


@contextmanager
def profile_region(name: str, device: torch.device | str | None = None) -> Iterator[ProfileResult]:
    """Measure wall time and, when available, accelerator memory usage.

    Synchronization is performed only for CUDA/NPU devices, so profiling CPU
    code remains cheap and deterministic in unit tests.  Memory counters are
    reset at region entry to make the peak value local to the measured block.
    """

    result = ProfileResult(name=name, device=str(torch.device(device)) if device is not None else None)
    kind = _device_type(device)
    if kind == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif kind == "npu" and hasattr(torch, "npu"):
        reset = getattr(torch.npu, "reset_peak_memory_stats", None)
        if reset is not None:
            reset(device)
    _synchronize(device)
    start = time.perf_counter()
    try:
        yield result
    finally:
        _synchronize(device)
        result.elapsed_ms = (time.perf_counter() - start) * 1000.0
        result.peak_memory_bytes = _memory(device, peak=True)
        result.memory_allocated_bytes = _memory(device, peak=False)


T = TypeVar("T")


def profile_callable(
    name: str, fn: Callable[[], T], *, device: torch.device | str | None = None
) -> tuple[T, ProfileResult]:
    """Run ``fn`` and return both its value and a :class:`ProfileResult`."""

    with profile_region(name, device) as result:
        value = fn()
    return value, result


def benchmark_callable(
    name: str,
    fn: Callable[[], object],
    *,
    device: torch.device | str | None = None,
    warmup: int = 2,
    iterations: int = 10,
) -> BenchmarkResult:
    """Warm up and repeatedly profile ``fn``, returning stable summary data."""

    if warmup < 0:
        raise ValueError("benchmark warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("benchmark iterations must be positive")
    for _ in range(warmup):
        fn()
    _synchronize(device)

    samples = []
    memory_samples = []
    for _ in range(iterations):
        value, sample = profile_callable(name, fn, device=device)
        samples.append(sample.elapsed_ms)
        if sample.peak_memory_bytes is not None:
            memory_samples.append(sample.peak_memory_bytes)
        # Do not retain one iteration's potentially large model output while
        # measuring the next iteration's peak memory.
        del value
    samples.sort()
    midpoint = iterations // 2
    median = samples[midpoint] if iterations % 2 else (samples[midpoint - 1] + samples[midpoint]) / 2
    p95 = samples[max(0, ceil(iterations * 0.95) - 1)]
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        median_ms=median,
        p95_ms=p95,
        min_ms=samples[0],
        max_ms=samples[-1],
        peak_memory_bytes=max(memory_samples) if memory_samples else None,
        device=str(torch.device(device)) if device is not None else None,
    )


__all__ = ["BenchmarkResult", "ProfileResult", "benchmark_callable", "profile_callable", "profile_region"]
