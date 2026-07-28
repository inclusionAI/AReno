"""Per-stage dataset preprocessing profiler.

Provides ``StageProfiler`` – a low-overhead context manager that times
individual stages of the data preprocessing pipeline (file read, contract
conversion, tokenization, filtering, batching).  When ``enabled=False`` (the
default) every ``stage()`` call is a pure no-op with zero cost.

The profiler also supports ``inject_delay_s`` for deterministic testing: a test
can inject a known delay into a specific stage and assert that only that
stage's ``total_seconds`` increases, satisfying the issue's acceptance criterion
"inject known delays and confirm correct attribution".

``DataProfileReport`` renders results in two formats: ``to_dict()`` for
machine-readable JSON output via ``MetricsRecorder``, and ``render_human()``
for a human-readable table printed to stdout.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class StageStats:
    """Accumulated timing for one preprocessing stage.

    ``slow_records`` stores bounded identifiers only (cursor index + token
    count) – never prompt text or training content – satisfying the issue's
    privacy requirement.
    """

    name: str
    calls: int = 0
    total_seconds: float = 0.0
    slow_records: list[dict] = field(default_factory=list)
    inject_delay_s: float = 0.0


@dataclass(slots=True)
class DataProfileReport:
    """Aggregated profiling report across all stages."""

    stages: dict[str, StageStats]
    records_scanned: int
    records_skipped_long: int
    wall_seconds: float

    def to_dict(self) -> dict:
        """Return a machine-readable dict suitable for JSON serialisation."""

        return {
            "records_scanned": self.records_scanned,
            "records_skipped_long": self.records_skipped_long,
            "wall_seconds": self.wall_seconds,
            "stages": {
                name: {
                    "calls": s.calls,
                    "total_seconds": s.total_seconds,
                    "slow_records": list(s.slow_records),
                }
                for name, s in self.stages.items()
            },
        }

    def render_human(self) -> str:
        """Return a human-readable multi-line table for stdout."""

        lines = [
            "Data preprocessing profile:",
            f"  records_scanned={self.records_scanned} skipped_long={self.records_skipped_long} wall={self.wall_seconds:.4f}s",
            "  per-stage:",
        ]
        for name in sorted(self.stages):
            s = self.stages[name]
            slow = f" slow={len(s.slow_records)}" if s.slow_records else ""
            lines.append(f"    {name}: calls={s.calls} total={s.total_seconds:.4f}s{slow}")
        return "\n".join(lines)


class StageProfiler:
    """Low-overhead stage timer.

    When ``enabled=False`` (the default) ``stage()`` is a pure no-op: the
    context manager yields immediately without any dict lookup, perf_counter
    call, or closure allocation.
    """

    def __init__(self, enabled: bool, slow_threshold_s: float = 1.0):
        self.enabled = enabled
        self.slow_threshold_s = slow_threshold_s
        self.stages: dict[str, StageStats] = {}

    @contextmanager
    def stage(self, name: str, *, index: int | None = None, tokens: int | None = None,
              inject_delay_s: float | None = None):
        """Time the block inside the ``with`` statement.

        Parameters
        ----------
        name:
            Stage identifier (e.g. ``"record_access"``, ``"tokenize"``).
        index, tokens:
            Bounded identifiers for slow-record attribution.  Only stored when
            elapsed time exceeds ``slow_threshold_s``.
        inject_delay_s:
            Optional delay to inject **inside** the timed block.  Used by tests
            to verify that the delay is attributed to the correct stage.  Never
            exposed via CLI.
        """

        if not self.enabled:
            yield
            return
        stats = self.stages.setdefault(name, StageStats(name=name))
        delay = inject_delay_s if inject_delay_s is not None else stats.inject_delay_s
        start = time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            stats.calls += 1
            stats.total_seconds += elapsed
            if elapsed >= self.slow_threshold_s and index is not None:
                stats.slow_records.append({"index": index, "tokens": tokens})

    def build_report(self, *, records_scanned: int, records_skipped_long: int,
                     wall_seconds: float) -> DataProfileReport:
        """Assemble a ``DataProfileReport`` from accumulated stats."""

        return DataProfileReport(
            stages=dict(self.stages),
            records_scanned=records_scanned,
            records_skipped_long=records_skipped_long,
            wall_seconds=wall_seconds,
        )