"""Sample per-device GPU memory, utilization, and temperature history for a run.

Issue #257: AReno needs to track GPU memory and utilization history for a
training run as a focused capability. This module owns the *sampling* half: a
background daemon thread polls ``nvidia-smi`` at a bounded interval, keeps a
bounded in-memory history, and exposes human-readable + structured summaries.
It degrades to a no-op when NVIDIA tooling is unavailable.

The module is deliberately engine-agnostic and never imports ``torch``: the
training hot path (`trainer.fit()`, rollout, loss) is untouched. The CLI
(``areno/cli/train.py``) owns the lifecycle — start before ``fit()``, stop and
flush after. Tests inject a fake ``sample_fn`` so the core logic runs on CPU.

Artifact convention mirrors the existing per-run-per-pid files AReno already
writes into ``metrics_log_dir`` (e.g. ``areno_run_config.{pid}.json``):
``gpu_stats.{pid}.jsonl`` (one line per tick/device) and
``gpu_stats_summary.{pid}.json``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import statistics
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from typing import Callable

# nvidia-smi fields requested for every tick. Order is fixed so the CSV parser
# can read it positionally while still tolerating missing trailing columns.
_NVIDIA_SMI_QUERY = (
    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu"
)
_NVIDIA_SMI_FORMAT = "--format=csv,noheader,nounits"

# Hard ceiling on a single nvidia-smi call so a wedged tool cannot stall a run.
_NVIDIA_SMI_TIMEOUT_S = 10.0


@dataclass(slots=True)
class GPUSample:
    """One per-device reading at one instant."""

    timestamp_s: float
    index: int
    name: str | None
    mem_used_mb: int | None
    mem_total_mb: int | None
    util_pct: int | None
    temp_c: int | None


class GPUSampler:
    """Background sampler with bounded history linked to one run.

    Construction is cheap; ``start()`` launches a daemon thread, ``stop()``
    joins it. Read access (``history``/summaries) is intended after ``stop()``
    so the worker thread is no longer mutating; a lock still guards the deque
    snapshot for safety.
    """

    def __init__(
        self,
        *,
        interval_s: float,
        max_history: int,
        devices: list[int] | None = None,
        sample_fn: Callable[[], list[GPUSample]] | None = None,
        jsonl_path: str | None = None,
    ):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self._interval_s = float(interval_s)
        self._max_history = int(max_history)
        self._devices = set(devices) if devices is not None else None
        # ``sample_fn=None`` selects the real nvidia-smi path; tests pass a fake
        # to exercise parsing/aggregation without a GPU or subprocess.
        self._uses_default_sampler = sample_fn is None
        self._sample_fn = sample_fn or self._default_sample_once
        self._history: deque[GPUSample] = deque(maxlen=self._max_history)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reason: str | None = None
        self._active = False
        # When set, each tick's samples are appended here immediately (durably
        # streamed, not buffered until stop) so a crash mid-run loses at most
        # the in-flight frame. The handle is opened in start() and closed in
        # stop(); writes are flushed per tick.
        self._jsonl_path = jsonl_path
        self._jsonl_handle = None

    @property
    def reason(self) -> str | None:
        """Why sampling produced nothing (e.g. missing nvidia-smi), else None."""

        return self._reason

    @property
    def devices(self) -> list[int]:
        """Sorted device indices actually retained, after any device filtering."""

        with self._lock:
            return sorted({s.index for s in self._history})

    def start(self) -> None:
        """Launch the daemon thread, or mark the sampler inactive with a reason."""

        if self._active:
            return
        # Only the default path depends on nvidia-smi existing; an injected
        # sample_fn (tests, or a future NVML shim) bypasses the probe.
        if self._uses_default_sampler and shutil.which("nvidia-smi") is None:
            self._reason = "nvidia-smi not found"
            return
        if self._jsonl_path is not None:
            # Mirror the append semantics of rollout_samples.{pid}.jsonl: open
            # once, stream per tick, close on stop. Directory must exist (the
            # CLI creates metrics_log_dir before starting the sampler).
            self._jsonl_handle = open(self._jsonl_path, "a", encoding="utf-8")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="areno-gpu-stats", daemon=True)
        self._thread.start()
        self._active = True

    def stop(self) -> None:
        """Signal the worker to stop and join it. Idempotent / finally-safe."""

        if not self._active:
            self._stop.set()
            self._close_jsonl()
            return
        self._stop.set()
        if self._thread is not None:
            # Give the worker one interval to observe the event plus margin.
            self._thread.join(timeout=self._interval_s + 5.0)
        self._close_jsonl()
        self._active = False

    def _close_jsonl(self) -> None:
        """Close the streamed JSONL handle if open. Idempotent."""

        if self._jsonl_handle is not None:
            try:
                self._jsonl_handle.close()
            finally:
                self._jsonl_handle = None

    def is_active(self) -> bool:
        """Whether a sampling thread is currently running."""

        return self._active

    def history(self) -> list[GPUSample]:
        """Return a point-in-time snapshot copy of the bounded history."""

        with self._lock:
            return list(self._history)

    def dump_jsonl(self, path: str) -> int:
        """Write the full history as one JSON line per (tick, device) sample.

        Returns the number of lines written. Append semantics match the other
        per-pid JSONL artifacts under ``metrics_log_dir``.
        """

        lines = self.history()
        with open(path, "a", encoding="utf-8") as handle:
            for sample in lines:
                handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
        return len(lines)

    def write_summary(self, path: str) -> dict:
        """Write a structured per-run summary JSON and return it."""

        payload = self.summary()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return payload

    def summary(self) -> dict:
        """Build the structured per-run summary dict (no I/O)."""

        history = self.history()
        per_device: dict[str, dict] = {}
        for index in sorted({s.index for s in history}):
            rows = [s for s in history if s.index == index]
            mems = [s.mem_used_mb for s in rows if s.mem_used_mb is not None]
            mem_totals = [s.mem_total_mb for s in rows if s.mem_total_mb is not None]
            utils = [s.util_pct for s in rows if s.util_pct is not None]
            temps = [s.temp_c for s in rows if s.temp_c is not None]
            per_device[str(index)] = {
                "peak_mem_used_mb": max(mems) if mems else None,
                "mem_total_mb": mem_totals[0] if mem_totals else None,
                "mean_util_pct": round(statistics.fmean(utils)) if utils else None,
                "max_temp_c": max(temps) if temps else None,
                "n_samples": len(rows),
            }
        timestamps = [s.timestamp_s for s in history]
        duration_s = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0
        return {
            "pid": os.getpid(),
            "interval_s": self._interval_s,
            "max_history": self._max_history,
            "n_samples": len(history),
            "duration_s": round(duration_s, 3),
            "devices": sorted({s.index for s in history}),
            "reason": self._reason,
            "per_device": per_device,
        }

    def summary_text(self) -> str:
        """Return a human-readable summary block for the CLI."""

        summary = self.summary()
        lines = ["AReno GPU stats"]
        if summary["reason"]:
            lines.append(f"  {summary['reason']} — GPU sampling disabled for this run.")
            return "\n".join(lines)
        n_devices = len(summary["devices"])
        if n_devices == 0:
            lines.append("  No GPU samples recorded for this run.")
            return "\n".join(lines)
        lines.append(
            f"  Devices  {n_devices}    Samples  {summary['n_samples']}"
            f"  (interval={summary['interval_s']}s, history_cap={summary['max_history']},"
            f" duration={summary['duration_s']}s)"
        )
        for index in sorted(summary["devices"]):
            row = summary["per_device"][str(index)]
            lines.append(_format_device_line(index, row))
        return "\n".join(lines)

    # --- internals -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                samples = self._sample_fn()
            except Exception:
                # Any sampler failure is a degrade, never a training crash.
                samples = []
            if self._devices is not None:
                samples = [s for s in samples if s.index in self._devices]
            with self._lock:
                self._history.extend(samples)
            if self._jsonl_handle is not None:
                # Stream each tick's frame immediately so an abrupt exit loses
                # at most the in-flight frame, not the whole run's history.
                for sample in samples:
                    self._jsonl_handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
                self._jsonl_handle.flush()
            # Event.wait is interruptible: returns True if set during the wait.
            if self._stop.wait(self._interval_s):
                break

    def _default_sample_once(self) -> list[GPUSample]:
        smi = shutil.which("nvidia-smi")
        if smi is None:
            return []
        try:
            proc = subprocess.run(
                [smi, _NVIDIA_SMI_QUERY, _NVIDIA_SMI_FORMAT],
                check=False,
                text=True,
                capture_output=True,
                timeout=_NVIDIA_SMI_TIMEOUT_S,
            )
        except Exception:
            return []
        if proc.returncode != 0:
            return []
        now = time.perf_counter()
        return [replace(sample, timestamp_s=now) for sample in parse_nvidia_smi_csv(proc.stdout)]


def parse_nvidia_smi_csv(stdout: str) -> list[GPUSample]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` output.

    Positional columns are ``index,name,memory.used,memory.total,
    utilization.gpu,temperature.gpu``. Missing trailing columns (e.g. a board
    with no temperature sensor) yield ``None`` for that field rather than
    dropping the sample. Rows whose index cannot be parsed are skipped.
    Timestamps are zero here; the caller stamps them.
    """

    samples: list[GPUSample] = []
    for row in csv.reader(io.StringIO(stdout)):
        if not row:
            continue
        index = _safe_int(row[0])
        if index is None:
            continue
        samples.append(
            GPUSample(
                timestamp_s=0.0,
                index=index,
                name=row[1].strip() if len(row) > 1 else None,
                mem_used_mb=_safe_int(row[2]) if len(row) > 2 else None,
                mem_total_mb=_safe_int(row[3]) if len(row) > 3 else None,
                util_pct=_safe_int(row[4]) if len(row) > 4 else None,
                temp_c=_safe_int(row[5]) if len(row) > 5 else None,
            )
        )
    return samples


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _format_device_line(index: int, row: dict) -> str:
    def maybe(value, suffix):
        return "?" if value is None else f"{value}{suffix}"

    used = row.get("peak_mem_used_mb")
    total = row.get("mem_total_mb")
    if used is not None and total is not None:
        mem = f"{used}/{total} MB"
    elif used is not None:
        mem = f"{used} MB"
    else:
        mem = "? MB"
    return (
        f"  device {index}  peak_mem {mem}   "
        f"mean_util {maybe(row.get('mean_util_pct'), '%')}   "
        f"max_temp {maybe(row.get('max_temp_c'), 'C')}"
    )
