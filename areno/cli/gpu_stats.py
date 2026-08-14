"""Bounded per-device GPU telemetry for one CLI training run.

The sampler stays outside the trainer hot path: a daemon thread polls
``nvidia-smi``, maps physical GPUs to the run's logical CUDA device order, and
atomically refreshes a bounded JSONL snapshot beside AReno's other run
artifacts. Sampling and artifact failures are recorded for diagnostics but
never raised into the training loop.
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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

_NVIDIA_SMI_QUERY = "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu"
_NVIDIA_SMI_FORMAT = "--format=csv,noheader,nounits"
_NVIDIA_SMI_TIMEOUT_S = 10.0
_MAX_ERROR_TEXT = 300


@dataclass(frozen=True, slots=True)
class GPUSample:
    """One per-device reading at one wall-clock instant.

    ``index`` is the logical CUDA index used by the run. ``physical_index`` and
    ``uuid`` preserve the nvidia-smi identity so multi-GPU mappings remain
    auditable when ``CUDA_VISIBLE_DEVICES`` reorders devices.
    """

    timestamp_s: float
    index: int
    name: str | None
    mem_used_mb: int | None
    mem_total_mb: int | None
    util_pct: int | None
    temp_c: int | None
    physical_index: int | None = None
    uuid: str | None = None


class GPUSampler:
    """Background sampler with bounded in-memory and on-disk history."""

    def __init__(
        self,
        *,
        interval_s: float,
        max_history: int,
        device_selectors: Sequence[str] | None = None,
        logical_device_indices: Sequence[int] | None = None,
        sample_fn: Callable[[], list[GPUSample]] | None = None,
        jsonl_path: str | None = None,
    ):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self._interval_s = float(interval_s)
        self._max_history = int(max_history)
        self._device_selectors = (
            None if device_selectors is None else tuple(str(item).strip() for item in device_selectors)
        )
        self._logical_device_indices = (
            None if logical_device_indices is None else tuple(int(index) for index in logical_device_indices)
        )
        if self._logical_device_indices is not None:
            if self._device_selectors is None:
                raise ValueError("logical_device_indices requires device_selectors")
            if len(self._logical_device_indices) != len(self._device_selectors):
                raise ValueError("logical_device_indices and device_selectors must have equal length")
        self._uses_default_sampler = sample_fn is None
        self._sample_fn = sample_fn or self._default_sample_once
        self._history: deque[GPUSample] = deque(maxlen=self._max_history)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reason: str | None = None
        self._failure: dict[str, str] | None = None
        self._active = False
        self._jsonl_path = Path(jsonl_path) if jsonl_path is not None else None

    @property
    def reason(self) -> str | None:
        """Why telemetry is unavailable or partial, else ``None``."""

        return self._reason

    @property
    def devices(self) -> list[int]:
        """Sorted logical CUDA indices retained in the bounded history."""

        with self._lock:
            return sorted({sample.index for sample in self._history})

    def start(self) -> None:
        """Launch the daemon thread, or record a clean unavailable state."""

        if self._active:
            return
        if self._uses_default_sampler and shutil.which("nvidia-smi") is None:
            self._record_failure("discovery", "nvidia-smi not found")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="areno-gpu-stats", daemon=True)
        self._thread.start()
        self._active = True

    def stop(self) -> None:
        """Request shutdown without ever blocking longer than one query timeout."""

        self._stop.set()
        thread = self._thread
        if thread is None:
            self._active = False
            return
        thread.join(timeout=_NVIDIA_SMI_TIMEOUT_S + 1.0)
        self._active = thread.is_alive()
        if self._active:
            self._record_failure("shutdown", f"sampler thread did not stop within {_NVIDIA_SMI_TIMEOUT_S + 1.0:.1f}s")

    def is_active(self) -> bool:
        """Whether a sampling thread is currently alive."""

        thread = self._thread
        return bool(self._active and thread is not None and thread.is_alive())

    def history(self) -> list[GPUSample]:
        """Return a point-in-time copy of the bounded history."""

        with self._lock:
            return list(self._history)

    def dump_jsonl(self, path: str) -> int:
        """Atomically replace ``path`` with the bounded history snapshot."""

        samples = self.history()
        _write_jsonl_snapshot(Path(path), samples)
        return len(samples)

    def write_summary(self, path: str) -> dict:
        """Atomically write a structured per-run summary and return it."""

        payload = self.summary()
        _atomic_write_text(Path(path), json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return payload

    def summary(self) -> dict:
        """Build the structured per-run summary without performing I/O."""

        history = self.history()
        per_device: dict[str, dict] = {}
        for index in sorted({sample.index for sample in history}):
            rows = [sample for sample in history if sample.index == index]
            mems = [sample.mem_used_mb for sample in rows if sample.mem_used_mb is not None]
            mem_totals = [sample.mem_total_mb for sample in rows if sample.mem_total_mb is not None]
            utils = [sample.util_pct for sample in rows if sample.util_pct is not None]
            temps = [sample.temp_c for sample in rows if sample.temp_c is not None]
            first = rows[0]
            per_device[str(index)] = {
                "physical_index": first.physical_index,
                "uuid": first.uuid,
                "name": first.name,
                "peak_mem_used_mb": max(mems) if mems else None,
                "mem_total_mb": mem_totals[0] if mem_totals else None,
                "mean_util_pct": round(statistics.fmean(utils)) if utils else None,
                "max_temp_c": max(temps) if temps else None,
                "n_samples": len(rows),
            }
        timestamps = [sample.timestamp_s for sample in history]
        duration_s = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
        return {
            "pid": os.getpid(),
            "interval_s": self._interval_s,
            "max_history": self._max_history,
            "n_samples": len(history),
            "duration_s": round(duration_s, 3),
            "devices": sorted({sample.index for sample in history}),
            "device_selectors": list(self._device_selectors or ()),
            "reason": self._reason,
            "failure": self._failure,
            "per_device": per_device,
        }

    def summary_text(self) -> str:
        """Return a compact human-readable CLI summary."""

        summary = self.summary()
        lines = ["AReno GPU stats"]
        if not summary["devices"]:
            reason = summary["reason"] or "No GPU samples recorded for this run."
            lines.append(f"  {reason}")
            return "\n".join(lines)
        lines.append(
            f"  Devices  {len(summary['devices'])}    Samples  {summary['n_samples']}"
            f"  (interval={summary['interval_s']}s, history_cap={summary['max_history']},"
            f" duration={summary['duration_s']}s)"
        )
        for index in summary["devices"]:
            lines.append(_format_device_line(index, summary["per_device"][str(index)]))
        if summary["reason"]:
            lines.append(f"  WARNING: {summary['reason']}")
        return "\n".join(lines)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                samples = map_visible_devices(
                    self._sample_fn(),
                    self._device_selectors,
                    logical_device_indices=self._logical_device_indices,
                )
                if self._device_selectors is not None and len(samples) < len(self._device_selectors):
                    matched = {str(sample.physical_index) for sample in samples}
                    self._record_failure(
                        "mapping",
                        f"matched {len(samples)}/{len(self._device_selectors)} selectors; "
                        f"selectors={list(self._device_selectors)!r}, physical_indices={sorted(matched)!r}",
                    )
                with self._lock:
                    self._history.extend(samples)
                    snapshot = list(self._history)
            except Exception as exc:
                self._record_failure("sampling", f"{type(exc).__name__}: {exc}")
                snapshot = None
            if self._jsonl_path is not None and snapshot is not None:
                try:
                    _write_jsonl_snapshot(self._jsonl_path, snapshot)
                except Exception as exc:
                    self._record_failure("artifact", f"{type(exc).__name__}: {exc}")
            if self._stop.wait(self._interval_s):
                break

    def _default_sample_once(self) -> list[GPUSample]:
        smi = shutil.which("nvidia-smi")
        if smi is None:
            self._record_failure("discovery", "nvidia-smi not found")
            return []
        try:
            proc = subprocess.run(
                [smi, _NVIDIA_SMI_QUERY, _NVIDIA_SMI_FORMAT],
                check=False,
                text=True,
                capture_output=True,
                timeout=_NVIDIA_SMI_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            self._record_failure("query", f"nvidia-smi timed out after {_NVIDIA_SMI_TIMEOUT_S:.1f}s")
            return []
        except OSError as exc:
            self._record_failure("query", f"{type(exc).__name__}: {exc}")
            return []
        if proc.returncode != 0:
            detail = proc.stderr.strip() or f"exit status {proc.returncode}"
            self._record_failure("query", detail)
            return []
        samples = parse_nvidia_smi_csv(proc.stdout)
        if not samples:
            self._record_failure("parse", "nvidia-smi returned no parseable GPU rows")
            return []
        now = time.time()
        return [replace(sample, timestamp_s=now) for sample in samples]

    def _record_failure(self, stage: str, message: str) -> None:
        detail = " ".join(str(message).split())[:_MAX_ERROR_TEXT]
        self._failure = {"stage": stage, "message": detail}
        self._reason = f"GPU stats {stage} failed: {detail}"


def visible_device_selectors(
    world_size: int,
    environ: dict[str, str] | None = None,
    *,
    logical_device_indices: Sequence[int] | None = None,
) -> list[str]:
    """Resolve the physical device selectors used by this AReno run.

    CUDA accepts physical indices and GPU/MIG UUIDs in ``CUDA_VISIBLE_DEVICES``.
    ``logical_device_indices`` follows the CUDA indices selected by AReno's
    train/rollout topology. When it is omitted, the first ``world_size`` CUDA
    devices are selected for backward compatibility.
    """

    logical_devices = (
        list(range(world_size)) if logical_device_indices is None else [int(index) for index in logical_device_indices]
    )
    if any(index < 0 for index in logical_devices):
        raise ValueError("logical CUDA device indices must be non-negative")
    env = os.environ if environ is None else environ
    configured = env.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        return [str(index) for index in logical_devices]
    selectors = [token.strip() for token in configured.split(",") if token.strip()]
    if not selectors:
        return []
    unavailable = [index for index in logical_devices if index >= len(selectors)]
    if unavailable:
        raise ValueError(
            f"logical CUDA devices {unavailable!r} are outside CUDA_VISIBLE_DEVICES with {len(selectors)} entries"
        )
    return [selectors[index] for index in logical_devices]


def map_visible_devices(
    samples: Sequence[GPUSample],
    selectors: Sequence[str] | None,
    *,
    logical_device_indices: Sequence[int] | None = None,
) -> list[GPUSample]:
    """Filter physical samples and assign the run's logical CUDA indices."""

    if selectors is None:
        if logical_device_indices is not None:
            raise ValueError("logical_device_indices requires selectors")
        return list(samples)
    logical_devices = (
        list(range(len(selectors)))
        if logical_device_indices is None
        else [int(index) for index in logical_device_indices]
    )
    if len(logical_devices) != len(selectors):
        raise ValueError("logical_device_indices and selectors must have equal length")
    mapped: list[GPUSample] = []
    for logical_index, selector in zip(logical_devices, selectors, strict=True):
        match = next((sample for sample in samples if _matches_selector(sample, selector)), None)
        if match is None:
            continue
        physical_index = match.physical_index if match.physical_index is not None else match.index
        mapped.append(replace(match, index=logical_index, physical_index=physical_index))
    return mapped


def parse_nvidia_smi_csv(stdout: str) -> list[GPUSample]:
    """Parse index, UUID, name, memory, utilization, and temperature CSV rows."""

    samples: list[GPUSample] = []
    for row in csv.reader(io.StringIO(stdout)):
        if not row:
            continue
        physical_index = _safe_int(row[0])
        if physical_index is None:
            continue
        samples.append(
            GPUSample(
                timestamp_s=0.0,
                index=physical_index,
                physical_index=physical_index,
                uuid=_optional_text(row, 1),
                name=_optional_text(row, 2),
                mem_used_mb=_safe_int_at(row, 3),
                mem_total_mb=_safe_int_at(row, 4),
                util_pct=_safe_int_at(row, 5),
                temp_c=_safe_int_at(row, 6),
            )
        )
    return samples


def _matches_selector(sample: GPUSample, selector: str) -> bool:
    if selector.isdigit():
        physical_index = sample.physical_index if sample.physical_index is not None else sample.index
        return physical_index == int(selector)
    uuid = sample.uuid or ""
    return bool(uuid) and (uuid == selector or uuid.startswith(selector) or selector.startswith(uuid))


def _optional_text(row: Sequence[str], index: int) -> str | None:
    if len(row) <= index:
        return None
    text = row[index].strip()
    return text or None


def _safe_int_at(row: Sequence[str], index: int) -> int | None:
    return _safe_int(row[index]) if len(row) > index else None


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def _write_jsonl_snapshot(path: Path, samples: Sequence[GPUSample]) -> None:
    text = "".join(json.dumps(asdict(sample), ensure_ascii=False) + "\n" for sample in samples)
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _format_device_line(index: int, row: dict) -> str:
    def maybe(value, suffix):
        return "?" if value is None else f"{value}{suffix}"

    used = row.get("peak_mem_used_mb")
    total = row.get("mem_total_mb")
    if used is not None and total is not None:
        memory = f"{used}/{total} MB"
    elif used is not None:
        memory = f"{used} MB"
    else:
        memory = "? MB"
    physical = row.get("physical_index")
    mapping = f" (physical {physical})" if physical is not None and physical != index else ""
    return (
        f"  device {index}{mapping}  peak_mem {memory}   "
        f"mean_util {maybe(row.get('mean_util_pct'), '%')}   "
        f"max_temp {maybe(row.get('max_temp_c'), 'C')}"
    )
