"""Pre-launch GPU occupancy check.

Queries ``nvidia-smi`` for free memory, utilization, and owning compute
processes on each selected GPU.  The results are compared against
user-configurable thresholds to produce human-readable warnings.

This module is intentionally dependency-free (only standard library) so it
can run on CPU-only machines without importing torch.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass(slots=True)
class GpuStatus:
    """Snapshot of one GPU's memory, utilization, and compute processes."""

    index: int
    name: str
    total_mem_mb: int
    free_mem_mb: int
    used_mem_mb: int
    utilization_pct: int
    processes: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class GpuWarning:
    """A single threshold-exceeded warning for one GPU."""

    device_index: int
    kind: str  # "low_memory" or "high_utilization"
    message: str


def query_gpu_status(devices: list[int]) -> list[GpuStatus]:
    """Query nvidia-smi for memory, utilization, and processes on *devices*.

    Returns an empty list when nvidia-smi is unavailable or errors out,
    so callers can safely skip warnings on CPU-only machines.
    """

    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    if not devices:
        return []
    indices = ",".join(str(d) for d in devices)

    gpu_info = _run_smi(
        smi,
        "--query-gpu=memory.total,memory.free,memory.used,utilization.gpu,name",
        indices,
    )
    if gpu_info is None:
        return []

    proc_info = _run_smi(
        smi,
        "--query-compute-apps=pid,process_name,used_memory",
        indices,
    )

    statuses: list[GpuStatus] = []
    for i, row in enumerate(gpu_info):
        device_index = devices[i] if i < len(devices) else i
        total, free, used, util, name = _parse_gpu_row(row)
        statuses.append(
            GpuStatus(
                index=device_index,
                name=name,
                total_mem_mb=total,
                free_mem_mb=free,
                used_mem_mb=used,
                utilization_pct=util,
            )
        )

    # Attach processes to matching GPU (nvidia-smi groups by GPU index in -i).
    proc_map = _parse_compute_apps(proc_info, devices)
    for status in statuses:
        status.processes = proc_map.get(status.index, [])

    return statuses


def check_gpu_occupancy(
    statuses: list[GpuStatus],
    *,
    mem_free_warn_pct: int = 10,
    util_warn_pct: int = 90,
) -> list[GpuWarning]:
    """Compare *statuses* against thresholds and return warnings.

    - ``low_memory``: free memory percentage falls below *mem_free_warn_pct*.
    - ``high_utilization``: GPU utilization exceeds *util_warn_pct*.
    """

    warnings: list[GpuWarning] = []
    for status in statuses:
        if status.total_mem_mb <= 0:
            continue
        free_pct = status.free_mem_mb / status.total_mem_mb * 100
        if free_pct < mem_free_warn_pct:
            procs = _format_processes(status)
            warnings.append(
                GpuWarning(
                    device_index=status.index,
                    kind="low_memory",
                    message=(
                        f"GPU {status.index} ({status.name}) has only "
                        f"{status.free_mem_mb} MB / {status.total_mem_mb} MB free "
                        f"({free_pct:.1f}%). Processes occupying this GPU: {procs}"
                    ),
                )
            )
        if status.utilization_pct > util_warn_pct:
            procs = _format_processes(status)
            warnings.append(
                GpuWarning(
                    device_index=status.index,
                    kind="high_utilization",
                    message=(
                        f"GPU {status.index} ({status.name}) utilization is "
                        f"{status.utilization_pct}% (> {util_warn_pct}% threshold). "
                        f"Processes occupying this GPU: {procs}"
                    ),
                )
            )
    return warnings


def format_gpu_warnings(warnings: list[GpuWarning], statuses: list[GpuStatus]) -> str:
    """Produce a multi-line human-readable summary of all warnings."""

    if not warnings:
        return ""

    lines = ["GPU occupancy warnings:"]
    for w in warnings:
        lines.append(f"  [WARN] {w.message}")

    # Append a compact table of all queried GPUs for context.
    if statuses:
        lines.append("")
        lines.append("  GPU overview:")
        for s in statuses:
            free_pct = s.free_mem_mb / s.total_mem_mb * 100 if s.total_mem_mb > 0 else 0
            lines.append(
                f"    GPU {s.index} ({s.name}): "
                f"{s.free_mem_mb}/{s.total_mem_mb} MB free ({free_pct:.0f}%), "
                f"util {s.utilization_pct}%, "
                f"{len(s.processes)} process(es)"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_smi(smi: str, query: str, indices: str) -> list[list[str]] | None:
    """Run nvidia-smi with *query* for *indices* and return parsed CSV rows."""

    try:
        proc = subprocess.run(
            [smi, f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", indices],
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    reader = csv.reader(io.StringIO(proc.stdout))
    return [row for row in reader if row]


def _parse_gpu_row(row: list[str]) -> tuple[int, int, int, int, str]:
    """Parse one nvidia-smi GPU query row into (total, free, used, util, name)."""

    def _to_int(value: str, default: int = 0) -> int:
        try:
            return int(value.strip())
        except (ValueError, TypeError):
            return default

    total = _to_int(row[0]) if len(row) > 0 else 0
    free = _to_int(row[1]) if len(row) > 1 else 0
    used = _to_int(row[2]) if len(row) > 2 else 0
    util = _to_int(row[3]) if len(row) > 3 else 0
    name = row[4].strip() if len(row) > 4 else "unknown"
    return total, free, used, util, name


def _parse_compute_apps(
    rows: list[list[str]] | None,
    devices: list[int],
) -> dict[int, list[dict]]:
    """Parse compute-apps rows into {device_index: [{pid, name, used_mem_mb}]}.

    nvidia-smi ``--query-compute-apps`` does not include a GPU index column,
    so we distribute processes across the queried devices by their order.
    When all processes share one GPU (single device query), they map there.
    """

    if rows is None:
        return {}
    result: dict[int, list[dict]] = {d: [] for d in devices}
    # When multiple devices are queried, nvidia-smi lists processes per-GPU
    # in order.  A blank line separates groups.  We use a simple positional
    # heuristic: if only one device was queried, all processes belong to it.
    if len(devices) == 1:
        idx = devices[0]
        for row in rows:
            pid, name, used = _parse_proc_row(row)
            result[idx].append({"pid": pid, "name": name, "used_mem_mb": used})
        return result
    # Multi-device: assign to the first device when we cannot disambiguate.
    # A more precise mapping would require `nvidia-smi pmon` or parsing the
    # full XML query, which is out of scope for the warning-only feature.
    for row in rows:
        pid, name, used = _parse_proc_row(row)
        result[devices[0]].append({"pid": pid, "name": name, "used_mem_mb": used})
    return result


def _parse_proc_row(row: list[str]) -> tuple[int, str, int]:
    """Parse one compute-apps row into (pid, process_name, used_mem_mb)."""

    def _to_int(value: str, default: int = 0) -> int:
        try:
            return int(value.strip())
        except (ValueError, TypeError):
            return default

    pid = _to_int(row[0]) if len(row) > 0 else 0
    name = row[1].strip() if len(row) > 1 else "unknown"
    used = _to_int(row[2]) if len(row) > 2 else 0
    return pid, name, used


def _format_processes(status: GpuStatus) -> str:
    """Format the process list for inclusion in warning messages."""

    if not status.processes:
        return "none"
    parts = [
        f"{p['name']}(pid={p['pid']}, {p['used_mem_mb']} MB)"
        for p in status.processes
    ]
    return ", ".join(parts)
