"""Disk space budget estimation and runtime watermark monitoring.

Provides :class:`DiskBudget` for preflight estimation and :class:`DiskMonitor`
for runtime free-space monitoring. The monitor warns once when free space
drops below a configurable threshold and triggers a controlled stop at a
critical threshold, reusing the existing trainer-loop ``return`` exit path.

Thresholds default to percentages of total disk capacity (warn 5%, stop 1%)
with optional absolute-value overrides that take whichever is stricter.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Per-step overhead: TensorBoard scalars (~2 KB) + rollout samples JSONL (~10 KB).
_PER_STEP_BYTES = 12 * 1024


@dataclass(frozen=True)
class DiskBudget:
    """Result of preflight disk space estimation.

    Attributes
    ----------
    paths:
        Filesystem paths checked (save_path, metrics_log_dir, etc.).
    free_bytes:
        Minimum free space across all checked paths' partitions.
    total_bytes:
        Total capacity of the partition with the least free space.
    estimated_usage_bytes:
        Estimated total disk usage for the full training run.
    sufficient:
        True when free space exceeds estimated usage plus the stop threshold.
    detail:
        Human-readable summary.
    next_step:
        Actionable suggestion when insufficient.
    """

    paths: list[str]
    free_bytes: int
    total_bytes: int
    estimated_usage_bytes: int
    sufficient: bool
    detail: str
    next_step: str = ""


@dataclass
class DiskMonitorConfig:
    """Configuration for runtime disk monitoring.

    Thresholds are computed as ``max(percent_threshold, absolute_override)``
    so the stricter value always wins.
    """

    warn_percent: float = 5.0
    stop_percent: float = 1.0
    warn_gb: float | None = None
    stop_gb: float | None = None
    check_interval_steps: int = 10
    include_checkpoints: bool = False


class DiskMonitor:
    """Runtime disk-space watermark monitor.

    Call :meth:`check` every training step; it internally throttles to
    ``check_interval_steps`` and returns ``"ok"``, ``"warn"``, or ``"stop"``.
    Warnings are emitted exactly once per transition into the warn band.
    """

    def __init__(
        self,
        config: DiskMonitorConfig,
        paths: list[str],
        total_steps: int,
        *,
        checkpoint_size_bytes: int = 0,
        save_interval: int = 100,
    ) -> None:
        self._config = config
        self._paths = [str(p) for p in paths if p]
        self._total_steps = total_steps
        self._checkpoint_size_bytes = checkpoint_size_bytes
        self._save_interval = save_interval
        self._already_warned = False
        self._last_check_step = -1
        self.last_free_bytes: int = 0
        self._warn_bytes, self._stop_bytes = self._compute_thresholds()

    def check(self, step: int) -> Literal["ok", "warn", "stop"]:
        """Return the current disk status, throttled by ``check_interval_steps``.

        Only every ``check_interval_steps``-th call actually probes the
        filesystem; intermediate calls return ``"ok"`` without I/O.
        """

        if step - self._last_check_step < self._config.check_interval_steps and step > 0:
            return "ok"
        self._last_check_step = step

        free = _min_free_bytes(self._paths)
        self.last_free_bytes = free

        if free <= self._stop_bytes:
            return "stop"
        if free <= self._warn_bytes:
            if not self._already_warned:
                self._already_warned = True
                logger.warning(
                    "disk_space: free=%.2f GB, warn threshold=%.2f GB",
                    free / 1e9,
                    self._warn_bytes / 1e9,
                )
            return "warn"
        return "ok"

    def estimate_budget(self) -> DiskBudget:
        """Compute a :class:`DiskBudget` for preflight display."""

        estimated = estimate_disk_usage(
            total_steps=self._total_steps,
            save_interval=self._save_interval,
            checkpoint_size_bytes=self._checkpoint_size_bytes,
            include_checkpoints=self._config.include_checkpoints,
        )
        free = _min_free_bytes(self._paths)
        total = _min_total_bytes(self._paths)
        sufficient = free > estimated + self._stop_bytes
        if sufficient:
            detail = (
                f"estimated {estimated / 1e6:.1f} MB usage, "
                f"{free / 1e9:.1f} GB free — sufficient"
            )
            next_step = ""
        else:
            detail = (
                f"estimated {estimated / 1e6:.1f} MB usage, "
                f"{free / 1e9:.1f} GB free — insufficient"
            )
            next_step = (
                "Free up disk space, reduce --epochs/--max-steps, "
                "or move --save-path / --metrics-log-dir to a larger partition."
            )
        return DiskBudget(
            paths=self._paths,
            free_bytes=free,
            total_bytes=total,
            estimated_usage_bytes=estimated,
            sufficient=sufficient,
            detail=detail,
            next_step=next_step,
        )

    def _compute_thresholds(self) -> tuple[int, int]:
        total = _min_total_bytes(self._paths)
        warn = int(total * self._config.warn_percent / 100)
        stop = int(total * self._config.stop_percent / 100)
        if self._config.warn_gb is not None:
            warn = max(warn, int(self._config.warn_gb * 1e9))
        if self._config.stop_gb is not None:
            stop = max(stop, int(self._config.stop_gb * 1e9))
        return warn, stop


def estimate_disk_usage(
    total_steps: int,
    save_interval: int,
    checkpoint_size_bytes: int = 0,
    *,
    include_checkpoints: bool = False,
) -> int:
    """Estimate total disk usage for a training run.

    Per-step overhead covers TensorBoard scalars (~2 KB) and rollout
    samples JSONL (~10 KB). Checkpoint sizes are excluded by default;
    pass ``include_checkpoints=True`` and a non-zero ``checkpoint_size_bytes``
    to include them.
    """

    metrics_total = total_steps * _PER_STEP_BYTES
    if not include_checkpoints or checkpoint_size_bytes <= 0 or save_interval <= 0:
        return metrics_total
    num_saves = max(total_steps // save_interval, 0)
    return metrics_total + num_saves * checkpoint_size_bytes


def build_disk_monitor_from_config(
    trainer_config: Any,
    *,
    disk_monitor_config: DiskMonitorConfig | None,
) -> DiskMonitor | None:
    """Build a :class:`DiskMonitor` from a trainer config, or ``None`` if disabled.

    This helper is called by trainers to decide whether disk monitoring
    is active. When ``disk_monitor_config`` is ``None``, monitoring is off.
    """

    if disk_monitor_config is None:
        return None

    paths: list[str] = []
    save_path = getattr(trainer_config, "save_path", None)
    if save_path:
        paths.append(str(save_path))
    metrics_log_dir = getattr(trainer_config, "metrics_log_dir", None)
    if metrics_log_dir:
        paths.append(str(metrics_log_dir))
    if not paths:
        return None

    # Estimate total steps.
    max_steps = getattr(trainer_config, "max_steps", None)
    epochs = getattr(trainer_config, "epochs", 1)
    if max_steps is not None:
        total_steps = max_steps
    else:
        total_steps = epochs * 1000  # Conservative fallback.

    save_interval = getattr(trainer_config, "save_interval", 100)

    # Checkpoint size: try reading safetensors file size if available.
    ckpt_size = 0
    ckpt_path = getattr(trainer_config, "ckpt", None)
    if ckpt_path and Path(ckpt_path).is_dir():
        for f in Path(ckpt_path).glob("*.safetensors"):
            ckpt_size += f.stat().st_size
        # Also check index for total size.
        index_file = Path(ckpt_path) / "model.safetensors.index.json"
        if index_file.exists() and ckpt_size == 0:
            try:
                index_data = json.loads(index_file.read_text(encoding="utf-8"))
                ckpt_size = int(index_data.get("metadata", {}).get("total_size", 0))
            except (json.JSONDecodeError, OSError):
                pass

    return DiskMonitor(
        config=disk_monitor_config,
        paths=paths,
        total_steps=total_steps,
        checkpoint_size_bytes=ckpt_size,
        save_interval=save_interval,
    )


def format_disk_budget_text(budget: DiskBudget) -> str:
    """Render a :class:`DiskBudget` as human-readable text."""

    status = "OK" if budget.sufficient else "FAIL"
    lines = [
        f"{status:<4} disk budget  {', '.join(budget.paths) if budget.paths else '(no paths)'}",
        f"     {budget.detail}",
    ]
    if not budget.sufficient and budget.next_step:
        lines.append(f"Next:\n  {budget.next_step}")
    return "\n".join(lines)


def disk_budget_to_json(budget: DiskBudget) -> str:
    """Serialise a :class:`DiskBudget` to JSON."""

    return json.dumps(asdict(budget), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _min_free_bytes(paths: list[str]) -> int:
    """Return the minimum free space across all paths' partitions."""

    if not paths:
        return 0
    free_values = []
    for p in paths:
        try:
            usage = shutil.disk_usage(str(p))
            free_values.append(usage.free)
        except OSError:
            free_values.append(0)
    return min(free_values) if free_values else 0


def _min_total_bytes(paths: list[str]) -> int:
    """Return the minimum total capacity across all paths' partitions."""

    if not paths:
        return 0
    total_values = []
    for p in paths:
        try:
            usage = shutil.disk_usage(str(p))
            total_values.append(usage.total)
        except OSError:
            total_values.append(0)
    return min(total_values) if total_values else 0
