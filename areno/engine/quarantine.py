"""Quarantine writer for training samples that fail during execution.

Writes a size- and count-limited local JSONL file for reproduction purposes.
Sensitive fields (prompt/completion text) are redacted; only metadata
(lengths, indices, truncated hashes) is retained.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ENTRIES = 200
_DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
_DEFAULT_FAILURE_RATE_THRESHOLD = 0.5  # stop after 50% consecutive failures
_DEFAULT_FAILURE_RATE_WINDOW = 20  # check over last 20 samples


class QuarantineConfig:
    """Configuration for the quarantine writer.

    Attributes:
        enabled: Master switch. When False, QuarantineWriter is a no-op.
        output_dir: Directory for quarantine.jsonl. Defaults to metrics_log_dir.
        max_entries: Maximum number of entries written before the file is frozen.
        max_file_bytes: Maximum file size before the file is frozen.
        failure_rate_threshold: If the failure rate over the last
            ``failure_rate_window`` samples exceeds this fraction, the next
            failure propagates the original exception instead of quarantining.
        failure_rate_window: Sliding window size for failure-rate detection.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        output_dir: str | None = None,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
        failure_rate_threshold: float = _DEFAULT_FAILURE_RATE_THRESHOLD,
        failure_rate_window: int = _DEFAULT_FAILURE_RATE_WINDOW,
    ) -> None:
        self.enabled = enabled
        self.output_dir = output_dir
        self.max_entries = max_entries
        self.max_file_bytes = max_file_bytes
        self.failure_rate_threshold = failure_rate_threshold
        self.failure_rate_window = failure_rate_window


class QuarantineThresholdExceeded(Exception):
    """Failure rate exceeded the configured threshold.

    The ``original_error`` attribute (if set) carries the exception that
    should be re-raised to the caller.
    """

    def __init__(self, message: str, original_error: BaseException | None = None) -> None:
        super().__init__(message)
        self.original_error = original_error


class QuarantineWriter:
    """Thread-safe writer for quarantined training samples.

    Usage::

        writer = QuarantineWriter(config)
        try:
            writer.record(phase="reward", reason=..., sample_meta={...})
        except QuarantineThresholdExceeded as exc:
            raise exc.original_error or exc
    """

    def __init__(self, config: QuarantineConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._entry_count = 0
        self._file_path: Path | None = None
        self._file_handle: Any = None
        self._frozen = False
        self._failure_history: list[bool] = []
        if config.enabled and config.output_dir:
            self._file_path = Path(config.output_dir) / f"quarantine.{os.getpid()}.jsonl"

    @property
    def entry_count(self) -> int:
        """Number of entries written so far."""

        return self._entry_count

    @property
    def frozen(self) -> bool:
        """Whether the writer has been frozen (no more entries accepted)."""

        return self._frozen

    def record(
        self,
        *,
        phase: str,
        reason: str,
        sample_meta: dict[str, Any],
    ) -> None:
        """Record a single failing sample.

        Raises:
            QuarantineThresholdExceeded: If the failure rate over the last
                ``failure_rate_window`` samples exceeds the threshold.
        """

        if not self._config.enabled or self._frozen:
            return

        with self._lock:
            self._track_failure(failed=True)
            self._check_threshold()

            if self._entry_count >= self._config.max_entries:
                self._freeze()
                return

            entry = self._build_entry(phase=phase, reason=reason, sample_meta=sample_meta)
            self._write_entry(entry)

    def record_success(self) -> None:
        """Track a successful sample for failure-rate computation."""

        if not self._config.enabled:
            return
        with self._lock:
            self._track_failure(failed=False)

    def close(self) -> None:
        """Flush and close the quarantine file."""

        with self._lock:
            if self._file_handle is not None:
                self._file_handle.flush()
                self._file_handle.close()
                self._file_handle = None

    # --- Internal helpers ---

    def _track_failure(self, *, failed: bool) -> None:
        self._failure_history.append(failed)
        if len(self._failure_history) > self._config.failure_rate_window:
            self._failure_history.pop(0)

    def _check_threshold(self) -> None:
        if len(self._failure_history) < self._config.failure_rate_window:
            return
        rate = sum(self._failure_history) / len(self._failure_history)
        if rate > self._config.failure_rate_threshold:
            self._frozen = True
            raise QuarantineThresholdExceeded(
                f"failure rate {rate:.0%} over last {len(self._failure_history)} "
                f"samples exceeds threshold {self._config.failure_rate_threshold:.0%}"
            )

    def _build_entry(
        self, *, phase: str, reason: str, sample_meta: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "phase": phase,
            "reason": reason,
            "pid": os.getpid(),
            "step": sample_meta.get("step"),
            "epoch": sample_meta.get("epoch"),
            "prompt_index": sample_meta.get("prompt_index"),
            "sample_index": sample_meta.get("sample_index"),
            "prompt_len": sample_meta.get("prompt_len"),
            "completion_len": sample_meta.get("completion_len"),
            "prompt_hash": _truncated_hash(sample_meta.get("prompt_text")),
            "completion_hash": _truncated_hash(sample_meta.get("completion_text")),
            "timestamp": _now_iso(),
        }

    def _write_entry(self, entry: dict[str, Any]) -> None:
        if self._file_path is None:
            return
        try:
            if self._file_handle is None:
                self._file_path.parent.mkdir(parents=True, exist_ok=True)
                self._file_handle = self._file_path.open("a", encoding="utf-8")
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            self._file_handle.write(line)
            self._file_handle.flush()
            self._entry_count += 1
        except Exception:
            logger.warning("quarantine write failed", exc_info=True)
            return
        if self._file_path.stat().st_size >= self._config.max_file_bytes:
            self._freeze()

    def _freeze(self) -> None:
        self._frozen = True
        logger.warning(
            "quarantine file frozen: entries=%d path=%s",
            self._entry_count,
            self._file_path,
        )


def _truncated_hash(text: str | None) -> str | None:
    """Return a truncated SHA-256 hash of *text*, or ``None`` if *text* is ``None``."""

    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
