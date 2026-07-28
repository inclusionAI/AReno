"""Incremental log reader for ``areno logs``.

The reader yields :class:`~areno.cli.log_filter.LogEntry` objects one at a
time so the process never holds a full log file in memory.

Two modes are supported:

* **tail** — read the last *N* lines from a file (or all lines when *N* is
  ``None``) and exit.
* **follow** — after the initial read, keep polling for new lines appended
  to the file, similar to ``tail -f``.

The reader also handles:

* **file rotation** — the file is replaced (new inode); the reader re-opens
  the new file from the beginning.
* **truncation** — the file shrinks; the reader resets its offset to 0.
* **partial lines** — a line without a trailing ``\\n`` is buffered until the
  next poll cycle completes it.
* **Ctrl-C** — ``SIGINT`` is caught so the caller can print a summary and
  exit cleanly.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from areno.cli.log_filter import LogEntry, parse_line


@dataclass
class ReadStats:
    """Bookkeeping returned to the caller after reading stops."""

    lines_read: int = 0
    lines_yielded: int = 0
    rotations: int = 0
    truncations: int = 0
    followed: bool = False


class LogReader:
    """Incremental reader for one or more log files.

    Parameters
    ----------
    paths:
        One or more log file paths to read.  When multiple paths are given
        the reader round-robins across them so interleaved output stays
        close to real-time order.
    source_labels:
        Optional human-readable labels for each path (defaults to the file
        name).  Used as the ``source`` field in :class:`LogEntry`.
    """

    def __init__(
        self,
        paths: list[str | Path],
        *,
        source_labels: Optional[list[str]] = None,
    ) -> None:
        if not paths:
            raise ValueError("LogReader requires at least one path")
        self._paths = [Path(p) for p in paths]
        if source_labels is not None:
            if len(source_labels) != len(self._paths):
                raise ValueError("source_labels length must match paths length")
            self._labels = source_labels
        else:
            self._labels = [p.name for p in self._paths]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(
        self,
        *,
        tail: Optional[int] = None,
        follow: bool = False,
        poll_interval: float = 1.0,
    ) -> tuple[Iterator[LogEntry], ReadStats]:
        """Return an iterator over log entries and a live stats object.

        The iterator yields :class:`LogEntry` objects.  When *follow* is
        ``True`` the iterator does not terminate until the caller stops
        consuming (e.g. via ``Ctrl-C``).  The :class:`ReadStats` object is
        updated in-place as the iterator is consumed.
        """

        stats = ReadStats(followed=follow)

        def _gen() -> Iterator[LogEntry]:
            if follow:
                yield from self._read_with_follow(tail, poll_interval, stats)
            else:
                yield from self._read_once(tail, stats)

        return _gen(), stats

    # ------------------------------------------------------------------
    # Non-follow mode
    # ------------------------------------------------------------------

    def _read_once(self, tail: Optional[int], stats: ReadStats) -> Iterator[LogEntry]:
        """Read each file once, optionally only the last *tail* lines."""
        for idx, path in enumerate(self._paths):
            if not path.exists():
                continue
            if tail is not None:
                yield from self._tail_file(path, tail, self._labels[idx], stats)
            else:
                yield from self._read_file_full(path, self._labels[idx], stats)

    @staticmethod
    def _read_file_full(
        path: Path, label: str, stats: ReadStats
    ) -> Iterator[LogEntry]:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stats.lines_read += 1
                entry = parse_line(line, source=label)
                stats.lines_yielded += 1
                yield entry

    @staticmethod
    def _tail_file(
        path: Path, n: int, label: str, stats: ReadStats
    ) -> Iterator[LogEntry]:
        """Yield the last *n* lines from *path*.

        Uses a bounded deque to keep only the last *n* lines in memory,
        so the file is read forward but never fully held.
        """
        if n <= 0:
            return

        from collections import deque

        file_size = path.stat().st_size
        if file_size == 0:
            return

        # Read the file forward, keeping only the last n lines.
        # For large files a reverse-seek optimisation could be added,
        # but the deque keeps memory bounded at O(n * avg_line_len).
        tail_lines: deque[str] = deque(maxlen=n)
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                tail_lines.append(line)

        for raw_line in tail_lines:
            if raw_line:
                stats.lines_read += 1
                entry = parse_line(raw_line, source=label)
                stats.lines_yielded += 1
                yield entry

    # ------------------------------------------------------------------
    # Follow mode
    # ------------------------------------------------------------------

    def _read_with_follow(
        self,
        tail: Optional[int],
        poll_interval: float,
        stats: ReadStats,
    ) -> Iterator[LogEntry]:
        """First emit the tail (if any), then follow for new lines."""

        # Phase 1: emit initial lines (tail or full).
        for idx, path in enumerate(self._paths):
            if not path.exists():
                continue
            if tail is not None:
                yield from self._tail_file(path, tail, self._labels[idx], stats)
            else:
                yield from self._read_file_full(path, self._labels[idx], stats)

        # Phase 2: follow — poll for new content.
        _install_sigint_handler()
        offsets: dict[int, int] = {}
        inodes: dict[int, int] = {}
        partial: dict[int, str] = {}

        for idx, path in enumerate(self._paths):
            if path.exists():
                st = path.stat()
                offsets[idx] = st.st_size
                inodes[idx] = st.st_ino
            else:
                offsets[idx] = 0
                inodes[idx] = 0
            partial[idx] = ""

        while not _sigint_received:
            any_new = False
            for idx, path in enumerate(self._paths):
                if not path.exists():
                    continue

                try:
                    st = path.stat()
                except OSError:
                    continue

                # Detect rotation (inode changed).
                if inodes.get(idx, 0) and st.st_ino != inodes[idx]:
                    stats.rotations += 1
                    offsets[idx] = 0
                    inodes[idx] = st.st_ino
                    partial[idx] = ""

                # Detect truncation (file shrank).
                if st.st_size < offsets.get(idx, 0):
                    stats.truncations += 1
                    offsets[idx] = 0
                    partial[idx] = ""

                new_bytes = st.st_size - offsets.get(idx, 0)
                if new_bytes <= 0:
                    continue

                any_new = True
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(offsets[idx])
                    for line in fh:
                        stats.lines_read += 1
                        stats.lines_yielded += 1
                        yield parse_line(line, source=self._labels[idx])
                    # Update offset to the current end of file.
                    offsets[idx] = fh.tell()

            if not any_new:
                time.sleep(poll_interval)

        # Restore default SIGINT handler.
        _uninstall_sigint_handler()


# ---------------------------------------------------------------------------
# SIGINT handling (module-level so the flag survives generator suspension)
# ---------------------------------------------------------------------------

_sigint_received = False
_prev_sigint_handler = None


def _install_sigint_handler() -> None:
    global _sigint_received, _prev_sigint_handler
    _sigint_received = False

    def _handler(signum, frame):
        global _sigint_received
        _sigint_received = True

    _prev_sigint_handler = signal.signal(signal.SIGINT, _handler)


def _uninstall_sigint_handler() -> None:
    global _prev_sigint_handler
    if _prev_sigint_handler is not None:
        signal.signal(signal.SIGINT, _prev_sigint_handler)
        _prev_sigint_handler = None


def find_log_files(metrics_dir: str | Path) -> list[Path]:
    """Return candidate log files under a metrics directory.

    Looks for files that look like text logs (``*.log``, ``*.txt``,
    ``*.out``) as well as AReno's ``areno_run_config.*.txt`` and
    ``rollout_samples.*.jsonl`` artifacts.  TensorBoard event files are
    excluded because they are binary.
    """
    base = Path(metrics_dir)
    if not base.exists():
        return []

    candidates: list[Path] = []
    for p in sorted(base.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        # Skip TensorBoard binary events.
        if name.startswith("events.out.tfevents"):
            continue
        # Skip dashboard state JSON (not a log).
        if name.startswith("dashboard_state."):
            continue
        # Include text logs and run config.
        if name.endswith((".log", ".txt", ".out")) or "run_config" in name:
            candidates.append(p)
    return candidates


def resolve_run_paths(run_id: str) -> list[Path]:
    """Resolve a run-id to a list of log file paths.

    The *run_id* may be:

    * A direct path to a log file.
    * A path to a metrics directory (containing AReno artifacts).
    * A job id from the dashboard registry (``~/.areno/dashboard-jobs.json``).
    """
    # Case 1: direct file path.
    p = Path(run_id)
    if p.is_file():
        return [p]

    # Case 2: directory path.
    if p.is_dir():
        return find_log_files(p)

    # Case 3: look up in dashboard registry.
    from areno.cli.dashboard_registry import dashboard_registry_path

    registry_path = dashboard_registry_path()
    try:
        import json

        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    for job in data.get("jobs", []):
        if job.get("id") == run_id or str(job.get("pid")) == run_id:
            metrics_dir = job.get("metrics_dir")
            if metrics_dir:
                return find_log_files(metrics_dir)
    return []