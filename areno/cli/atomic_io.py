"""Atomic file-writing utilities shared across CLI, API, and engine modules.

These helpers consolidate the write-to-temp-then-rename pattern already used
ad-hoc in ``dashboard_registry``, ``metrics``, and ``dashboard.server`` into a
single place with proper exception cleanup: if either the write or the rename
fails, the temporary file is always removed so that no stale ``.tmp`` files
are left behind.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write *content* to *path* atomically.

    The content is first written to ``{path}.tmp`` and then renamed to the
    final destination via :py:meth:`pathlib.Path.replace` (POSIX ``rename``).
    If either the write or the rename raises, the temporary file is cleaned
    up before re-raising the original exception.
    """

    dest = Path(path)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding=encoding)
        tmp.replace(dest)
    except BaseException:
        # ``BaseException`` so that ``KeyboardInterrupt`` also triggers cleanup.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    """Write *content* (bytes) to *path* atomically.

    Same semantics as :func:`atomic_write_text` but for binary data.
    """

    dest = Path(path)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        tmp.write_bytes(content)
        tmp.replace(dest)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: str | Path,
    data: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
) -> None:
    """Serialise *data* as JSON and write to *path* atomically.

    Combines :func:`json.dumps` with :func:`atomic_write_text`.
    """

    text = json.dumps(data, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)
    atomic_write_text(path, text + "\n")