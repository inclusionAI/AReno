#!/usr/bin/env python3
"""Capture non-mutating process metadata suitable for runtime triage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    root = Path("/proc") / str(args.pid)
    if not root.exists():
        print(json.dumps({"ok": False, "error": "process not found", "pid": args.pid}))
        return 1
    cmdline = (read(root / "cmdline") or "").replace("\x00", " ").strip()
    status = read(root / "status") or ""
    selected = {}
    for line in status.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"Name", "State", "PPid", "Threads", "VmRSS", "VmSize"}:
            selected[key] = value.strip()
    result = {
        "ok": True,
        "pid": args.pid,
        "cmdline": cmdline,
        "cwd": os.readlink(root / "cwd"),
        "status": selected,
        "py_spy_command": f"py-spy dump -p {args.pid}",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
