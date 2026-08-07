#!/usr/bin/env python3
"""Capture non-mutating process metadata suitable for runtime triage."""

from __future__ import annotations

import os
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))

from areno_skill_sdk import build_parser, skill_main

# ... (existing code) ...


def read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


@skill_main
def main() -> dict:
    parser = build_parser("Capture non-mutating process metadata suitable for runtime triage.")
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()

    root = Path("/proc") / str(args.pid)
    if not root.exists():
        return {"ok": False, "error": "process not found", "pid": args.pid}
    cmdline = (read(root / "cmdline") or "").replace("\x00", " ").strip()
    status = read(root / "status") or ""
    selected = {}
    for line in status.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"Name", "State", "PPid", "Threads", "VmRSS", "VmSize"}:
            selected[key] = value.strip()
    return {
        "ok": True,
        "pid": args.pid,
        "cmdline": cmdline,
        "cwd": os.readlink(root / "cwd"),
        "status": selected,
        "py_spy_command": f"py-spy dump -p {args.pid}",
    }


if __name__ == "__main__":
    raise SystemExit(main())
