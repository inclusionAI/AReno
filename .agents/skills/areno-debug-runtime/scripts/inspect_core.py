#!/usr/bin/env python3
"""Print core metadata and a safe non-interactive gdb command."""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import Result, build_parser, skill_main


@skill_main
def main() -> Result:
    parser = build_parser("Print core metadata and a safe non-interactive gdb command.")
    parser.add_argument("core", type=Path)
    parser.add_argument("--executable", required=True)
    args = parser.parse_args()

    exists = args.core.is_file()
    return Result(
        ok=exists,
        data={
            "core": str(args.core),
            "size_bytes": args.core.stat().st_size if exists else None,
            "command": ["gdb", "-batch", "-ex", "thread apply all bt full", args.executable, str(args.core)],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())