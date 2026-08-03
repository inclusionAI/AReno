#!/usr/bin/env python3
"""Print core metadata and a safe non-interactive gdb command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("core", type=Path)
    parser.add_argument("--executable", required=True)
    args = parser.parse_args()
    exists = args.core.is_file()
    result = {
        "ok": exists,
        "core": str(args.core),
        "size_bytes": args.core.stat().st_size if exists else None,
        "command": ["gdb", "-batch", "-ex", "thread apply all bt full", args.executable, str(args.core)],
    }
    print(json.dumps(result, indent=2))
    return 0 if exists else 1


if __name__ == "__main__":
    raise SystemExit(main())
