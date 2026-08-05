#!/usr/bin/env python3
"""Build a bounded Nsight Systems command without executing it."""

from __future__ import annotations

import argparse
import json
import shlex


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print(json.dumps({"ok": False, "error": "command is required"}))
        return 1
    built = [
        "nsys",
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--duration",
        str(args.duration),
        "--output",
        args.output,
        *command,
    ]
    print(json.dumps({"ok": True, "argv": built, "shell": shlex.join(built)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
