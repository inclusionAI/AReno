#!/usr/bin/env python3
"""Build a bounded Nsight Systems command without executing it."""

from __future__ import annotations

import argparse
import pathlib
import shlex
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import SkillError, build_parser, skill_main


@skill_main
def main() -> dict:
    parser = build_parser("Build a bounded Nsight Systems command without executing it.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SkillError("command is required", stage="validate")
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
    return {"ok": True, "argv": built, "shell": shlex.join(built)}


if __name__ == "__main__":
    raise SystemExit(main())