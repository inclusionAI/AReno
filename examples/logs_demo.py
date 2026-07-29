#!/usr/bin/env python3
"""Minimal deterministic example for ``areno logs`` (issue #253).

This script creates a tiny log file, then demonstrates successful filtering
and an invalid-input error — all without external databases, network
services, or sandboxes.

Usage::

    python examples/logs_demo.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SAMPLE_LOG = """\
2026-07-28 10:00:00,123 INFO areno.engine.training: step=0 loss=2.30 rank=0
2026-07-28 10:00:01,456 WARNING areno.engine.training: lr_clipped rank=0
2026-07-28 10:00:02,789 ERROR areno.engine.inference: OOM at rank=1
2026-07-28 10:00:03,012 INFO areno.engine.rollout: rollout complete rank=1
2026-07-28 10:00:04,345 DEBUG areno.engine.training: gradient norm=0.5 rank=0
"""


def main() -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write(SAMPLE_LOG)
        log_path = f.name

    try:
        print("=== Log file created ===")
        print(f"  path: {log_path}")
        print(f"  lines: {SAMPLE_LOG.count(chr(10))}")
        print()

        # --- Success path: basic read ---
        print("=== areno logs <file> (all lines) ===")
        _run(["areno", "logs", log_path])
        print()

        # --- Success path: severity filter ---
        print("=== areno logs <file> --severity error ===")
        _run(["areno", "logs", log_path, "--severity", "error"])
        print()

        # --- Success path: grep filter ---
        print("=== areno logs <file> --grep loss --tail 1 ===")
        _run(["areno", "logs", log_path, "--grep", "loss", "--tail", "1"])
        print()

        # --- Success path: JSON output ---
        print("=== areno logs <file> --tail 1 --output json ===")
        _run(["areno", "logs", log_path, "--tail", "1", "--output", "json"])
        print()

        # --- Boundary: tail 0 ---
        print("=== areno logs <file> --tail 0 (boundary) ===")
        _run(["areno", "logs", log_path, "--tail", "0"])
        print("  (no output expected)")
        print()

        # --- Invalid input: bad severity ---
        print("=== areno logs <file> --severity trace (invalid) ===")
        result = _run(["areno", "logs", log_path, "--severity", "trace"], check=False)
        if result.returncode != 0:
            print(f"  exit code: {result.returncode} (error expected)")
        print()

        print("=== Demo complete ===")

    finally:
        os.unlink(log_path)


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    # Use "python -m areno.cli.main" so the demo works without installing areno.
    if cmd and cmd[0] == "areno":
        cmd = [sys.executable, "-m", "areno.cli.main"] + cmd[1:]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


if __name__ == "__main__":
    main()