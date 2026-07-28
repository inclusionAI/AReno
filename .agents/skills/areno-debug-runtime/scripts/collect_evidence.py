#!/usr/bin/env python3
"""Collect a runtime failure-evidence bundle via areno.cli.debug.

Produces a structured ``FailureBundle`` (JSON + Markdown) in the target
output directory.  This is the primary entry point for the ``areno-debug-runtime``
skill when operators need a self-contained diagnostic snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from areno.cli.debug import FailureBundle, collect_failure_bundle, write_bundle, _render_markdown, _safe_traceback_from_file


def _build_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect AReno failure evidence.")
    parser.add_argument("--output-dir", type=Path, default=Path("./areno-debug"), help="Output directory for the bundle.")
    parser.add_argument("--traceback-file", type=Path, default=None, help="Read traceback from a file (post-mortem).")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU/CUDA collection.")
    parser.add_argument("--no-env", action="store_true", help="Skip environment variable collection.")
    parser.add_argument("--no-redact", dest="redact", action="store_false", help="Disable sensitive-value redaction.")
    parser.add_argument("--json", action="store_true", help="Print the JSON bundle to stdout instead of the Markdown summary.")
    parser.add_argument("command", nargs="*", help="Original command line that triggered this collection.")
    return parser


def main() -> int:
    # Use parse_known_args to tolerate subcommand flags (e.g. --ckpt, --algo)
    # without requiring a -- separator before the command.
    parsed, unknown = _build_args().parse_known_args()
    # Any unknown args get folded into the command positional.
    command = list(parsed.command) + unknown

    error: BaseException | None = None
    if parsed.traceback_file is not None:
        error_str = _safe_traceback_from_file(parsed.traceback_file)
        if error_str is not None:
            error = RuntimeError(error_str)
    else:
        error = None

    bundle = collect_failure_bundle(
        command=list(command) if command else None,
        config=None,
        error=error,
        include_env=not parsed.no_env,
        include_gpu=not parsed.no_gpu,
        redact_env=parsed.redact,
    )

    written_path = write_bundle(bundle, parsed.output_dir)

    if parsed.json:
        print(json.dumps(bundle.to_ordered_dict(), indent=2))
    else:
        print(_render_markdown(bundle))
        print(f"\n---\nBundle written to: {written_path}")

    if bundle.collection_warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())