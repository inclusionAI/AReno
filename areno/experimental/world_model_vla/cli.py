"""Internal command-line interface for the world-model VLA workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from areno.experimental.world_model_vla.assets import create_snapshot_manifest
from areno.experimental.world_model_vla.config import WorldModelVLAConfig
from areno.experimental.world_model_vla.workflow import WorldModelVLAWorkflow


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="Path to workflow JSON config.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage experimental frozen-world-model VLA post-training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--snapshot", type=Path, required=True)
    manifest.add_argument("--sizes-only", action="store_true", help="Record sizes without computing SHA-256.")
    manifest.add_argument("--force", action="store_true", help="Replace an existing snapshot manifest.")

    for name in ("plan", "status"):
        command = subparsers.add_parser(name)
        _add_config_argument(command)

    verify = subparsers.add_parser("verify")
    _add_config_argument(verify)
    verify.add_argument("--sha256", action="store_true", help="Hash files that have manifest checksums.")

    for name in ("submit", "resume"):
        command = subparsers.add_parser(name)
        _add_config_argument(command)
        command.add_argument("--dry-run", action="store_true", help="Print the sbatch command without submitting.")
        command.add_argument("--force", action="store_true", help="Allow resubmitting an already recorded job.")

    run_stage = subparsers.add_parser("run-stage")
    _add_config_argument(run_stage)
    run_stage.add_argument("--end-step", type=int, required=True)
    run_stage.add_argument("--auto-continue", action="store_true")

    run_eval = subparsers.add_parser("run-eval")
    _add_config_argument(run_eval)
    run_eval.add_argument("--checkpoint", type=Path)
    return parser


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _submission_report(job_id: str | None, command: list[str]) -> dict[str, Any]:
    return {"job_id": job_id, "command": command}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            manifest = create_snapshot_manifest(
                args.snapshot,
                include_hashes=not args.sizes_only,
                overwrite=args.force,
            )
            _print_json(
                {
                    "path": str(manifest.path),
                    "files": manifest.files,
                    "total_bytes": manifest.total_bytes,
                    "includes_sha256": manifest.includes_sha256,
                }
            )
            return 0
        config = WorldModelVLAConfig.from_json(args.config)
        workflow = WorldModelVLAWorkflow(config)

        if args.command == "plan":
            _print_json(workflow.plan())
        elif args.command == "verify":
            report = workflow.verify(check_hashes=args.sha256)
            _print_json(report)
            return 0 if report["ok"] else 1
        elif args.command == "submit":
            job_id, command = workflow.submit_initial(dry_run=args.dry_run, force=args.force)
            _print_json(_submission_report(job_id, command))
        elif args.command == "resume":
            job_id, command = workflow.submit_resume(dry_run=args.dry_run, force=args.force)
            _print_json(_submission_report(job_id, command))
        elif args.command == "run-stage":
            current_job_id = os.environ.get("SLURM_JOB_ID")
            if args.auto_continue and current_job_id is None:
                raise RuntimeError("--auto-continue requires SLURM_JOB_ID")
            checkpoint = workflow.run_stage(args.end_step)
            report: dict[str, Any] = {"checkpoint": str(checkpoint)}
            if args.auto_continue:
                successor_id, command = workflow.submit_successor(args.end_step, current_job_id=current_job_id)
                report["successor"] = _submission_report(successor_id, command)
            _print_json(report)
        elif args.command == "run-eval":
            checkpoint = None if args.checkpoint is None else args.checkpoint.expanduser().resolve()
            _print_json(workflow.run_evaluation(checkpoint))
        elif args.command == "status":
            _print_json(workflow.status())
        else:
            parser.error(f"unsupported command: {args.command}")
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


__all__ = ["build_parser", "main"]
