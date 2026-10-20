"""Run a bounded real-GPU OPD experiment and verify saved student checkpoints.

Use --dry-run to print the command without downloading models or using CUDA.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from check_models import check_model_refs
from verify_run import verify_run

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "examples/opd/prompts.jsonl"


def build_command(args, *, python_executable: str = sys.executable) -> list[str]:
    """Use AReno's real CLI and existing GPU backend, without mock training."""
    return [
        python_executable, "-m", "areno.cli.train",
        "--algo", "opd",
        "--ckpt", args.student,
        "--teacher-ckpt", args.teacher,
        "--dataset-path", str(args.dataset),
        "--model-hub", "hf",
        "--save-path", str(args.output_dir / "checkpoints"),
        "--save-interval", "1",
        "--metrics-log-dir", str(args.output_dir / "metrics"),
        "--epochs", "1", "--max-steps", str(args.steps),
        "--tp-size", str(args.tp_size),
        "--world-size", str(args.world_size),
        "--batch-size", "1", "--n-samples", "1",
        "--mini-bs", "1", "--score-micro-bs", "1",
        "--max-prompt-tokens", "128", "--max-new-tokens", "32",
        "--max-running-prompts", "1",
        "--temperature", "1.0", "--top-k", "-1", "--top-p", "1.0",
        "--lr", "1e-6", "--min-lr", "1e-7",
        "--opd-coef", "1.0", "--attn-backend", args.attn_backend,
        "--disable-thinking",
    ]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/opd-smoke")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--attn-backend", choices=("native", "flash"), default="native")
    parser.add_argument("--skip-model-check", action="store_true",
                        help="Skip tokenizer preflight only if you independently verified compatible token IDs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.world_size <= 0 or args.tp_size <= 0 or args.world_size % args.tp_size:
        raise ValueError("--world-size must be a positive multiple of --tp-size")
    if not args.dataset.is_file():
        raise ValueError(f"prompt dataset not found: {args.dataset}")
    # Verify raw prompts without loading `datasets` or downloading checkpoints.
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < args.steps:
        raise ValueError("dataset must contain at least --steps nonempty prompt rows")
    if any(not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip() for row in rows):
        raise ValueError("every dataset row must contain a nonempty 'prompt' string")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid OPD smoke-run arguments: {exc}") from exc
    cmd = build_command(args)
    print("Command:\n" + shlex.join(cmd), flush=True)
    if args.dry_run:
        print("Dry run only: no model downloads, training, or checkpoints.")
        return 0

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Real-model training requires PyTorch with CUDA support") from exc
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device detected. Run Stage 6 on a supported NVIDIA GPU machine.")
    if torch.cuda.device_count() < args.world_size:
        raise SystemExit("Not enough visible CUDA devices for --world-size")
    if not args.skip_model_check:
        try:
            report = check_model_refs(args.student, args.teacher)
        except (RuntimeError, ValueError, OSError) as exc:
            raise SystemExit(f"Model compatibility check failed: {exc}") from exc
        print("Model compatibility: " + json.dumps(report), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "run.log").exists() or (args.output_dir / "checkpoints/step_000001").exists():
        raise SystemExit("Output directory contains a previous run; choose a fresh --output-dir")
    (args.output_dir / "command.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    manifest = {
        "status": "running", "student": args.student, "teacher": args.teacher,
        "dataset": str(args.dataset.resolve()), "steps": args.steps,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(args.world_size)],
        "command": cmd,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    with (args.output_dir / "run.log").open("w", encoding="utf-8") as logfile:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            logfile.write(line)
        code = process.wait()
    if code:
        manifest["status"] = "failed"
        manifest["exit_code"] = code
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"AReno exited with code {code}; inspect {args.output_dir / 'run.log'}")

    try:
        report = verify_run(args.output_dir, steps=args.steps)
    except (ValueError, OSError) as exc:
        manifest["status"] = "failed_verification"
        manifest["verification_error"] = str(exc)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"Training exited successfully but evidence check failed: {exc}") from exc
    manifest.update(status="verified", verification=report)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\nStage 6 verification PASSED. Saved student checkpoints:")
    for path in report["checkpoint_paths"]:
        print(" ", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
