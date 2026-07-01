from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "areno-gsm8k-gspo"
REPO_URL = "https://github.com/inclusionAI/AReno.git"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO_DIR = Path("/workspace/areno")
DEFAULT_CKPT = "Qwen/Qwen3.5-0.8B"
MODAL_BRANCH_ENV = "ARENO_MODAL_BRANCH"


app = modal.App(APP_NAME)

image = modal.Image.from_dockerfile(
    str(PROJECT_ROOT / "Dockerfile"),
    build_args={
        "ARENO_REPO_URL": REPO_URL,
        "ARENO_BRANCH": os.environ.get(MODAL_BRANCH_ENV, "__local__"),
    },
)


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 3,
)
def run_gsm8k_gspo(branch: str, ckpt: str = DEFAULT_CKPT) -> None:
    """Run a short GSM8K GSPO train task on Modal."""

    print(f"Running AReno branch built into image: {branch}", flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    _run(
        [
            "areno",
            "train",
            "--ckpt",
            ckpt,
            "--dataset-path",
            "gsm8k:main",
            "--dataset-loader-fn",
            "examples/math/dataset_loader.py",
            "--reward-fn-path",
            "examples/math/math_verify_reward.py",
            "--algo",
            "gspo",
            "--tp-size",
            "1",
            "--world-size",
            "1",
            "--batch-size",
            "2",
            "--n-samples",
            "8",
            "--mini-bs",
            "1",
            "--max-running-prompts",
            "16",
            "--max-new-tokens",
            "1024",
            "--epochs",
            "1",
            "--drop-rollout-state",
        ],
        cwd=REMOTE_REPO_DIR,
        env=env,
    )


@app.local_entrypoint()
def main(branch: str, ckpt: str = DEFAULT_CKPT) -> None:
    run_gsm8k_gspo.remote(branch=branch, ckpt=ckpt)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch AReno GSM8K GSPO training on Modal.")
    parser.add_argument("--branch", required=True, help="AReno git branch to checkout inside the Modal job.")
    parser.add_argument("--modal-token-id", required=True, help="Modal token ID used by the local Modal client.")
    parser.add_argument(
        "--modal-token-secret", required=True, help="Modal token secret used by the local Modal client."
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help=f"Actor checkpoint or HF repo ID. Default: {DEFAULT_CKPT}")
    return parser.parse_args()


def _launch_with_modal_cli(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env["MODAL_TOKEN_ID"] = args.modal_token_id
    env["MODAL_TOKEN_SECRET"] = args.modal_token_secret
    env["ARENO_MODAL_LAUNCHED"] = "1"
    env[MODAL_BRANCH_ENV] = args.branch

    script = Path(__file__).resolve()
    command = [
        sys.executable,
        "-m",
        "modal",
        "run",
        str(script),
        "--branch",
        args.branch,
        "--ckpt",
        args.ckpt,
    ]
    _run(command, env=env)


if __name__ == "__main__" and os.environ.get("ARENO_MODAL_LAUNCHED") != "1":
    _launch_with_modal_cli(_parse_args())
