from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import modal

APP_NAME = "areno-train"
REPO_URL = "https://github.com/inclusionAI/AReno.git"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO_DIR = Path("/workspace/areno")
DEFAULT_CKPT = "Qwen/Qwen3.5-0.8B"
DEFAULT_DATASET_PATH = "gsm8k:main"
DEFAULT_DATASET_LOADER_FN = "examples/math/dataset_loader.py"
DEFAULT_REWARD_FN_PATH = "examples/math/math_verify_reward.py"
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


@dataclass
class TrainJobConfig:
    branch: str
    ckpt: str
    algo: str
    dataset_path: str
    dataset_loader_fn: str | None
    reward_fn_path: str | None
    agent_fn: str | None
    tp_size: int
    world_size: int
    batch_size: int
    n_samples: int
    mini_bs: int
    score_micro_bs: int
    max_running_prompts: int | None
    max_prompt_tokens: int | None
    max_new_tokens: int | None
    max_context_len: int | None
    epochs: int
    max_steps: int | None
    extra_train_args: list[str]


def _add_optional(command: list[str], option: str, value: str | int | float | None) -> None:
    if value not in (None, ""):
        command.extend([option, str(value)])


def _build_train_command(config: TrainJobConfig) -> list[str]:
    command = [
        "areno",
        "train",
        "--ckpt",
        config.ckpt,
        "--dataset-path",
        config.dataset_path,
        "--algo",
        config.algo,
        "--tp-size",
        str(config.tp_size),
        "--world-size",
        str(config.world_size),
        "--batch-size",
        str(config.batch_size),
        "--n-samples",
        str(config.n_samples),
        "--mini-bs",
        str(config.mini_bs),
        "--score-micro-bs",
        str(config.score_micro_bs),
        "--epochs",
        str(config.epochs),
    ]
    _add_optional(command, "--dataset-loader-fn", config.dataset_loader_fn)
    _add_optional(command, "--reward-fn-path", config.reward_fn_path)
    _add_optional(command, "--agent-fn", config.agent_fn)
    _add_optional(command, "--max-running-prompts", config.max_running_prompts)
    _add_optional(command, "--max-prompt-tokens", config.max_prompt_tokens)
    _add_optional(command, "--max-new-tokens", config.max_new_tokens)
    _add_optional(command, "--max-context-len", config.max_context_len)
    _add_optional(command, "--max-steps", config.max_steps)
    command.extend(config.extra_train_args)
    return command


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 3,
)
def run_areno_train(config_dict: dict) -> None:
    """Run an AReno train task on Modal."""

    config = TrainJobConfig(**config_dict)
    print(f"Running AReno branch built into image: {config.branch}", flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    _run(_build_train_command(config), cwd=REMOTE_REPO_DIR, env=env)


@app.local_entrypoint()
def main(config_json: str) -> None:
    import json

    run_areno_train.remote(json.loads(config_json))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch an AReno training job on Modal.")
    parser.add_argument("--branch", required=True, help="AReno git branch to checkout inside the Modal job.")
    parser.add_argument("--modal-token-id", required=True, help="Modal token ID used by the local Modal client.")
    parser.add_argument(
        "--modal-token-secret", required=True, help="Modal token secret used by the local Modal client."
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help=f"Actor checkpoint or HF repo ID. Default: {DEFAULT_CKPT}")
    parser.add_argument("--algo", default="gspo", help="AReno training algorithm. Default: gspo")
    parser.add_argument(
        "--dataset-path", default=DEFAULT_DATASET_PATH, help=f"Dataset path. Default: {DEFAULT_DATASET_PATH}"
    )
    parser.add_argument(
        "--dataset-loader-fn",
        default=DEFAULT_DATASET_LOADER_FN,
        help=f"Dataset loader file/function. Default: {DEFAULT_DATASET_LOADER_FN}",
    )
    parser.add_argument(
        "--reward-fn-path",
        default=DEFAULT_REWARD_FN_PATH,
        help=f"Reward function file. Default: {DEFAULT_REWARD_FN_PATH}",
    )
    parser.add_argument("--agent-fn", default=None, help="Optional agentic run_agent.py file/function.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size. Default: 1")
    parser.add_argument("--world-size", type=int, default=1, help="World size. Default: 1")
    parser.add_argument("--batch-size", type=int, default=2, help="Prompt batch size. Default: 2")
    parser.add_argument("--n-samples", type=int, default=8, help="Rollout samples per prompt. Default: 8")
    parser.add_argument("--mini-bs", type=int, default=1, help="Training microbatch size. Default: 1")
    parser.add_argument("--score-micro-bs", type=int, default=8, help="Role scoring microbatch size. Default: 8")
    parser.add_argument("--max-running-prompts", type=int, default=16, help="Concurrent rollout prompts. Default: 16")
    parser.add_argument("--max-prompt-tokens", type=int, default=None, help="Optional max prompt tokens.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max generated tokens. Default: 1024")
    parser.add_argument("--max-context-len", type=int, default=None, help="Optional total context limit.")
    parser.add_argument("--epochs", type=int, default=1, help="Epoch count. Default: 1")
    parser.add_argument("--max-steps", type=int, default=5, help="Optional trainer step cap. Default: 5")
    parser.add_argument(
        "--extra-train-arg",
        action="append",
        default=[],
        help="Additional areno train argument. Repeat for multiple args. Default: --drop-rollout-state",
    )
    parser.add_argument(
        "--extra-train-args",
        default="",
        help='Additional areno train arguments parsed with shell-style splitting, e.g. "--greedy --temperature 0.7".',
    )
    parser.add_argument("--keep-rollout-state", action="store_true", help="Do not add --drop-rollout-state by default.")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> TrainJobConfig:
    return TrainJobConfig(
        branch=args.branch,
        ckpt=args.ckpt,
        algo=args.algo,
        dataset_path=args.dataset_path,
        dataset_loader_fn=args.dataset_loader_fn,
        reward_fn_path=args.reward_fn_path,
        agent_fn=args.agent_fn,
        tp_size=args.tp_size,
        world_size=args.world_size,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        mini_bs=args.mini_bs,
        score_micro_bs=args.score_micro_bs,
        max_running_prompts=args.max_running_prompts,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        max_context_len=args.max_context_len,
        epochs=args.epochs,
        max_steps=args.max_steps,
        extra_train_args=([] if args.keep_rollout_state else ["--drop-rollout-state"])
        + args.extra_train_arg
        + shlex.split(args.extra_train_args),
    )


def _launch_with_modal_cli(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env["MODAL_TOKEN_ID"] = args.modal_token_id
    env["MODAL_TOKEN_SECRET"] = args.modal_token_secret
    env["ARENO_MODAL_LAUNCHED"] = "1"
    env[MODAL_BRANCH_ENV] = args.branch

    import json

    script = Path(__file__).resolve()
    command = [
        sys.executable,
        "-m",
        "modal",
        "run",
        str(script),
        "--config-json",
        json.dumps(asdict(_config_from_args(args))),
    ]
    _run(command, env=env)


if __name__ == "__main__" and os.environ.get("ARENO_MODAL_LAUNCHED") != "1":
    _launch_with_modal_cli(_parse_args())
