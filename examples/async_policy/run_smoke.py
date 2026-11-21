"""Single-node multi-GPU smoke run for the experimental async policy trainer.

Rollout runs on one device group and training on another so generation and
optimizer steps overlap. Example on a 5-GPU node (rollout on GPUs 0-1,
training on GPUs 3-4):

    python run_smoke.py --model Qwen/Qwen3-0.6B \
        --rollout-devices 0,1 --train-devices 3,4 \
        --n-samples 4 --max-steps 10

The model is resolved through ModelScope when a repository id is given; a
local checkpoint path is used as-is. See README.md for the knobs and how to
confirm the pipeline is actually overlapping.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MATH_EXAMPLES = REPO_ROOT / "examples" / "math"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="ModelScope model id or local checkpoint path")
    parser.add_argument("--dataset-path", default="gsm8k:main", help="dataset reference: repo/name:config[:split] or a local path")
    parser.add_argument("--model-hub", default="modelscope", choices=["modelscope", "hf"], help="hub used to resolve remote dataset references")
    parser.add_argument("--algo", default="async-grpo", choices=["async-grpo", "async-gspo"])
    parser.add_argument("--rollout-devices", default="0,1", help="logical GPU indices for the rollout engine")
    parser.add_argument("--train-devices", default="3,4", help="logical GPU indices for the train engine")
    parser.add_argument("--tp-size", type=int, default=2, help="train engine tensor-parallel size")
    parser.add_argument("--rollout-tp-size", type=int, default=2, help="rollout engine tensor-parallel size")
    parser.add_argument("--n-samples", type=int, default=4, help="completions sampled per prompt")
    parser.add_argument("--batch-size", type=int, default=4, help="prompt groups per train step")
    parser.add_argument("--mini-bs", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--queue-max-groups", type=int, default=8, help="bounded queue capacity (backpressure)")
    parser.add_argument("--max-staleness", type=int, default=2, help="max policy versions a queued group may lag")
    parser.add_argument("--stale-policy", default="drop", choices=["drop", "error"])
    parser.add_argument("--policy-sync-bucket-mb", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve_model(model: str) -> str:
    path = Path(model)
    if path.exists():
        return str(path)
    from modelscope import snapshot_download

    return snapshot_download(model)


def load_math_dataset(dataset_path: str, model_hub: str):
    from datasets import load_dataset, load_from_disk

    # Mirror the CLI's --dataset-path semantics: local paths, saved datasets,
    # and hub references of the form repo/name:config[:split] resolved through
    # the requested hub (ModelScope by default, per AReno convention).
    from areno.cli.train import _load_dataset_from_path

    loader_path = MATH_EXAMPLES / "dataset_loader.py"
    spec = importlib.util.spec_from_file_location("areno_math_dataset_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    def default_loader(path: str):
        return _load_dataset_from_path(
            path, model_hub=model_hub, load_dataset=load_dataset, load_from_disk=load_from_disk
        )

    return module.load_training_dataset(dataset_path, default_loader=default_loader)


def main() -> None:
    args = parse_args()

    import areno.api
    from areno.api.algorithms import get_algorithm
    from areno.api.rewards import load_reward_fn
    from areno.api.trainer_factory import build_trainer
    from areno.experimental.async_policy.config import AsyncPolicyTrainerConfig

    model_path = resolve_model(args.model)
    train_devices = [int(index) for index in args.train_devices.split(",") if index]
    config = AsyncPolicyTrainerConfig(
        algo=args.algo,
        ckpt=model_path,
        dataset_path=args.dataset_path,
        epochs=1,
        max_steps=args.max_steps,
        tp_size=args.tp_size,
        world_size=len(train_devices),
        train_devices=train_devices,
        rollout_devices=[int(index) for index in args.rollout_devices.split(",") if index],
        rollout_tp_size=args.rollout_tp_size,
        policy_sync_bucket_mb=args.policy_sync_bucket_mb,
        batch_size=args.batch_size,
        mini_bs=args.mini_bs,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        queue_max_groups=args.queue_max_groups,
        max_staleness=args.max_staleness,
        stale_policy=args.stale_policy,
        seed=args.seed,
        # areno-native attention kernels keep the recipe free of the optional
        # flash-attn dependency; swap to "flash" if flash-attn is installed.
        attn_backend="native",
    )
    loss_fn = get_algorithm(config.algo).make_loss_fn(config)
    reward_fn = load_reward_fn(str(MATH_EXAMPLES / "math_verify_reward.py"))
    dataset = load_math_dataset(config.dataset_path, args.model_hub)
    api_trainer = areno.api.Trainer(
        config.world_size,
        config.ckpt,
        backend_type=config.backend_type(),
        metrics_log_dir=config.metrics_log_dir,
        custom_config=config.backend_config(),
        score_micro_bs=config.score_micro_bs,
    )
    trainer = build_trainer(config, instance=api_trainer, dataset=dataset, reward_fn=reward_fn, loss_fn=loss_fn)
    trainer.fit()


if __name__ == "__main__":
    main()
