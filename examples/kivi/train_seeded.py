"""Run ``areno train`` with a per-step rollout seed and a selected KV-cache dtype.

Experiment harness for the KIVI report. It forwards every argument after
``--`` to ``areno train`` unchanged and only adds three things the CLI does not
expose yet:

* ``--kv-cache-dtype`` / ``--kivi-*``: appended to the engine runtime config.
* ``--seed``: rollout sampling for step ``s`` uses ``seed * 1_000_003 + s``, so
  arms that share a seed draw identical random numbers on identical inputs
  while the dataset order stays fixed across seeds and arms.
* ``--resume-step``: continue from a checkpoint saved after that many steps,
  skipping the prompts those steps consumed and continuing the seed schedule.

The algorithm, loss, advantages and logging are the stock policy trainer.

    python examples/kivi/train_seeded.py --seed 1 --kv-cache-dtype kivi2 -- \
        --algo gspo --ckpt /path/to/Qwen3-0.6B --dataset-path /path/to/gsm8k_main_train ...
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--kv-cache-dtype", choices=["auto", "kivi2", "kivi4"], default="auto")
    parser.add_argument("--kivi-group-size", type=int, default=32)
    parser.add_argument("--kivi-residual-length", type=int, default=128)
    parser.add_argument("--resume-step", type=int, default=0, help="Steps already trained into --ckpt.")
    parser.add_argument("--resume-prompts", type=int, default=0, help="Dataset rows those steps consumed.")
    argv = sys.argv[1:]
    if "--" not in argv:
        parser.error("separate areno train arguments with --")
    split = argv.index("--")
    return parser.parse_args(argv[:split]), argv[split + 1 :]


def main() -> None:
    args, train_args = parse_args()

    import areno.cli.train as train_cli
    from areno.api.trainer_config import RolloutTrainerConfig, TrainerConfig
    from areno.api.trainers.policy_only import PolicyOnlyTrainer

    runtime_overrides = {"kv_cache_dtype": args.kv_cache_dtype}
    if args.kv_cache_dtype != "auto":
        runtime_overrides.update(kivi_group_size=args.kivi_group_size, kivi_residual_length=args.kivi_residual_length)

    def with_kv_cache(cuda_config):
        def cuda_config_with_kv_cache(self):
            config = cuda_config(self)
            config.runtime.update(runtime_overrides)
            return config

        return cuda_config_with_kv_cache

    # Rollout trainers (GSPO, GRPO, PPO) define their own cuda_config, so wrap
    # every definition rather than only the base class one.
    for config_class in (TrainerConfig, RolloutTrainerConfig):
        config_class.cuda_config = with_kv_cache(config_class.cuda_config)

    run_prompt_rollout = PolicyOnlyTrainer._run_prompt_rollout

    async def seeded_prompt_rollout(self, sampling_params, prompt_batch):
        step = args.resume_step + self._dashboard_step
        seeded = sampling_params.model_copy(update={"seed": args.seed * 1_000_003 + step})
        return await run_prompt_rollout(self, seeded, prompt_batch)

    PolicyOnlyTrainer._run_prompt_rollout = seeded_prompt_rollout

    if args.resume_prompts:
        load_dataset_for_training = train_cli._load_dataset_for_training

        def skip_consumed_prompts(*load_args, **load_kwargs):
            dataset = load_dataset_for_training(*load_args, **load_kwargs)
            return dataset.select(range(args.resume_prompts, len(dataset)))

        train_cli._load_dataset_for_training = skip_consumed_prompts

    from areno.cli.main import main as areno_main

    sys.argv = ["areno", "train", *train_args]
    areno_main()


if __name__ == "__main__":
    main()
