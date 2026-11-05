"""Run ``areno train`` with seeded rollouts and a resumable data position.

Harness for the activation-compression reinforcement-learning runs. Every
argument after ``--`` is passed to ``areno train`` unchanged, including
``--activation-compression``. The harness only adds what the CLI does not
expose:

* ``--seed``: rollout sampling for step ``s`` uses ``seed * 1_000_003 + s``, so
  arms that share a seed draw identical random numbers on identical inputs
  while the dataset order stays fixed across seeds and arms;
* ``--resume-step``/``--resume-prompts``: continue a run from a checkpoint
  written by ``--save-interval`` (pass it as ``--ckpt``), skipping the prompts
  the saved run consumed and continuing its seed schedule. Checkpoints hold the
  weights and the calibrated compression plan, which a compressed resume
  reuses; the optimizer restarts, as it does for any ``--ckpt`` start.

    python examples/activation_compression/train_seeded.py --seed 1 -- \\
        --algo gspo --ckpt $MODEL --dataset-path $GSM8K_TRAIN \\
        --world-size 1 --tp-size 1 --activation-compression adaptive ...
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume-step", type=int, default=0, help="Optimizer steps already trained into --ckpt.")
    parser.add_argument("--resume-prompts", type=int, default=0, help="Dataset rows those steps consumed.")
    argv = sys.argv[1:]
    if "--" not in argv:
        parser.error("separate areno train arguments with --")
    split = argv.index("--")
    return parser.parse_args(argv[:split]), argv[split + 1 :]


def main() -> None:
    args, train_args = parse_args()

    import areno.cli.train as train_cli
    from areno.api.trainers.policy_only import PolicyOnlyTrainer

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
