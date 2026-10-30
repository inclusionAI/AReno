"""Run ``areno train`` with seeded rollouts and resumable optimizer state.

Harness for the GaLore reinforcement-learning runs. Every argument after
``--`` is passed to ``areno train`` unchanged, including ``--galore-*``. The
harness only adds what the CLI does not expose:

* ``--seed``: rollout sampling for step ``s`` uses ``seed * 1_000_003 + s``, so
  optimizer arms that share a seed draw identical random numbers on identical
  inputs while the dataset order stays fixed across seeds and arms;
* ``--save-optimizer-state``: every ``--save-interval`` checkpoint also stores
  ``Trainer.save_optimizer_state`` next to the weights;
* ``--resume-step``/``--resume-prompts``: after ``Trainer.init()`` restore the
  optimizer state stored in ``--ckpt`` with ``Trainer.load_optimizer_state``,
  skip the prompts the saved run consumed and continue its seed schedule.
  ``--max-steps`` and checkpoint names count the resumed run's own steps.

The algorithm, loss, advantages and metrics are the stock policy trainer.

    python examples/galore/train_seeded.py --seed 1 --save-optimizer-state -- \\
        --algo gspo --ckpt /models/Qwen3-0.6B --dataset-path /data/gsm8k_main_train \\
        --world-size 1 --tp-size 1 --galore-rank 128 --save-path runs/s1 --save-interval 50 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--save-optimizer-state", action="store_true")
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

    if args.save_optimizer_state:
        maybe_save = PolicyOnlyTrainer._maybe_save

        def save_with_optimizer_state(self, epoch: int, step: int) -> None:
            maybe_save(self, epoch, step)
            if self.config.save_path is not None and (step + 1) % self.config.save_interval == 0:
                # Same directory the stock trainer just wrote; a resumed run
                # numbers it by local step, so give it a fresh --save-path.
                self.areno.save_optimizer_state(str(Path(self.config.save_path) / f"step_{step + 1:06d}"))

        PolicyOnlyTrainer._maybe_save = save_with_optimizer_state

    if args.resume_step:
        fit_initialized = PolicyOnlyTrainer._fit_initialized

        def resumed_fit(self) -> None:
            restored = self.areno.load_optimizer_state(self.config.ckpt)
            if restored != args.resume_step:
                raise ValueError(f"--resume-step {args.resume_step} but {self.config.ckpt} holds step {restored}")
            fit_initialized(self)

        PolicyOnlyTrainer._fit_initialized = resumed_fit

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
