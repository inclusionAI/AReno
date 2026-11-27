"""Check that the public Trainer restores optimizer state exactly.

Initializes a ``Trainer`` from a checkpoint written by ``save_checkpoint`` plus
``save_optimizer_state``, restores it with ``load_optimizer_state``, saves the
optimizer state again and compares every tensor of the two state files. Any
difference in FP32 masters, moments, step counters or GaLore bases would show
up here, independent of the nondeterminism of later training steps.

    python examples/galore/check_state_roundtrip.py --ckpt runs/e2e/cont/ckpt/step_000004 \\
        --output /tmp/roundtrip --galore-rank 128 --galore-update-proj-gap 4 --galore-scale 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--galore-rank", type=int, default=None)
    parser.add_argument("--galore-update-proj-gap", type=int, default=200)
    parser.add_argument("--galore-scale", type=float, default=1.0)
    args = parser.parse_args()

    from compare_resume import compare_checkpoints

    import areno.api
    from areno.api.trainer_config import TrainerConfig

    config = TrainerConfig(
        algo="sft",
        ckpt=str(args.ckpt),
        dataset_path="unused",
        backend="cuda",
        world_size=1,
        tp_size=1,
        galore_rank=args.galore_rank,
        galore_update_proj_gap=args.galore_update_proj_gap,
        galore_scale=args.galore_scale,
    )
    trainer = areno.api.Trainer(
        1, str(args.ckpt), backend_type=config.backend_type(), custom_config=config.backend_config()
    )
    trainer.init()
    try:
        step = trainer.load_optimizer_state(str(args.ckpt))
        trainer.save_optimizer_state(str(args.output))
    finally:
        trainer.close()
    report = {"restored_step": step, "optimizer_state": compare_checkpoints(args.ckpt, args.output)}
    # Weights are not re-saved; only the optimizer-state files are compared.
    report["optimizer_state"]["only_in_first"] = [
        name for name in report["optimizer_state"]["only_in_first"] if not name.startswith("weights/")
    ]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
