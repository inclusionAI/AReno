"""Train a JevForge-style decision scorer with AReno.

Example (8 GPUs, TP=1 so DP=8):

    python examples/classify/jev/train.py \
        --records /path/to/jevforge/records --ckpt Qwen/Qwen3.5-0.8B \
        --save-path runs/jev_qwen35_08b --world-size 8 --tp-size 1

Checkpoints land in `<save-path>/step_XXXXXX/` as an HF backbone plus
`score_head.safetensors`; convert one with `export_jevforge.py` to evaluate
or serve it with JevForge.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import load_questions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", required=True, help="JevForge split directory or a single records jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--ckpt", required=True, help="local HF directory or ModelScope repo id")
    parser.add_argument("--model-hub", default="modelscope", choices=["modelscope", "hf"])
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--questions-per-step", type=int, default=12)
    parser.add_argument("--microbatch-tokens", type=int, default=24000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--head-warmup-steps", type=int, default=12)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lr-decay-style", default="cosine", choices=["constant", "linear", "cosine"])
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--attn-backend", default="flash", choices=["flash", "native"])
    parser.add_argument("--metrics-log-dir", default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from areno import Trainer
    from areno.api.algorithms import get_algorithm
    from areno.api.trainer_factory import build_trainer
    from areno.cli.model_refs import resolve_model_refs_for_config
    from areno.experimental.classify import ClassifyTrainerConfig

    config = ClassifyTrainerConfig(
        algo="classify",
        ckpt=args.ckpt,
        dataset_path=args.records,
        backend="cuda",
        model_hub=args.model_hub,
        save_path=args.save_path,
        save_interval=args.save_interval,
        epochs=args.epochs,
        max_steps=args.max_steps,
        tp_size=args.tp_size,
        world_size=args.world_size,
        batch_size=args.questions_per_step,
        optimizer_lr=args.backbone_lr,
        optimizer_min_lr=0.0,
        lr_decay_steps=args.max_steps,
        lr_decay_style=args.lr_decay_style,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        activation_checkpointing=not args.no_activation_checkpointing,
        attn_backend=args.attn_backend,
        metrics_log_dir=args.metrics_log_dir,
        brier_weight=args.brier_weight,
        microbatch_tokens=args.microbatch_tokens,
        max_seq_len=args.max_seq_len,
        score_head_lr=args.head_lr,
        score_head_warmup_steps=args.head_warmup_steps,
        seed=args.seed,
    )
    config = resolve_model_refs_for_config(config)
    questions = load_questions(args.records, args.split, label_smoothing=args.label_smoothing)
    logging.info("loaded %d %s questions from %s", len(questions), args.split, args.records)

    instance = Trainer(
        config.world_size,
        config.ckpt,
        backend_type=config.backend_type(),
        custom_config=config.backend_config(),
        metrics_log_dir=config.metrics_log_dir,
    )
    loss_fn = get_algorithm(config.algo).make_loss_fn(config)
    trainer = build_trainer(config, instance=instance, dataset=questions, reward_fn=None, loss_fn=loss_fn)
    trainer.fit()


if __name__ == "__main__":
    main()
