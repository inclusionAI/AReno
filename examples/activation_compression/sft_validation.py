"""Fixed-budget SFT with held-out validation loss for activation compression.

Compression only changes what backward sees, so it is accepted by comparing
the final validation loss with the uncompressed BF16 baseline on identical data
and budget:

* rows are loaded through the same dataset-loader path as ``areno train``;
* ``--val-rows`` rows chosen by ``--split-seed`` form a validation split that
  never changes between seeds or arms;
* ``--seed`` only permutes the remaining training rows, so both arms with the
  same seed consume the same ``--steps * --batch-size`` rows;
* validation loss is the token-mean response NLL that the live actor assigns
  through the public ``Trainer.score_logprobs("actor", ...)``.

Each step's training metrics (loss, timing, allocator peaks, ``actc_*``
saved-activation bytes and calibration results) and every validation result are
appended to ``<output>/metrics.jsonl``. ``--save-at`` writes a checkpoint (with
the calibrated plan); ``--resume-from`` plus ``--resume-step`` continues from
it with the same data order.

    python examples/activation_compression/sft_validation.py --ckpt $MODEL \\
        --dataset-path $ALPACA --compression adaptive --seed 1 --output runs/sft/actc_s1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-loader-fn", default="examples/sft/alpaca/dataset_loader.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compression", choices=["none", "adaptive"], default="none")
    parser.add_argument("--error-budget", type=float, default=0.01)
    parser.add_argument("--bits", default="8,4,2", help="Candidate precisions for adaptive compression.")
    parser.add_argument("--check-interval", type=int, default=50, help="Steps between plan error checks.")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--val-rows", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mini-bs", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--score-micro-bs", type=int, default=32)
    parser.add_argument("--save-at", type=int, default=None, help="Save a checkpoint after this step.")
    parser.add_argument("--resume-from", default=None, help="Directory written by --save-at.")
    parser.add_argument("--resume-step", type=int, default=0, help="Steps already trained into --resume-from.")
    return parser.parse_args()


def trainer_config(args: argparse.Namespace):
    from areno.api.trainer_config import TrainerConfig

    return TrainerConfig(
        algo="sft",
        ckpt=args.ckpt,
        dataset_path=args.dataset_path,
        backend="cuda",
        dataset_loader_fn=args.dataset_loader_fn,
        world_size=1,
        tp_size=1,
        batch_size=args.batch_size,
        mini_bs=args.mini_bs,
        score_micro_bs=args.score_micro_bs,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        max_steps=args.steps,
        optimizer_lr=args.lr,
        optimizer_min_lr=args.min_lr,
        lr_decay_steps=args.steps,
        lr_decay_style="cosine",
        weight_decay=args.weight_decay,
        activation_checkpointing=not args.no_activation_checkpointing,
        activation_compression=args.compression,
        activation_compression_error_budget=args.error_budget,
        activation_compression_bits=tuple(int(bits) for bits in args.bits.split(",")),
        activation_compression_check_interval=args.check_interval,
        metrics_log_dir=str(Path(args.output) / "tb"),
    )


def load_rows(config) -> list[dict]:
    from datasets import load_dataset, load_from_disk

    from areno.cli.train import _load_dataset_for_training

    dataset = _load_dataset_for_training(
        config.dataset_path,
        dataset_loader_fn=config.dataset_loader_fn,
        model_hub=config.model_hub,
        load_dataset=load_dataset,
        load_from_disk=load_from_disk,
    )
    return [dict(dataset[index]) for index in range(len(dataset))]


def permutation(size: int, seed: int) -> list[int]:
    import torch

    return torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()


def encoded_rows(rows, indices, tokenizer, args):
    """Yield TrainSequences for ``indices`` in order, skipping over-budget rows like SFTTrainer."""

    from areno.api.trainers.sft import _record_to_train_sequence

    for index in indices:
        sequence = _record_to_train_sequence(
            rows[index],
            tokenizer,
            None,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        if sequence is not None:
            yield sequence


def validation_loss(trainer, rows, micro_bs: int) -> tuple[float, int]:
    """Token-mean NLL of response tokens under the live actor."""

    logprobs = trainer.score_logprobs("actor", [row.tokens for row in rows], microbatch_size=micro_bs)
    total = 0.0
    count = 0
    for row, values in zip(rows, logprobs, strict=True):
        # values[i] is log p(tokens[i] | tokens[:i]); position 0 has no target.
        for position in range(1, len(row.tokens)):
            if not row.prompt_mask[position]:
                total -= float(values[position])
                count += 1
    return total / max(count, 1), count


def run_metadata(args: argparse.Namespace) -> dict:
    import torch
    import transformers

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def git(*command: str) -> str:
        try:
            return subprocess.run(["git", *command], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    model = Path(args.ckpt)
    dataset = Path(args.dataset_path)
    return {
        "args": vars(args),
        "areno_commit": git("rev-parse", "HEAD"),
        "areno_tree": git("rev-parse", "HEAD:areno"),
        "areno_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "model_files_sha256": {path.name: sha256(path) for path in sorted(model.glob("*.safetensors"))},
        "dataset_sha256": sha256(dataset) if dataset.is_file() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": "bfloat16 weights and activations, FP32 master/optimizer math",
    }


def main() -> None:
    args = parse_args()
    import areno.api
    from areno.cli.train import _loss_fn_for_config

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(run_metadata(args), indent=2) + "\n")
    config = trainer_config(args)
    rows = load_rows(config)
    loss_fn = _loss_fn_for_config(config)
    trainer = areno.api.Trainer(
        1,
        args.resume_from or args.ckpt,
        backend_type=config.backend_type(),
        custom_config=config.backend_config(),
        metrics_log_dir=config.metrics_log_dir,
        score_micro_bs=args.score_micro_bs,
    )
    trainer.init()
    log = (output / "metrics.jsonl").open("a")

    def record(kind: str, step: int, values: dict) -> None:
        log.write(json.dumps({"kind": kind, "step": step, "time": time.time(), **values}) + "\n")
        log.flush()

    try:
        tokenizer = trainer.get_tokenizer()
        split = permutation(len(rows), args.split_seed)
        validation = []
        cursor = 0
        while len(validation) < args.val_rows:
            validation.extend(encoded_rows(rows, split[cursor : cursor + 1], tokenizer, args))
            cursor += 1
        pool = split[cursor:]
        order = [pool[index] for index in permutation(len(pool), args.seed)]
        train_rows = encoded_rows(rows, order, tokenizer, args)
        start = args.resume_step
        # Skip the rows a resumed run already consumed.
        for _ in range(start * args.batch_size):
            next(train_rows)
        if args.val_rows and start == 0:
            loss, tokens = validation_loss(trainer, validation, args.score_micro_bs)
            record("eval", 0, {"val_loss": loss, "val_tokens": tokens})
        for step in range(start + 1, args.steps + 1):
            batch = [next(train_rows) for _ in range(args.batch_size)]
            result = trainer.train(batch, loss_fn, mini_bs=args.mini_bs)
            record("train", step, result)
            if args.val_rows and (step % args.eval_interval == 0 or step == args.steps):
                loss, tokens = validation_loss(trainer, validation, args.score_micro_bs)
                record("eval", step, {"val_loss": loss, "val_tokens": tokens})
            if args.save_at == step:
                path = output / f"step_{step:06d}"
                trainer.save_checkpoint(str(path))
                record("save", step, {"path": str(path)})
    finally:
        log.close()
        trainer.close()


if __name__ == "__main__":
    main()
