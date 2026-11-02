"""Bounded public-Trainer trial; short local text is not a quality benchmark.

Run each optimizer/seed in a fresh process. Model paths and output paths are
explicit so this script can also run against an isolated baseline checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import torch

from areno import Trainer
from areno.api import CudaConfig, TrainSequence, gspo_loss_fn, sft_loss_fn
from areno.engine.config import RuntimeConfig
from areno.engine.training import TrainingManager

if "ARENO_BENCH_SEED" in os.environ:
    torch.manual_seed(int(os.environ["ARENO_BENCH_SEED"]))
    random.seed(int(os.environ["ARENO_BENCH_SEED"]))

# Old baseline commits lack the optional native profiler. Apply equivalent
# instrumentation in the benchmark harness, including in spawned workers.
if "profile_training" not in RuntimeConfig.__dataclass_fields__:
    _original_train_step = TrainingManager._train_step

    def _profiled_train_step(self, *args, **kwargs):
        torch.cuda.synchronize(self.worker.device)
        torch.cuda.reset_peak_memory_stats(self.worker.device)
        started = time.perf_counter()
        result = _original_train_step(self, *args, **kwargs)
        torch.cuda.synchronize(self.worker.device)
        if result is not None:
            result["metrics"].update(
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.worker.device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.worker.device),
                cuda_step_time_s=time.perf_counter() - started,
            )
        return result

    TrainingManager._train_step = _profiled_train_step


def examples(split: str, dataset: str) -> list[dict]:
    if dataset == "repo-text":
        repo = Path(__file__).resolve().parents[1]
        corpus = "\n\n".join((repo / name).read_text() for name in ("README.md", "CONTRIBUTING.md", "AGENTS.md"))
        chunks = [corpus[start : start + 256] for start in range(0, len(corpus) - 255, 256)]
        assert len(chunks) >= 80
        return [{"text": text} for text in (chunks[:64] if split == "train" else chunks[-16:])]
    offset = 1 if split == "train" else 33
    return [
        {"prompt": f"Calculate {a} + {b}.\nAnswer:", "response": f" {a + b}."}
        for a in range(offset, offset + 8)
        for b in range(2, 10)
    ]


def encode(tokenizer, rows):
    encoded = []
    for row in rows:
        if "text" in row:
            tokens = tokenizer.encode(row["text"], add_special_tokens=False)[:64]
            if len(tokens) < 2:
                raise ValueError("the 64-token smoke benchmark requires at least one response target")
            encoded.append(
                TrainSequence(
                    tokens=tokens,
                    prompt_len=1,
                    logprobs=[],
                    scalar_advantage=1.0,
                    reward=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                )
            )
            continue
        prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
        response = tokenizer.encode(row["response"], add_special_tokens=False)
        tokens = (prompt + response)[:64]
        if len(tokens) <= len(prompt):
            raise ValueError("response targets do not fit the smoke benchmark; use scripts/galore_sft for long prompts")
        encoded.append(
            TrainSequence(
                tokens=tokens,
                prompt_len=len(prompt),
                logprobs=[],
                scalar_advantage=1.0,
                reward=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )
        )
    return encoded


def evaluate(trainer, rows):
    scores = trainer.score_logprobs("actor", [row.tokens for row in rows], microbatch_size=2)
    # The public scorer restores a leading zero; index i scores token i.
    targets = [value for row, values in zip(rows, scores, strict=True) for value in values[row.prompt_len :]]
    return -sum(targets) / len(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--optimizer", choices=["adamw", "adam8", "galore"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--projector-backend", choices=["svd", "recycled"], default="svd")
    parser.add_argument("--compact-master", action="store_true")
    parser.add_argument("--tile-rows", type=int, default=16384)
    parser.add_argument("--fused-writeback", action="store_true")
    parser.add_argument("--transport-moments", action="store_true")
    parser.add_argument("--exclude-vocab", action="store_true", help="Use ordinary AdamW for vocabulary parameters")
    parser.add_argument("--data-dir", help="Read fixed train.jsonl/validation.jsonl verbatim")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--diagnostic-gspo", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--dataset", choices=["repo-text", "arithmetic"], default="repo-text")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.data_dir:
        data_dir = Path(args.data_dir)
        train_rows, val_rows = (
            [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "validation")
        )
    else:
        train_rows, val_rows = examples("train", args.dataset), examples("validation", args.dataset)[:16]
    for name, rows in (("train", train_rows), ("validation", val_rows)):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if args.data_only:
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["ARENO_BENCH_SEED"] = str(args.seed)
    optimizer = dict(lr=1e-5, min_lr=1e-5, betas=(0.9, 0.95), weight_decay=0.01, grad_clip_norm=1.0)
    if args.optimizer == "galore":
        optimizer.update(
            galore_rank=args.rank,
            galore_update_proj_gap=args.gap,
            galore_scale=args.scale,
            galore_projector_backend=args.projector_backend,
            galore_projector_seed=args.seed,
            galore_compact_master=args.compact_master,
            galore_tile_rows=args.tile_rows if args.compact_master else 0,
            galore_release_gradients=args.compact_master,
            galore_fused_writeback=args.fused_writeback,
            galore_transport_moments=args.transport_moments,
            galore_exclude_vocab=args.exclude_vocab,
        )
    elif args.optimizer == "adam8":
        optimizer["adam_8bit"] = True
    if args.resume:
        optimizer["resume_training_state"] = True
    if args.save and args.optimizer != "galore":
        optimizer["save_training_state"] = True
    runtime = dict(attn_backend="native", compile_model=False, activation_checkpointing=True, keep_rollout_state=True)
    if "profile_training" in RuntimeConfig.__dataclass_fields__:
        runtime["profile_training"] = True
    trainer = Trainer(
        world_size=1,
        model_path=args.model,
        custom_config=CudaConfig(devices=[0], tp_size=1, optimizer=optimizer, runtime=runtime),
        metrics_log_dir=str(output / "events"),
    )
    metadata = dict(
        arguments=vars(args),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dtype="bfloat16",
        batch_size=2,
        context_cap=64,
        warmup_steps=2,
        train_sha256=hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256((output / "validation.jsonl").read_bytes()).hexdigest(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    source_root = Path(__import__("areno").__file__).resolve().parent
    source_digest = hashlib.sha256()
    for source in sorted(source_root.rglob("*.py")):
        source_digest.update(str(source.relative_to(source_root)).encode())
        source_digest.update(source.read_bytes())
    metadata["source_sha256"] = source_digest.hexdigest()
    metadata["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    try:
        trainer.init()
        tokenizer = trainer.get_tokenizer()
        train = encode(tokenizer, train_rows)
        validation = encode(tokenizer, val_rows)
        order = list(range(len(train)))
        random.Random(args.seed).shuffle(order)
        initial_loss = evaluate(trainer, validation)
        start_step = trainer._ctx.global_step + 1
        with (output / "steps.jsonl").open("w") as raw:
            for index in range(args.steps):
                step = start_step + index
                indices = [order[(2 * step + item) % len(order)] for item in range(2)]
                batch = [train[sample_index] for sample_index in indices]
                scored = trainer.score_logprobs("actor", [sample.tokens for sample in batch], microbatch_size=2)
                for sample, logprobs in zip(batch, scored, strict=True):
                    sample.logprobs = list(logprobs)
                result = trainer.train(batch, sft_loss_fn, mini_bs=2)
                if "logp_abs_diff_mean" not in result:
                    raise AssertionError("native SFT logprob metric is missing")
                row = dict(step=step, sample_indices=indices, warmup=index < 2, **result)
                raw.write(json.dumps(row) + "\n")
                raw.flush()
                print(json.dumps(row), flush=True)
        final_loss = evaluate(trainer, validation)
        # Store a fixed logprob vector for genuine checkpoint reload comparisons.
        score_probe = trainer.score_logprobs("actor", [validation[0].tokens], microbatch_size=1)
        summary = dict(
            initial_validation_loss=initial_loss,
            final_validation_loss=final_loss,
            probe_logprobs=score_probe,
            completed_steps=args.steps,
            next_step=trainer._ctx.global_step + 1,
        )
        if args.save:
            summary["checkpoint"] = trainer.save_checkpoint(str(output / "checkpoint"))
        if args.diagnostic_gspo:
            batch = train[:2]
            old = trainer.score_logprobs("actor", [row.tokens for row in batch], microbatch_size=2)
            for row, values in zip(batch, old, strict=True):
                row.logprobs = list(values)
            diagnostic = trainer.train(batch, gspo_loss_fn, mini_bs=2)
            summary["diagnostic_gspo"] = diagnostic
            if "logp_abs_diff_mean" not in diagnostic:
                raise AssertionError("native GSPO logprob diagnostic is missing")
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
