"""Longer matched public-Trainer experiments, never select using validation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from common import batch_indices, encode, evaluate, metadata, parser, setup, trainer, write

from areno.api import sft_loss_fn

UTC = timezone.utc


def main():
    p = parser(__doc__)
    p.add_argument("--steps", type=int, default=1024)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-from")
    p.add_argument("--eval-every", type=int, default=256)
    p.add_argument("--monitor-examples", type=int, default=128)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--stop-file", type=Path)
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    log = setup(args, "train")
    output_checkpoint = args.checkpoint_dir
    if args.resume_from:
        args.checkpoint_dir = Path(args.resume_from)
        args.resume = True
    info = metadata(args)
    info["checkpoint_load_path"] = str(args.checkpoint_dir) if args.resume else None
    info["checkpoint_output_path"] = str(output_checkpoint)
    write(args.output_dir / "metadata.json", info)
    engine = trainer(args, resume=args.resume)
    args.checkpoint_dir = output_checkpoint
    started = time.perf_counter()
    measured, all_rows, orders, native_errors, cumulative_targets = [], [], {}, [], 0
    try:
        engine.init()
        training, train_stats = encode(engine.get_tokenizer(), args, "train")
        validation, val_stats = encode(engine.get_tokenizer(), args, "validation")
        write(args.output_dir / "encoding.json", {"train": train_stats, "validation": val_stats})
        # Fixed validation subset is a monitor only; never changes training or selection.
        monitor = [
            validation[i]
            for i in sorted(range(len(validation)), key=lambda i: hashlib.sha256(str(i).encode()).hexdigest())[
                : args.monitor_examples
            ]
        ]
        start_step = engine._ctx.global_step + 1
        write(
            args.output_dir / "heartbeat.json",
            {"status": "initialized", "step": start_step, "timestamp": datetime.now(UTC).isoformat()},
        )
        with (args.output_dir / "steps.jsonl").open("w") as raw, (args.output_dir / "curve.jsonl").open("w") as curves:

            def record_eval(step):
                result = dict(
                    step=step,
                    timestamp=datetime.now(UTC).isoformat(),
                    split="validation_monitor",
                    **evaluate(engine, monitor, args.micro_batch),
                )
                curves.write(json.dumps(result) + "\n")
                curves.flush()
                log.info("validation_monitor %s", json.dumps(result))

            record_eval(start_step)
            for index in range(args.steps):
                if args.stop_file and args.stop_file.exists():
                    next_step = engine._ctx.global_step + 1
                    recovery = (
                        engine.save_checkpoint(str(args.output_dir / "recovery")) if next_step > start_step else None
                    )
                    write(args.output_dir / "recovery.json", {"checkpoint": recovery, "next_step": next_step})
                    write(
                        args.output_dir / "heartbeat.json",
                        {"status": "preempted", "step": next_step, "timestamp": datetime.now(UTC).isoformat()},
                    )
                    raise SystemExit(75)
                step = start_step + index
                indices = batch_indices(step, args.seed, len(training), args.batch, orders)
                batch = [training[i] for i in indices]
                stamp = datetime.now(UTC).isoformat()
                begin = time.perf_counter()
                scored = engine.score_logprobs("actor", [s.tokens for s in batch], microbatch_size=args.micro_batch)
                for sample, values in zip(batch, scored, strict=True):
                    sample.logprobs = list(values)
                scored_at = time.perf_counter()
                result = engine.train(batch, sft_loss_fn, mini_bs=args.micro_batch)
                if result.get("update_optimizer_steps") != 1:
                    raise AssertionError("one public call must perform exactly one optimizer update")
                if not math.isfinite(result["loss"]) or "logp_abs_diff_mean" not in result:
                    raise AssertionError("nonfinite loss or missing native SFT diagnostic")
                native_errors.append(result["logp_abs_diff_mean"])
                targets = sum(len(s.tokens) - s.prompt_len for s in batch)
                cumulative_targets += targets
                row = dict(
                    timestamp=stamp,
                    step=step,
                    sample_indices=indices,
                    batch_sha256=hashlib.sha256(
                        json.dumps([(s.tokens, s.prompt_len) for s in batch], separators=(",", ":")).encode()
                    ).hexdigest(),
                    target_tokens=targets,
                    cumulative_target_tokens=cumulative_targets,
                    epoch=(step * args.batch) // len(training),
                    warmup=index < 2,
                    scoring_seconds=scored_at - begin,
                    score_plus_train_seconds=time.perf_counter() - begin,
                    **result,
                )
                raw.write(json.dumps(row) + "\n")
                all_rows.append(row)
                raw.flush()
                log.info("step %s", json.dumps(row))
                write(
                    args.output_dir / "heartbeat.json",
                    {"status": "training", "step": step, "timestamp": datetime.now(UTC).isoformat()},
                )
                if index >= 2:
                    measured.append(row)
                if args.eval_every and (step + 1) % args.eval_every == 0:
                    record_eval(step + 1)
                if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0 and index + 1 < args.steps:
                    # Recovery checkpoint in its own path; final endpoint is separate.
                    recovery = engine.save_checkpoint(str(args.output_dir / "recovery"))
                    write(args.output_dir / "recovery.json", {"checkpoint": recovery, "next_step": step + 1})
            if (start_step + args.steps) % max(args.eval_every, 1):
                record_eval(start_step + args.steps)
        args.checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = engine.save_checkpoint(str(args.checkpoint_dir))
        normal = [r for r in measured if not r["update_refresh_any"]]
        refresh = [r for r in measured if r["update_refresh_any"]]

        def mean(rows, key):
            return sum(r[key] for r in rows) / len(rows) if rows else None

        summary = dict(
            status="complete",
            timestamp=datetime.now(UTC).isoformat(),
            seed=args.seed,
            optimizer=args.optimizer,
            start_step=start_step,
            completed_updates=args.steps,
            next_step=engine._ctx.global_step + 1,
            checkpoint=checkpoint,
            elapsed_seconds=time.perf_counter() - started,
            target_tokens=cumulative_targets,
            normal_seconds=mean(normal, "update_cuda_time_s"),
            refresh_seconds=mean(refresh, "update_cuda_time_s"),
            measured_refreshes=len(refresh),
            measured_steps=len(measured),
            mean_score_plus_train_seconds=mean(measured, "score_plus_train_seconds"),
            target_tokens_per_compute_second=(
                sum(r["target_tokens"] for r in measured) / sum(r["score_plus_train_seconds"] for r in measured)
            )
            if measured
            else None,
            allocated_peak_bytes=max(r["update_peak_allocated_bytes"] for r in measured or all_rows),
            reserved_peak_bytes=max(r["update_peak_reserved_bytes"] for r in measured or all_rows),
            native_logp_abs_diff_max=max(native_errors),
            last_train_loss=all_rows[-1]["loss"],
            trailing_train_loss=mean(all_rows[-64:], "loss"),
            limitation="new-corpus exploratory track; full final validation/test evaluated independently",
        )
        write(args.output_dir / "summary.json", summary)
        write(
            args.output_dir / "heartbeat.json",
            {"status": "complete", "step": summary["next_step"], "timestamp": datetime.now(UTC).isoformat()},
        )
        log.info("complete %s", json.dumps(summary))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
