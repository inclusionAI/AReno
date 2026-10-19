"""Small helpers and isolated AReno workers for the Ling Tiny notebook."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from collections import Counter
from pathlib import Path

TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "f_proj", "o_proj",
    "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "dense",
)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def typed_equal(actual: object, expected: object) -> bool:
    """Compare the complete JSON tree, including types and array order."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(typed_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON constant: {value}")


def strict_score(completion: str, expected: dict) -> dict:
    """No substring extraction, fences, coercion, or partial business credit."""
    try:
        parsed = json.loads(completion, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, TypeError):
        return {"valid_json": False, "exact_match": False}
    return {"valid_json": True, "exact_match": typed_equal(parsed, expected)}


def check_splits(train: list[dict], valid: list[dict]) -> None:
    if (len(train), len(valid)) != (1400, 100):
        raise ValueError("Expected the published fixed splits: 1,400 train and 100 validation rows.")
    ids, requests = set(), set()
    for row in train + valid:
        for field in ("case_id", "bucket", "prompt", "response"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Missing or empty {field}")
        if not strict_score(row["response"], row["ground_truth"])["exact_match"]:
            raise ValueError(f"Target disagrees with ground_truth: {row['case_id']}")
        request = dict(row["input"])
        request.pop("order_id", None)
        fingerprint = json.dumps(request, sort_keys=True, ensure_ascii=False)
        if row["case_id"] in ids or fingerprint in requests:
            raise ValueError(f"Duplicate example or overlapping splits: {row['case_id']}")
        ids.add(row["case_id"])
        requests.add(fingerprint)


def tokenize_rows(rows: list[dict], tokenizer, config: dict) -> list:
    from areno.api import TrainSequence
    from areno.api.data_utils import prompt_response_to_tokens_and_mask

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("The checkpoint tokenizer must define eos_token_id.")
    sequences = []
    for row in rows:
        tokens, mask = prompt_response_to_tokens_and_mask(row["prompt"], row["response"], tokenizer, eos)
        prompt_len = sum(mask)
        response_len = len(tokens) - prompt_len
        if (prompt_len > config["max_prompt_tokens"] or response_len > config["max_new_tokens"]
                or len(tokens) > config["max_context_len"]):
            raise ValueError(
                f"{row['case_id']}: prompt={prompt_len}, response={response_len}, total={len(tokens)} "
                "exceeds the configured token budgets. Increase the budgets explicitly; no rows were dropped."
            )
        sequences.append(TrainSequence(
            tokens=tokens, prompt_mask=mask, eos_token_id=eos,
            logprobs=[0.0] * len(tokens), advantages=[0.0] * len(tokens),
        ))
    return sequences


def preflight(config: dict) -> dict:
    from areno.api.tokenizer import configure_chat_template_enable_thinking, load_tokenizer

    if config["epochs"] < 1 or config["batch_size"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive.")
    if config["max_prompt_tokens"] + config["max_new_tokens"] > config["max_context_len"]:
        raise ValueError("max_context_len must cover prompt plus generation budgets.")
    tokenizer = load_tokenizer(config["model_path"])
    configure_chat_template_enable_thinking(tokenizer, False)
    train, valid = (read_jsonl(config[key]) for key in ("train_file", "valid_file"))
    check_splits(train, valid)
    result = {}
    for name, rows in (("train", train), ("validation", valid)):
        seqs = tokenize_rows(rows, tokenizer, config)
        result[name] = {
            "rows": len(seqs), "max_prompt_tokens": max(sum(s.prompt_mask) for s in seqs),
            "max_response_tokens": max(len(s.tokens) - sum(s.prompt_mask) for s in seqs),
            "max_total_tokens": max(len(s.tokens) for s in seqs),
        }
    return result


def create_trainer(config: dict, stage: str):
    from areno import Trainer
    from areno.api import CUDA, MLX, CudaConfig, LoraConfig, MlxConfig

    adapter = str(Path(config["run_dir"]) / "adapter") if stage == "after" else None
    lora = LoraConfig(
        rank=config["lora_rank"], alpha=config["lora_alpha"],
        target_modules=TARGET_MODULES, adapter_path=adapter,
    )
    optimizer = {
        "lr": config["learning_rate"], "min_lr": config["learning_rate"],
        "lr_decay_style": "constant", "weight_decay": 0.0, "grad_clip_norm": 1.0,
    }
    common = dict(lora=lora, optimizer=optimizer, base_model_name_or_path=config["model_id"])
    if config["backend"] == "mlx":
        import mlx.core as mx

        mx.random.seed(config["seed"])
        backend = MLX
        backend_config = MlxConfig(
            **common, compile_train_step=False, gradient_checkpointing=True,
            keep_rollout_state=False, completion_batch_size=1, prefill_batch_size=1,
            max_running_prompts=1, logits_chunk_size=256,
        )
    else:
        import torch

        torch.manual_seed(config["seed"])
        backend = CUDA
        backend_config = CudaConfig(
            **common, tp_size=1, max_running_prompts=1,
            runtime={"activation_checkpointing": True, "keep_rollout_state": False,
                     "eager_decode": True, "attn_backend": "native"},
        )
    return Trainer(
        world_size=1, model_path=config["model_path"], backend_type=backend,
        custom_config=backend_config, metrics_log_dir=str(Path(config["run_dir"]) / "events" / stage),
    )


def train(trainer, rows: list[dict], config: dict) -> None:
    from areno.api import sft_loss_fn

    sequences = tokenize_rows(rows, trainer.get_tokenizer(), config)
    run_dir = Path(config["run_dir"])
    step = 0
    with (run_dir / "training.jsonl").open("w", encoding="utf-8") as stream:
        for epoch in range(config["epochs"]):
            order = list(range(len(sequences)))
            random.Random(config["seed"] + epoch).shuffle(order)
            for start in range(0, len(order), config["batch_size"]):
                batch = [sequences[i] for i in order[start:start + config["batch_size"]]]
                metrics = trainer.train(batch, sft_loss_fn, mini_bs=1)
                step += 1
                if not math.isfinite(float(metrics["loss"])):
                    raise RuntimeError(f"Non-finite training loss at step {step}")
                record = {"step": step, "epoch": epoch + 1, "rows": len(batch), **metrics}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(json.dumps(record), flush=True)
    # Save after the final batch, even when it does not coincide with an interval.
    trainer.save_checkpoint(str(run_dir / "adapter"))
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (run_dir / "adapter" / name).is_file():
            raise RuntimeError(f"Missing saved adapter file: {name}")


async def evaluate(trainer, rows: list[dict], config: dict, stage: str) -> None:
    from areno.api import SamplingParams

    tokenizer = trainer.get_tokenizer()
    # Recheck the backend's tokenizer; MLX may supply its own wrapper.
    tokenize_rows(rows, tokenizer, config)
    params = SamplingParams(
        greedy=True, seed=config["seed"], max_new_tokens=config["max_new_tokens"],
        max_context_len=config["max_context_len"],
    )
    counts, buckets = Counter(), {}
    with (Path(config["run_dir"]) / f"{stage}.jsonl").open("w", encoding="utf-8") as stream:
        async with trainer.rollout_session(sampling_params=params, max_running_prompts=1, proxy=False):
            for index, row in enumerate(rows):
                result = trainer.rollout_batch([row["prompt"]], n_samples=1, sampling_params=params)[0]
                tokens = result.sequences[0].resp_tokens
                completion = tokenizer.decode(tokens, skip_special_tokens=True)
                score = strict_score(completion, row["ground_truth"])
                count = {"rows": 1, **score, "at_token_limit": len(tokens) >= config["max_new_tokens"]}
                counts.update(count)
                buckets.setdefault(row["bucket"], Counter()).update(count)
                stream.write(json.dumps({
                    "case_id": row["case_id"], "bucket": row["bucket"], "completion": completion,
                    "expected": row["ground_truth"], **score, "generated_tokens": len(tokens),
                }, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{stage}: {index + 1}/{len(rows)}", flush=True)
        trainer.finish_step()
    report = {"split": "validation", **counts, "buckets": buckets}
    report["exact_match_rate"] = counts["exact_match"] / counts["rows"]
    (Path(config["run_dir"]) / f"{stage}-summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("before", "train", "after"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    # Every stage fails before allocating model weights if data exceeds budgets.
    preflight(config)
    from areno.api.tokenizer import configure_chat_template_enable_thinking

    trainer = create_trainer(config, args.stage)
    try:
        trainer.init()
        configure_chat_template_enable_thinking(trainer.get_tokenizer(), False)
        if args.stage == "train":
            train(trainer, read_jsonl(config["train_file"]), config)
        else:
            asyncio.run(evaluate(trainer, read_jsonl(config["valid_file"]), config, args.stage))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
