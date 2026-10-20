"""Generate and score an independent, held-out OPD evaluation set.

Loads one HF-compatible student checkpoint at a time (base or after OPD).
Weights are not downloaded by --dry-run; real evaluation requires weights.
The deterministic generation metric and reference-answer NLL are reported
separately, as neither is identical to the on-policy distillation objective.

Example:
  python examples/opd/evaluate_models.py --model Qwen/Qwen3-0.6B \
    --output outputs/opd-eval/before.json
  python examples/opd/evaluate_models.py \
    --model outputs/opd-smoke/checkpoints/step_000002 \
    --tokenizer Qwen/Qwen3-0.6B --output outputs/opd-eval/after.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eval_core import exact_match, load_eval_set, summary_from_rows, validate_train_holdout_disjoint

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "examples/opd/heldout_eval.jsonl"
DEFAULT_TRAIN = ROOT / "examples/opd/prompts.jsonl"


def encode_prompt(tokenizer, prompt: str, *, disable_thinking: bool = True) -> list[int]:
    """Mirror AReno's encode_generation_prompt for raw user prompts."""
    if not getattr(tokenizer, "chat_template", None) or any(
        marker in prompt for marker in ("<|im_start|>", "<start_of_turn>", "<turn|>", "<|user|>", "<|assistant|>")
    ):
        return list(tokenizer.encode(prompt))
    kwargs = dict(tokenize=True, add_generation_prompt=True)
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs))
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs))


def _device_from_arg(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return requested


def _sync(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def evaluate_one(model, tokenizer, torch, device: str, row: dict, *, max_new_tokens: int) -> dict:
    """One deterministic generated answer and teacher-forced gold-answer NLL."""
    prompt_ids = encode_prompt(tokenizer, row["prompt"])
    answer_ids = list(tokenizer.encode(row["answer"], add_special_tokens=False))
    if not prompt_ids or not answer_ids:
        raise ValueError("empty encoded prompt/answer; cannot compute next-token NLL")
    if len(prompt_ids) < 1:
        raise ValueError("prompt must have at least one token")
    ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    _sync(torch, device)
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    _sync(torch, device)
    generation_s = time.perf_counter() - start
    sampled = generated[0, len(prompt_ids):].tolist()
    completion = tokenizer.decode(sampled, skip_special_tokens=True).strip()

    # Log p(y_i | prompt, y_{<i}) at positions prefix_len - 1 through ...
    full = torch.tensor([prompt_ids + answer_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=full).logits.float()[0]
        begin = len(prompt_ids) - 1
        answer_logits = logits[begin:begin + len(answer_ids), :]
        targets = full[0, len(prompt_ids):]
        if answer_logits.shape[0] != targets.shape[0]:
            raise RuntimeError("wrong answer-token alignment")
        nll = -torch.nn.functional.log_softmax(answer_logits, dim=-1).gather(
            -1, targets[:, None]
        ).sum().item()
    if not math.isfinite(nll) or nll < -1e-5:
        raise ValueError("nonfinite or negative held-out NLL")
    return {"id": row["id"], "prompt": row["prompt"], "answer": row["answer"],
            "aliases": row.get("aliases", []), "completion": completion,
            "correct": exact_match(completion, row),
            "answer_nll_sum": max(0.0, float(nll)), "answer_tokens": len(answer_ids),
            "generated_tokens": len(sampled), "generation_wall_time_s": generation_s}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model identifier or real saved checkpoint directory")
    p.add_argument("--tokenizer", help="Original student tokenizer (recommended for saved checkpoints)")
    p.add_argument("--dataset", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--training-dataset", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.output.exists():
        raise SystemExit("Output exists; choose a fresh --output path to avoid overwriting evidence")
    dataset, fingerprint = load_eval_set(args.dataset)
    validate_train_holdout_disjoint(args.training_dataset, dataset)
    if args.dry_run:
        print(json.dumps({"validated_prompts": len(dataset), "dataset_sha256": fingerprint,
                          "model": args.model, "tokenizer": args.tokenizer or args.model,
                          "no_model_loaded": True}, indent=2))
        return 0
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install torch and compatible transformers before real model evaluation") from exc
    device = _device_from_arg(torch, args.device)
    dtype = {"auto": torch.float32 if device == "cpu" else (torch.float16 if device == "cuda" else torch.float32),
             "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=False, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=False, torch_dtype=dtype)
    model.to(device).eval()
    rows = [evaluate_one(model, tokenizer, torch, device, row, max_new_tokens=args.max_new_tokens) for row in dataset]
    result = {"schema_version": 1, "evidence_kind": "real_model_inference", "model": args.model,
              "tokenizer": args.tokenizer or args.model, "device": device,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_sha256": fingerprint, "max_new_tokens": args.max_new_tokens,
              "decoding": "greedy", "disable_thinking": True,
              "metrics": summary_from_rows(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "output": str(args.output), "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
