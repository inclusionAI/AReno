"""Greedy GSM8K rollouts with a selected KV-cache dtype through the public Trainer.

Dumps per-prompt response tokens, rollout (behavior-policy) logprobs and math
rewards as JSONL, plus a summary JSON, so BF16 and KIVI runs over the same
prompts can be compared token by token with ``compare_rollouts.py``.

    python examples/kivi/rollout_eval.py --model /path/to/Qwen3-0.6B \
        --dataset /path/to/gsm8k_main_test --kv-cache-dtype kivi2 --output runs/kivi2
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import time
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local checkpoint directory.")
    parser.add_argument("--dataset", required=True, help="GSM8K split saved with `save_to_disk`.")
    parser.add_argument("--output", required=True, help="Output directory for rollouts.jsonl and summary.json.")
    parser.add_argument("--kv-cache-dtype", choices=["auto", "kivi2", "kivi4"], default="auto")
    parser.add_argument("--kivi-group-size", type=int, default=32)
    parser.add_argument("--kivi-residual-length", type=int, default=128)
    parser.add_argument("--attn-backend", choices=["flash", "native"], default="flash")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N prompts.")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-running-prompts", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=1, help="Completions per prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 selects greedy decoding.")
    parser.add_argument("--seed", type=int, default=None, help="Rollout sampling seed (stochastic decoding).")
    return parser.parse_args()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _rollout(trainer, prompt_tokens, n_samples, sampling_params, max_running_prompts):
    async with trainer.rollout_session(
        sampling_params=sampling_params, max_running_prompts=max_running_prompts, proxy=False
    ):
        return await trainer.rollout_token_batch_async(prompt_tokens, n_samples, sampling_params)


def main() -> None:
    args = parse_args()
    from datasets import load_from_disk

    from areno import Trainer
    from areno.api.config import CudaConfig
    from areno.api.models import SamplingParams
    from areno.api.rewards import load_reward_fn, make_reward_record
    from areno.api.tokenizer import encode_generation_prompt

    loader = _load_module(EXAMPLES / "math" / "dataset_loader.py", "math_dataset_loader")
    dataset = loader.load_training_dataset(args.dataset, default_loader=load_from_disk)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    reward_fn = load_reward_fn(str(EXAMPLES / "math" / "math_verify_reward.py"))

    runtime = {"attn_backend": args.attn_backend, "kv_cache_dtype": args.kv_cache_dtype}
    if args.kv_cache_dtype != "auto":
        runtime.update(kivi_group_size=args.kivi_group_size, kivi_residual_length=args.kivi_residual_length)
    trainer = Trainer(
        1,
        args.model,
        custom_config=CudaConfig(devices=[0], max_running_prompts=args.max_running_prompts, runtime=runtime),
    )
    trainer.init()
    try:
        tokenizer = trainer.get_tokenizer()
        prompt_tokens = [encode_generation_prompt(tokenizer, record["prompt"]) for record in dataset]
        sampling_params = SamplingParams(
            greedy=args.temperature == 0.0,
            temperature=args.temperature if args.temperature > 0 else 1.0,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            max_prompt_len=args.max_prompt_tokens,
        )
        start = time.perf_counter()
        results = asyncio.run(
            _rollout(trainer, prompt_tokens, args.n_samples, sampling_params, args.max_running_prompts)
        )
        elapsed = time.perf_counter() - start
    finally:
        trainer.close()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        (index, sample, sequence)
        for index, result in enumerate(results)
        for sample, sequence in enumerate(result.sequences)
    ]
    sequences = [sequence for _, _, sequence in rows]
    completions = list(tokenizer.batch_decode([sequence.resp_tokens for sequence in sequences]))
    rewards = []
    with (output / "rollouts.jsonl").open("w") as handle:
        for (index, sample, sequence), completion in zip(rows, completions, strict=True):
            record = dataset[index]
            reward = reward_fn(
                make_reward_record(
                    prompt=record["prompt"],
                    completion=completion,
                    source_record=dict(record),
                    answer=record["solutions"],
                )
            )
            rewards.append(reward)
            row = {
                "index": index,
                "sample": sample,
                "prompt_tokens": len(prompt_tokens[index]),
                "resp_tokens": sequence.resp_tokens,
                "resp_logprobs": sequence.resp_logprobs,
                "reward": reward,
            }
            handle.write(json.dumps(row) + "\n")
    generated = sum(len(sequence.resp_tokens) for sequence in sequences)
    summary = {
        "kv_cache_dtype": args.kv_cache_dtype,
        "runtime": runtime,
        "prompts": len(results),
        "samples_per_prompt": args.n_samples,
        "temperature": args.temperature,
        "seed": args.seed,
        "accuracy": sum(rewards) / max(1, len(rewards)),
        # Binomial standard error of the mean reward over all completions.
        "accuracy_stderr": ((sum(rewards) / len(rewards)) * (1 - sum(rewards) / len(rewards)) / len(rewards)) ** 0.5
        if rewards
        else None,
        "generated_tokens": generated,
        "mean_response_tokens": generated / max(1, len(sequences)),
        "rollout_seconds": elapsed,
        "generated_tokens_per_s": generated / elapsed,
        "max_new_tokens": args.max_new_tokens,
        "max_running_prompts": args.max_running_prompts,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
