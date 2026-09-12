"""End-to-end decode benchmark for the FP8 decode path.

Drives the real engine path (`ArenoEngine.begin_rollout_session` ->
`generate_rollout` under decode CUDA graphs) on a local checkpoint, once with
`--quant none` and once with `--quant fp8`, and reports decode tokens/s plus
peak device memory.

    CUDA_VISIBLE_DEVICES=0 python scripts/bench/fp8_decode_e2e_bench.py \
        --model ~/models/Qwen3-8B --quant none --runs 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

from areno.engine import ArenoEngine
from areno.engine.config import RuntimeConfig
from areno.engine.data import SamplingParams
from areno.engine.data.tokenizer import load_tokenizer

_BENCH_PROMPT = (
    "Write a detailed step-by-step plan for organizing a small research codebase: "
    "repository layout, naming conventions, test structure, documentation, and "
    "release automation. Include concrete examples for each step."
)


def _gpu_peak_mb(stop: dict, out: dict) -> None:
    """Sample every GPU and keep the high-water mark (engine workers own the
    memory, so the parent cannot use torch counters; index sampling breaks
    under CUDA_VISIBLE_DEVICES remapping)."""
    peak = 0
    while not stop["stop"]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.strip().splitlines():
            peak = max(peak, int(line))
        time.sleep(0.05)
    out["peak_mb"] = peak


def _bench_loss_fn(*_: object) -> object:
    """Placeholder loss function; the benchmark only decodes, never trains."""
    raise RuntimeError("the fp8 decode benchmark engine does not support training")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--quant", choices=["none", "fp8"], default="none")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1, help="number of concurrent prompts (1 = fp8 gemv window)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--attn-backend", choices=["native", "flash"], default="native")
    args = parser.parse_args()

    engine = ArenoEngine.from_pretrained(
        args.model,
        tp_size=args.tp,
        dp_size=1,
        devices=list(range(args.device, args.device + args.tp)),
        quant_method=args.quant,
        loss_fn=_bench_loss_fn,
        runtime_config=RuntimeConfig(attn_backend=args.attn_backend),
    )
    try:
        tokenizer = load_tokenizer(args.model)
        prompt_ids = tokenizer(_BENCH_PROMPT, add_special_tokens=True)["input_ids"]
        prompts = [prompt_ids for _ in range(args.batch)]

        import threading

        stop = {"stop": False}
        gpu_peak: dict = {}
        sampler = threading.Thread(target=_gpu_peak_mb, args=(stop, gpu_peak), daemon=True)
        sampler.start()

        engine.begin_rollout_session()
        # Warmup (compiles CUDA graphs, allocates caches).
        warm = engine.generate_rollout(
            prompts,
            max_new_tokens=args.max_new_tokens,
            max_running_prompts=args.batch,
            sampling_params=SamplingParams(temperature=0.0),
        )
        generated = sum(len(row) for row in warm.response_ids)
        results = []
        for run in range(args.runs):
            start = time.perf_counter()
            out = engine.generate_rollout(
                prompts,
                max_new_tokens=args.max_new_tokens,
                max_running_prompts=args.batch,
                sampling_params=SamplingParams(temperature=0.0),
            )
            elapsed = time.perf_counter() - start
            tokens = sum(len(row) for row in out.response_ids)
            results.append(
                {
                    "run": run,
                    "tokens": tokens,
                    "seconds": round(elapsed, 3),
                    "tokens_per_s": round(tokens / elapsed, 1),
                }
            )
        stop["stop"] = True
        sampler.join(timeout=2)
        engine.end_rollout_session()

        total_tokens = sum(r["tokens"] for r in results)
        total_time = sum(r["seconds"] for r in results)
        print(
            json.dumps(
                {
                    "quant": args.quant,
                    "model": args.model,
                    "tp": args.tp,
                    "batch": args.batch,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_tokens": len(prompt_ids),
                    "warmup_tokens": generated,
                    "runs": results,
                    "aggregate_tokens_per_s": round(total_tokens / total_time, 1),
                    "peak_gpu_mem_mb": gpu_peak.get("peak_mb"),
                    "sample_text": tokenizer.decode(out.response_ids[0][:40]),
                }
            )
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
