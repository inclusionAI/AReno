"""Bounded eager CUDA timing: unchanged full prefill and mixed prefill/decode.

Uses the real scheduler's prefill payloads and fixed decode tokens. These are
model/scheduler timings, excluding HTTP, tokenization, sampling and queueing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from areno.engine.config import EngineConfig, RuntimeConfig
from areno.engine.data.rollout_state import InferenceBatchState, payload_to_infer_meta
from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend
from areno.engine.modeling import build_model_on_device
from areno.engine.runtime.metadata import InferMeta
from areno.models.registry import config_from_hf, load_model_weights


def percentile(values, p):
    return float(torch.tensor(values, dtype=torch.float64).quantile(p))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="Original inference backend Python file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prompt-tokens", type=int, default=1025)
    parser.add_argument("--decode-tokens", type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    spec = importlib.util.spec_from_file_location("baseline_infer", args.baseline)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    config = config_from_hf(args.model)
    engine = EngineConfig(
        config,
        model_path=str(args.model),
        devices=[0],
        role="rollout",
        runtime=RuntimeConfig(compile_model=False, eager_decode=True),
    )
    model = build_model_on_device(engine, device)
    load_model_weights(model, config, args.model)
    model.eval()
    model.prepare_infer_weights()
    cache_len = max(2048, args.prompt_tokens + args.decode_tokens + 256)
    cache_blocks = 2 * ((cache_len + 255) // 256) + 1
    model.set_kv_caches(model.allocate_kv_caches(cache_blocks, 256, device), num_slots=3)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    stream = tokenizer.encode(
        "Alice has three apples. Bob has five apples. " * args.prompt_tokens, add_special_tokens=False
    )
    prompts = [stream[:32], stream[: args.prompt_tokens]]

    def sync_time():
        torch.cuda.synchronize()
        return time.perf_counter()

    def run(backend, budget):
        for layer in model.layers:
            if hasattr(layer.attention, "infer_backend"):
                layer.attention.infer_backend = backend("flash")
        model.reset_kv_caches()
        state = InferenceBatchState(
            prompts,
            args.decode_tokens,
            max_running_seqs=2,
            max_cache_len=cache_len,
            max_prefill_tokens=32,
            kv_block_size=256,
            num_cache_blocks=cache_blocks - 1,
        )
        counts = [0, 0]
        gaps = []

        def forward(ids, positions, meta, indices=None):
            out = model(ids, positions, infer_meta=meta, defer_lm_head=True)
            if indices is None or indices.numel():
                hidden = out.hidden_states if indices is None else out.hidden_states[:, indices]
                model.lm_head(hidden.to(model.lm_head.weight.dtype))

        def prefill():
            p = state.build_prefill_payload()
            forward(
                p["input_ids"].to(device)[None],
                p["position_ids"].to(device)[None],
                payload_to_infer_meta(p, device),
                p["sample_indices"].to(device),
            )
            for seq in state._last_active_ids:
                counts[seq] = 1
            return list(state._last_active_ids)

        prefill()  # Seed the existing decode request before the long prompt arrives.
        state.max_prefill_tokens = budget
        start = sync_time()
        last_short = start
        long_ttft = None
        while state.has_pending_prompts or any(0 < n < args.decode_tokens for n in counts):
            if state.has_pending_prompts:
                ready = prefill()
                now = sync_time()
                if 1 in ready:
                    long_ttft = (now - start) * 1000
            active = [seq for seq, n in enumerate(counts) if 0 < n < args.decode_tokens]
            if not active:
                continue
            positions = [len(prompts[seq]) + counts[seq] - 1 for seq in active]
            state.ensure_decode_blocks(active, positions)
            table = [
                state._seq_to_blocks[seq] + [0] * (state.max_blocks_per_seq - len(state._seq_to_blocks[seq]))
                for seq in active
            ]
            meta = InferMeta(
                mode="decode",
                cache_seqlens=torch.tensor(positions, device=device, dtype=torch.int32),
                block_table=torch.tensor(table, device=device, dtype=torch.int32),
                recurrent_slots=torch.tensor([state._seq_to_recurrent_slot[seq] for seq in active], device=device),
            )
            forward(
                torch.tensor([[stream[counts[seq] % len(stream)] for seq in active]], device=device),
                torch.tensor([positions], device=device),
                meta,
            )
            now = sync_time()
            if 0 in active:
                gaps.append((now - last_short) * 1000)
                last_short = now
            for seq in active:
                counts[seq] += 1
        elapsed = (sync_time() - start) * 1000
        return {
            "elapsed_ms": elapsed,
            "generated_tokens_per_s": (2 * args.decode_tokens - 1) * 1000 / elapsed,
            "long_prompt_ttft_ms": long_ttft,
            "ongoing_decode_p50_ms": statistics.median(gaps),
            "ongoing_decode_p99_ms": percentile(gaps, 0.99),
            "ongoing_decode_max_gap_ms": max(gaps),
        }

    variants = [
        ("original_full", old.FlashAttnInferBackend, cache_len),
        ("fixed_full", FlashAttnInferBackend, cache_len),
        ("fixed_chunk256", FlashAttnInferBackend, 256),
        ("fixed_chunk128", FlashAttnInferBackend, 128),
    ]
    samples = {name: [] for name, _, _ in variants}
    for name, backend, budget in variants:
        run(backend, budget)
        print(json.dumps({"warmup": name}), flush=True)
    for repeat in range(args.repeats):
        for name, backend, budget in variants if repeat % 2 == 0 else variants[::-1]:
            row = run(backend, budget)
            samples[name].append(row)
            print(json.dumps({"repeat": repeat, "variant": name, **row}), flush=True)
    medians = {name: {k: statistics.median(row[k] for row in rows) for k in rows[0]} for name, rows in samples.items()}
    ratio = medians["fixed_full"]["generated_tokens_per_s"] / medians["original_full"]["generated_tokens_per_s"]
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "model": str(args.model),
        "prompt_lengths": list(map(len, prompts)),
        "decode_tokens_per_request": args.decode_tokens,
        "repeats": args.repeats,
        "decode_mode": "eager",
        "fixed_tokens": True,
        "medians": medians,
        "samples": samples,
        "unchanged_budget_throughput_ratio": ratio,
        "unchanged_budget_nonregression_threshold": 0.9,
        "performance_accepted": ratio >= 0.9,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
    if not result["performance_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
