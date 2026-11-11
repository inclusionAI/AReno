"""Bounded CUDA validation of cached-prefix attention and a local model checkpoint.

Run from the candidate checkout, with --baseline pointing to an isolated main
checkout. Timing pairs alternate on identical tensors; no checkpoint is written.
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
from areno.engine.layers.attention_backend import infer
from areno.engine.modeling import build_model_on_device
from areno.engine.runtime.metadata import InferMeta
from areno.models.registry import config_from_hf, load_model_weights


def baseline_module(root, relative, name):
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def elapsed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000


@torch.inference_mode()
def attention_benchmark(old, device, heads, head_dim):
    results = []
    for batch in (1, 8, 32):
        torch.manual_seed(635)
        prefix, chunk, page = 513, 64, 256
        lengths = [chunk - row % 3 for row in range(batch)]
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
        prefixes = torch.tensor(
            [prefix if row % 2 == 0 else 0 for row in range(batch)], dtype=torch.int32, device=device
        )
        kv_lengths = prefixes + cu.diff()
        cu_k = torch.cat((cu[:1], kv_lengths.cumsum(0, dtype=torch.int32)))
        cache_k = torch.randn(batch * 3, page, heads, head_dim, dtype=torch.bfloat16, device=device)
        cache_v = torch.randn_like(cache_k)
        q = torch.randn(int(cu[-1]), heads, head_dim, dtype=torch.bfloat16, device=device)
        meta = InferMeta(
            mode="prefill",
            cu_seqlens=cu,
            max_seqlen=chunk,
            cache_seqlens=prefixes,
            block_table=torch.arange(batch * 3, dtype=torch.int32, device=device).view(batch, 3),
            cu_seqlens_k=cu_k,
            max_seqlen_k=int(kv_lengths.max()),
        )
        args = (q, q, q, cache_k, cache_v, meta, "flash", (-1, -1), head_dim**-0.5)

        def before():
            return old._prefill_with_cached_prefix(*args)

        def after():
            return infer._prefill_with_cached_prefix(*args)

        torch.testing.assert_close(after(), before(), atol=0.02, rtol=0.02)
        for _ in range(3):
            before()
            after()
        timings = {"before_ms": [], "after_ms": []}
        for repeat in range(20):
            pairs = [("before_ms", before), ("after_ms", after)]
            for key, fn in pairs if repeat % 2 == 0 else pairs[::-1]:
                timings[key].append(elapsed(fn)[1])
        medians = {key: statistics.median(value) for key, value in timings.items()}
        results.append(
            {
                "batch": batch,
                "heads": heads,
                "head_dim": head_dim,
                "prefix": prefix,
                "max_chunk": chunk,
                **medians,
                "speedup": medians["before_ms"] / medians["after_ms"],
                "samples": timings,
            }
        )
    return results


@torch.inference_mode()
def model_validation(model_path, baseline, device):
    config = config_from_hf(model_path)
    engine = EngineConfig(
        config,
        model_path=str(model_path),
        role="rollout",
        devices=[0],
        runtime=RuntimeConfig(compile_model=False, eager_decode=True),
    )
    model = build_model_on_device(engine, device)
    load_model_weights(model, config, model_path)
    model.eval()
    model.prepare_infer_weights()
    caches = model.allocate_kv_caches(16, 256, device)
    model.set_kv_caches(caches, num_slots=3)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    tokens = tokenizer.encode(
        "Read the facts carefully. Alice has three apples and Bob has five apples. " * 30, add_special_tokens=False
    )[:257]
    assert len(tokens) > 128

    def forward(ids, positions, meta, logit_index=-1):
        kwargs = {"input_ids": ids, "position_ids": positions, "infer_meta": meta}
        out = model(**kwargs, defer_lm_head=True)
        hidden = out.hidden_states[:, [logit_index]].to(dtype=model.lm_head.weight.dtype)
        return model.lm_head(hidden)[0, -1].float()

    def run(budget, companion=False):
        model.reset_kv_caches()
        state = InferenceBatchState(
            [tokens, tokens[:255]] if companion else [tokens],
            16,
            max_running_seqs=2,
            max_prefill_tokens=budget,
            max_cache_len=1024,
            kv_block_size=256,
            num_cache_blocks=15,
        )
        while state.has_pending_prompts:
            payload = state.build_prefill_payload()
            meta = payload_to_infer_meta(payload, device)
            logits = forward(
                payload["input_ids"].to(device)[None],
                payload["position_ids"].to(device)[None],
                meta,
                logit_index=len(tokens) - 1 if companion else -1,
            )
        final_logits = logits.cpu()
        fixed_logprobs, greedy, decode_logits = [], [], []
        for offset in range(4):
            greedy.append(int(logits.argmax()))
            token = tokens[offset]
            fixed_logprobs.append(float(torch.log_softmax(logits, -1)[token]))
            meta = InferMeta(
                mode="decode",
                cache_seqlens=torch.tensor([len(tokens) + offset], dtype=torch.int32, device=device),
                block_table=torch.tensor([state._seq_to_blocks[0] + [0]], dtype=torch.int32, device=device),
                recurrent_slots=torch.tensor([state._seq_to_recurrent_slot[0]], device=device),
            )
            logits = forward(
                torch.tensor([[token]], device=device), torch.tensor([[len(tokens) + offset]], device=device), meta
            )
            decode_logits.append(logits.cpu())
        return {
            "final_logits": final_logits,
            "logprobs": torch.tensor(fixed_logprobs),
            "greedy": greedy,
            "decode_logits": torch.stack(decode_logits),
        }

    reference = run(1024)
    # A separate full-prefill batch changes GEMM shapes without splitting the
    # target prompt. Record this sensitivity; it does not establish a tolerance
    # or excuse a missing-prefix / cross-sequence attention error.
    batched = run(1024, companion=True)
    batch_shape_baseline = {
        "max_final_logit_error": float((batched["final_logits"] - reference["final_logits"]).abs().max()),
        "max_decode_logit_error": float((batched["decode_logits"] - reference["decode_logits"]).abs().max()),
        "max_logprob_error": float((batched["logprobs"] - reference["logprobs"]).abs().max()),
        "greedy_equal": batched["greedy"] == reference["greedy"],
    }
    print(json.dumps({"batch_shape_baseline": batch_shape_baseline}), flush=True)
    baseline_error = None
    if config.model_type == "bailing_moe_v3":
        from areno.models.bailing_v3.model import BailingKDAAttention

        old_model = baseline_module(baseline, Path("areno/models/bailing_v3/model.py"), "baseline_bailing_v3")
        original_conv = BailingKDAAttention._causal_conv_infer_prefill

        def old_conv(self, x, weight, cache_idx, cu, slots, cached_prefix=None):
            return old_model.BailingKDAAttention._causal_conv_infer_prefill(self, x, weight, cache_idx, cu, slots)

        try:
            BailingKDAAttention._causal_conv_infer_prefill = old_conv
            broken = run(128)
            baseline_error = float((broken["final_logits"] - reference["final_logits"]).abs().max())
            print(json.dumps({"baseline_max_final_logit_error": baseline_error}), flush=True)
        finally:
            BailingKDAAttention._causal_conv_infer_prefill = original_conv
    results = []
    # BF16 recurrent kernels accumulate in different partitions. Tolerances are
    # fixed before running; record actual maximum errors and greedy agreement.
    for budget in (128, 65, 1):
        actual, ms = elapsed(lambda: run(budget))
        row = {
            "budget": budget,
            "time_ms": ms,
            "max_final_logit_error": float((actual["final_logits"] - reference["final_logits"]).abs().max()),
            "max_decode_logit_error": float((actual["decode_logits"] - reference["decode_logits"]).abs().max()),
            "max_logprob_error": float((actual["logprobs"] - reference["logprobs"]).abs().max()),
            "greedy_equal": actual["greedy"] == reference["greedy"],
        }
        row["checks"] = {}
        for key, atol, rtol in (("final_logits", 0.15, 0.02), ("decode_logits", 0.15, 0.02), ("logprobs", 0.05, 0.005)):
            try:
                torch.testing.assert_close(actual[key], reference[key], atol=atol, rtol=rtol)
                row["checks"][key] = True
            except AssertionError:
                row["checks"][key] = False
        row["passed"] = all(row["checks"].values()) and row["greedy_equal"]
        results.append(row)
        print(json.dumps({"model_partition": row}), flush=True)
    return model, {
        "model_type": config.model_type,
        "dtype": str(config.dtype),
        "prompt_tokens": len(tokens),
        "partitions": results,
        "batch_shape_baseline": batch_shape_baseline,
        "passed": all(row["passed"] for row in results),
        "baseline_max_final_logit_error": baseline_error,
        "logit_atol": 0.15,
        "logit_rtol": 0.02,
        "logprob_atol": 0.05,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-only", action="store_true", help="Skip the optional helper microbenchmark")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "requires real CUDA hardware"
    device = torch.device("cuda", 0)
    result = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "model": str(args.model),
        "baseline": str(args.baseline),
        "eager_model": True,
        "production_checkpoint_modified": False,
    }
    if not args.model_only:
        old = baseline_module(args.baseline, Path("areno/engine/layers/attention_backend/infer.py"), "baseline_infer")
        config = config_from_hf(args.model)
        result["attention"] = attention_benchmark(old, device, config.num_attention_heads, config.head_dim)
        print(json.dumps({"attention": result["attention"]}), flush=True)
    args.output.write_text(json.dumps(result, indent=2))
    _, result["model_correctness"] = model_validation(args.model, args.baseline, device)
    result["passed"] = result["model_correctness"]["passed"]
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"passed": result["passed"], "output": str(args.output)}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
