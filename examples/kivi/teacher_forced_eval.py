"""Teacher-forced long-context quality of KIVI caches against the model-dtype cache.

Loads the model in-process and decodes the *same* ground-truth tokens one by
one through several KV caches in lockstep: the model-dtype paged cache (the
reference) and KIVI caches. Every step feeds the true token, so the runs never
diverge and differences isolate the KV-cache error at each absolute position.
The first ``--prefill`` tokens go through prefill; every later position goes
through the decode path that rollout uses.

Reports, per absolute-position bucket and cache: perplexity of the true next
token, KL(reference || candidate) of the next-token distributions and top-1
agreement with the reference.

    python examples/kivi/teacher_forced_eval.py --model /path/to/Qwen3-0.6B \
        --text /path/to/wikitext.txt --length 32768 --rows 4 --output tf.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

BLOCK_SIZE = 256
BUCKET_EDGES = (1024, 2048, 4096, 8192, 16384, 32768, 65536)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True, help="UTF-8 text file; windows are cut from its token stream.")
    parser.add_argument("--length", type=int, default=8192, help="Tokens per sequence (prefill + decode).")
    parser.add_argument("--rows", type=int, default=4, help="Sequences decoded in parallel.")
    parser.add_argument("--prefill", type=int, default=256)
    parser.add_argument("--kv-cache-dtypes", nargs="+", default=["kivi2", "kivi4"])
    parser.add_argument("--kivi-group-size", type=int, default=32)
    parser.add_argument("--kivi-residual-length", type=int, default=128)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _windows(tokenizer, text_path: str, rows: int, length: int) -> torch.Tensor:
    ids = tokenizer(Path(text_path).read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
    if len(ids) < rows * length:
        # Repeat the corpus when it is shorter than the requested windows.
        ids = ids * math.ceil(rows * length / len(ids))
    stride = (len(ids) - length) // max(1, rows - 1) if rows > 1 else 0
    return torch.tensor([ids[row * stride : row * stride + length] for row in range(rows)], dtype=torch.long)


def _bucket(position: int) -> str:
    for edge in BUCKET_EDGES:
        if position < edge:
            return f"<{edge}"
    return f">={BUCKET_EDGES[-1]}"


class CacheSet:
    """One KV-cache configuration bound to the shared model before each forward."""

    def __init__(self, model, name: str, args, rows: int, device: torch.device):
        from areno.engine.layers.attention_backend.kivi import KIVI_BITS, KiviKVCache, KiviSpec

        self.name = name
        self.model = model
        blocks_per_row = -(-args.length // BLOCK_SIZE)
        self.block_table = torch.arange(rows * blocks_per_row, device=device, dtype=torch.int32).view(rows, -1)
        if name == "auto":
            self.caches = model.allocate_kv_caches(rows * blocks_per_row, BLOCK_SIZE, device)
        else:
            spec = KiviSpec(
                bits=KIVI_BITS[name], group_size=args.kivi_group_size, residual_length=args.kivi_residual_length
            )
            self.caches = [
                KiviKVCache.allocate(
                    spec,
                    num_blocks=rows * blocks_per_row,
                    block_size=BLOCK_SIZE,
                    num_slots=rows,
                    num_kv_heads=layer.self_attn.local_kv_heads,
                    head_dim=layer.self_attn.head_dim,
                    dtype=model.config.dtype,
                    device=device,
                )
                for layer in model.layers
            ]

    def bind(self) -> None:
        for layer, cache in zip(self.model.layers, self.caches, strict=True):
            layer.self_attn.clear_kv_cache()
            if self.name == "auto":
                layer.self_attn.set_kv_cache(*cache)
            else:
                layer.self_attn.set_kivi_cache(cache)

    def nbytes(self) -> int:
        if self.name == "auto":
            return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in self.caches)
        return sum(cache.nbytes() for cache in self.caches)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    from areno.engine.modeling import skip_torch_init
    from areno.engine.runtime.metadata import InferMeta
    from areno.models.registry import build_model, config_from_hf, load_model_weights

    device = torch.device("cuda", torch.cuda.current_device())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokens = _windows(tokenizer, args.text, args.rows, args.length).to(device)
    config = config_from_hf(args.model)
    # Same construction as the engine worker: parameters in the model dtype,
    # allocated on the device, with initializers skipped before loading.
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.dtype)
    try:
        with torch.device(device), skip_torch_init(enabled=True):
            model = build_model(config)
    finally:
        torch.set_default_dtype(default_dtype)
    load_model_weights(model, config, args.model)
    model.eval()

    names = ["auto", *args.kv_cache_dtypes]
    cache_sets = [CacheSet(model, name, args, args.rows, device) for name in names]
    rows, prefill = args.rows, args.prefill
    slots = torch.arange(rows, device=device)
    positions = torch.arange(prefill, device=device)
    block_ids = torch.cat([cache_sets[0].block_table[row, positions // BLOCK_SIZE] for row in range(rows)]).long()
    block_offsets = (positions % BLOCK_SIZE).repeat(rows)
    prefill_meta = InferMeta(
        mode="prefill",
        cu_seqlens=torch.arange(0, rows * prefill + 1, prefill, device=device, dtype=torch.int32),
        max_seqlen=prefill,
        block_table=cache_sets[0].block_table,
        cache_block_ids=block_ids,
        cache_block_offsets=block_offsets,
        recurrent_slots=slots,
    )
    prefill_ids = tokens[:, :prefill].reshape(1, -1)
    prefill_pos = positions.repeat(rows).view(1, -1)
    for cache_set in cache_sets:
        cache_set.bind()
        model(prefill_ids, prefill_pos, infer_meta=prefill_meta)

    stats = {name: {} for name in names}
    start = time.perf_counter()
    for position in range(prefill, args.length - 1):
        meta = InferMeta(
            mode="decode",
            cache_seqlens=torch.full((rows,), position, device=device, dtype=torch.int32),
            block_table=cache_sets[0].block_table,
            recurrent_slots=slots,
        )
        step_ids = tokens[:, position].view(1, rows)
        step_pos = torch.full((1, rows), position, device=device, dtype=torch.long)
        target = tokens[:, position + 1]
        logprobs = []
        for cache_set in cache_sets:
            cache_set.bind()
            logits = model(step_ids, step_pos, infer_meta=meta).logits_shard.view(rows, -1).float()
            logprobs.append(torch.log_softmax(logits, dim=-1))
        reference = logprobs[0]
        bucket = _bucket(position + 1)
        for name, logprob in zip(names, logprobs, strict=True):
            entry = stats[name].setdefault(bucket, {"tokens": 0, "nll": 0.0, "kl": 0.0, "top1_agree": 0.0})
            entry["tokens"] += rows
            entry["nll"] += float(-logprob.gather(1, target[:, None]).sum())
            entry["kl"] += float((reference.exp() * (reference - logprob)).sum())
            entry["top1_agree"] += float((logprob.argmax(-1) == reference.argmax(-1)).sum())
    elapsed = time.perf_counter() - start

    report = {
        "model": args.model,
        "text": args.text,
        "rows": rows,
        "length": args.length,
        "prefill": prefill,
        "kivi_group_size": args.kivi_group_size,
        "kivi_residual_length": args.kivi_residual_length,
        "cache_bytes": {cache_set.name: cache_set.nbytes() for cache_set in cache_sets},
        "decode_seconds": elapsed,
        "buckets": {},
    }
    for name in names:
        for bucket, entry in stats[name].items():
            count = entry["tokens"]
            report["buckets"].setdefault(bucket, {})[name] = {
                "tokens": count,
                "ppl": math.exp(entry["nll"] / count),
                "kl_vs_auto": entry["kl"] / count,
                "top1_agreement_vs_auto": entry["top1_agree"] / count,
            }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
