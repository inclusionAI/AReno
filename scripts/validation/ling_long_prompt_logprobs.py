"""Compare fixed-continuation logprobs for full/chunked rollout and train forward.

Run with torchrun for vocab-sharded tensor parallelism. This performs no
optimizer updates. Mean absolute differences cover continuation tokens only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist

from areno.engine.config import EngineConfig, RuntimeConfig
from areno.engine.data.rollout_state import InferenceBatchState, payload_to_infer_meta
from areno.engine.modeling import build_model_on_device
from areno.engine.parallel.context import TPContext, set_tp_context
from areno.engine.runtime.logprobs import vocab_parallel_selected_logprobs
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.models.registry import config_from_hf, load_model_weights


def compare(a, b):
    difference = (a.float() - b.float()).abs()
    return {
        "tokens": difference.numel(),
        "abs_diff_mean": float(difference.mean()),
        "abs_diff_p50": float(difference.quantile(0.5)),
        "abs_diff_p95": float(difference.quantile(0.95)),
        "abs_diff_max": float(difference.max()),
        "signed_diff_mean": float((a.float() - b.float()).mean()),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4097, 8193])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2048, 512, 128])
    parser.add_argument("--continuation-tokens", type=int, default=128)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", 0))
    size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if size > 1:
        dist.init_process_group("nccl")
    set_tp_context(
        TPContext(
            rank=rank,
            world_size=size,
            device=device,
            group=dist.group.WORLD if size > 1 else None,
            global_rank=rank,
            global_world_size=size,
            role="rollout",
            train_tp_size=size,
        )
    )
    config = config_from_hf(args.model)
    model = build_model_on_device(
        EngineConfig(
            config,
            model_path=str(args.model),
            devices=list(range(size)),
            tp_size=size,
            role="rollout",
            runtime=RuntimeConfig(compile_model=False, eager_decode=True),
        ),
        device,
    )
    load_model_weights(model, config, args.model)
    model.eval()
    model.prepare_infer_weights()
    cache_length = max(args.lengths) + args.continuation_tokens + 256
    blocks = (cache_length + 255) // 256 + 1
    model.set_kv_caches(model.allocate_kv_caches(blocks, 256, device), num_slots=2)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    rng = random.Random(635)
    facts = "\n".join(
        f"Record {i}: researcher {rng.choice(['Alice', 'Bob', 'Chen', 'Daria'])} "
        f"has {rng.randrange(1, 200)} samples in room {rng.randrange(1, 100)}. "
        "Keep the records in order and retain the quantities accurately."
        for i in range(max(args.lengths))
    )
    stream = tokenizer.encode("Read the following laboratory records carefully.\n" + facts, add_special_tokens=False)
    continuation = torch.tensor(stream[max(args.lengths) : max(args.lengths) + args.continuation_tokens], device=device)
    assert continuation.numel() == args.continuation_tokens

    def project(hidden):
        return model.lm_head(hidden.to(model.lm_head.weight.dtype)).reshape(-1, model.lm_head.weight.shape[0])

    def rollout(tokens, budget):
        model.eval()
        model.reset_kv_caches()
        state = InferenceBatchState(
            [tokens],
            args.continuation_tokens,
            max_running_seqs=1,
            max_cache_len=cache_length,
            max_prefill_tokens=budget,
            kv_block_size=256,
            num_cache_blocks=blocks - 1,
        )
        while state.has_pending_prompts:
            p = state.build_prefill_payload()
            out = model(
                p["input_ids"].to(device)[None],
                p["position_ids"].to(device)[None],
                infer_meta=payload_to_infer_meta(p, device),
                defer_lm_head=True,
            )
        hidden = out.hidden_states[:, -1:]
        table = torch.tensor(
            [state._seq_to_blocks[0] + [0] * (state.max_blocks_per_seq - len(state._seq_to_blocks[0]))],
            device=device,
            dtype=torch.int32,
        )
        values = []
        for step in range(args.continuation_tokens):
            values.append(vocab_parallel_selected_logprobs(project(hidden), continuation[step : step + 1]))
            if step + 1 < args.continuation_tokens:
                position = len(tokens) + step
                state.ensure_decode_blocks([0], [position])
                table.zero_()
                table[0, : len(state._seq_to_blocks[0])] = torch.tensor(
                    state._seq_to_blocks[0], device=device, dtype=torch.int32
                )
                meta = InferMeta(
                    mode="decode",
                    cache_seqlens=torch.tensor([position], device=device, dtype=torch.int32),
                    block_table=table,
                    recurrent_slots=torch.tensor([0], device=device),
                )
                hidden = model(
                    continuation[step : step + 1][None],
                    torch.tensor([[position]], device=device),
                    infer_meta=meta,
                    defer_lm_head=True,
                ).hidden_states
        return torch.cat(values).float().cpu()

    def train_forward(tokens):
        model.train()
        # Packed training uses grouped experts and no inference KV/recurrent cache.
        sequence = torch.cat((torch.tensor(tokens, device=device), continuation))
        count = sequence.numel()
        meta = TrainMeta(
            packed=True,
            cu_seqlens=torch.tensor([0, count], device=device, dtype=torch.int32),
            max_seqlen=count,
            sequence_parallel=False,
            activation_checkpointing=False,
        )
        out = model(sequence[None], torch.arange(count, device=device)[None], train_meta=meta, defer_lm_head=True)
        hidden = out.hidden_states[:, len(tokens) - 1 : len(tokens) - 1 + args.continuation_tokens]
        values = []
        # Match the rollout LM-head shape so this comparison isolates the model
        # forward paths, rather than adding a vocabulary-GEMM shape difference.
        for step in range(args.continuation_tokens):
            values.append(
                vocab_parallel_selected_logprobs(project(hidden[:, step : step + 1]), continuation[step : step + 1])
            )
        model.eval()
        return torch.cat(values).float().cpu()

    result = {
        "model": str(args.model),
        "dtype": str(config.dtype),
        "tp_size": size,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "continuation_tokens": args.continuation_tokens,
        "seed": 635,
        "comparison_axis": "same fixed continuation tokens, natural-log probabilities, temperature 1",
        "train_forward": "packed, model.train(), no gradients, sequence_parallel=False",
        "decode": "eager",
        "measurements": [],
    }
    for length in args.lengths:
        tokens = stream[:length]
        full = rollout(tokens, length)
        repeat = rollout(tokens, length)
        row = {
            "prompt_tokens": length,
            "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            "continuation_sha256": hashlib.sha256(json.dumps(continuation.tolist()).encode()).hexdigest(),
            "full_repeat": compare(full, repeat),
            "full_rollout_logprobs": full.tolist(),
            "chunks": [],
        }
        if rank == 0:
            print(json.dumps({"prompt_tokens": length, "full_repeat": row["full_repeat"]}), flush=True)
        for budget in args.budgets:
            actual = rollout(tokens, budget)
            chunk = {
                "budget": budget,
                "full_vs_chunk": compare(full, actual),
                "chunk_rollout_logprobs": actual.tolist(),
            }
            row["chunks"].append(chunk)
            if rank == 0:
                print(json.dumps({"prompt_tokens": length, "budget": budget, **chunk["full_vs_chunk"]}), flush=True)
        teacher = train_forward(tokens)
        row["train_forward_logprobs"] = teacher.tolist()
        row["full_rollout_vs_train"] = compare(full, teacher)
        for chunk in row["chunks"]:
            chunk["chunk_rollout_vs_train"] = compare(torch.tensor(chunk["chunk_rollout_logprobs"]), teacher)
        result["measurements"].append(row)
        if rank == 0:
            args.output.write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {
                        "prompt_tokens": length,
                        "full_rollout_vs_train": row["full_rollout_vs_train"],
                        "chunk_rollout_vs_train": {c["budget"]: c["chunk_rollout_vs_train"] for c in row["chunks"]},
                    }
                ),
                flush=True,
            )
    if size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
