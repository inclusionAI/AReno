"""Gradient error of a saved compression plan on held-out SFT batches.

Loads a checkpoint written by a compressed run (weights plus
``activation_compression.json``), packs held-out rows exactly like the SFT
trainer and, for every batch, compares the parameter gradients computed with
the plan against the uncompressed gradients, both under FlashAttention's
deterministic backward. The model runs through AReno's own layers, compiled as
in training, on one GPU. Per batch the report holds the relative L2 error, the
deterministic noise floor (two uncompressed passes) and the default backward's
run-to-run difference; ``--plan`` can override the saved plan with a uniform
precision for comparison.

    python examples/activation_compression/gradient_error.py --ckpt runs/sft/actc_s1/step_000300 \\
        --dataset-path $ALPACA --batches 8 --mini-bs 64 --output error.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory holding activation_compression.json.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-loader-fn", default="examples/sft/alpaca/dataset_loader.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--mini-bs", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=1234, help="Same split as sft_validation.py.")
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--plan", choices=["saved", "8", "4", "2"], default="saved")
    return parser.parse_args()


def held_out_batches(args, tokenizer):
    """Validation rows of sft_validation.py, packed like the SFT trainer's microbatches."""

    import torch
    from datasets import load_dataset, load_from_disk

    from areno.api.backend.cuda.training import annotate_sft_token_mean_packs, make_train_pack, sft_target_token_count
    from areno.api.trainers.sft import _record_to_train_sequence
    from areno.cli.train import _load_dataset_for_training

    rows = _load_dataset_for_training(
        args.dataset_path,
        dataset_loader_fn=args.dataset_loader_fn,
        load_dataset=load_dataset,
        load_from_disk=load_from_disk,
    )
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    sequences = []
    for index in order:
        sequence = _record_to_train_sequence(
            dict(rows[index]),
            tokenizer,
            None,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        if sequence is not None:
            sequences.append(sequence)
        if len(sequences) == args.batches * args.mini_bs:
            break
    packs = [make_train_pack(sequences[i : i + args.mini_bs]) for i in range(0, len(sequences), args.mini_bs)]
    counts = [sft_target_token_count(sequences[i : i + args.mini_bs]) for i in range(0, len(sequences), args.mini_bs)]
    for pack, count in zip(packs, counts, strict=True):
        annotate_sft_token_mean_packs([pack], [count], gradient_accumulation_steps=None)
    return packs


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoTokenizer

    from areno.api.backend.cuda.losses import sft_loss_fn
    from areno.engine.data import to_device
    from areno.engine.layers.attention_backend.train import deterministic_flash_backward
    from areno.engine.parallel.context import TPContext, set_tp_context
    from areno.engine.runtime.activation_compression import PLAN_FILE, ActivationCompressor, relative_error
    from areno.engine.runtime.logprobs import packed_next_token_logprobs_from_hidden
    from areno.engine.runtime.train_step import _pack_train_data, _train_meta
    from areno.models.registry import build_model, config_from_hf, load_model_weights

    device = torch.device("cuda", 0)
    set_tp_context(TPContext(0, 1, device, None))
    torch.set_float32_matmul_precision("high")  # as in training workers
    config = config_from_hf(args.ckpt)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(config.dtype)
        with torch.device(device):
            model = build_model(config)
    finally:
        torch.set_default_dtype(previous)
    load_model_weights(model, config, args.ckpt)
    model.train()
    state = json.loads((Path(args.ckpt) / PLAN_FILE).read_text())
    compressor = ActivationCompressor(model, error_budget=state["error_budget"], bits=state["bits"])
    if not compressor.load_state_dict(state):
        raise ValueError(f"{args.ckpt} does not hold a compatible compression plan")
    if args.plan != "saved":
        compressor.plan = [int(args.plan)] * len(compressor.plan)
    train_model = model if args.no_compile else torch.compile(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    def gradients(pack, plan):
        for parameter in parameters:
            parameter.grad = None
        data = to_device(_pack_train_data(dict(pack)), device)
        data["_activation_checkpointing_enabled"] = not args.no_activation_checkpointing
        data["_mst_chunk_size"] = 256
        tokens = data["input_ids"].long()
        meta = _train_meta(data, tokens, sequence_parallel=False)
        with compressor.capture(plan):
            out = train_model(
                input_ids=tokens, position_ids=data.get("position_ids"), train_meta=meta, defer_lm_head=True
            )
        logprobs = packed_next_token_logprobs_from_hidden(
            out.hidden_states, tokens, data["train_cu_seqlens"], model.lm_head, chunk_size=256
        )
        loss, _ = sft_loss_fn(data, logprobs)
        loss.backward()
        return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]

    packs = held_out_batches(args, AutoTokenizer.from_pretrained(args.ckpt))
    for pack in packs:  # settle compiled graphs for every shape first
        gradients(pack, None)
    results = []
    for index, pack in enumerate(packs):
        reference = gradients(pack, None)
        nondeterminism = relative_error(gradients(pack, None), reference)
        with deterministic_flash_backward():
            reference = gradients(pack, None)
            floor = relative_error(gradients(pack, None), reference)
            error = relative_error(gradients(pack, "current"), reference)
        stats = compressor.last_capture.stats
        results.append(
            {
                "batch": index,
                "tokens": int(pack["lengths"].sum()),
                "relative_error": error,
                "noise_floor": floor,
                "baseline_nondeterminism": nondeterminism,
                "stored_fraction": stats.stored_bytes / max(stats.saved_bytes, 1),
                "plan_misses": stats.plan_misses,
            }
        )
        print(json.dumps(results[-1]), flush=True)
    errors = [row["relative_error"] for row in results]
    summary = {
        "ckpt": args.ckpt,
        "plan": args.plan,
        "error_budget": compressor.error_budget,
        "batches": results,
        "max_relative_error": max(errors),
        "mean_relative_error": sum(errors) / len(errors),
        "rms_relative_error": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "within_budget": sum(value <= compressor.error_budget for value in errors),
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "batches"}))


if __name__ == "__main__":
    main()
