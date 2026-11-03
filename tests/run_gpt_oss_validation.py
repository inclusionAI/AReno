"""Bounded full GPT-OSS load/reference and two-step SDK training gates."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from areno import Trainer
from areno.api import CudaConfig, SamplingParams, TrainSequence
from areno.api.algorithms import sft_loss_fn
from areno.engine.parallel.collectives import all_gather_last_dim
from areno.engine.parallel.context import TPContext, set_tp_context
from areno.models.gpt_oss import GptOssAdapter
from areno.models.registry import read_hf_config


def sft_with_logprob_diff(data_pack, logprobs):
    loss, metrics = sft_loss_fn(data_pack, logprobs)
    mask = data_pack["packed_response_mask"].to(logprobs.device).bool()
    old = data_pack["packed_logprobs"].to(logprobs.device)
    difference = (logprobs.detach() - old)[mask].abs()
    metrics["mean_abs_logprob_diff"] = difference.mean()
    metrics["max_abs_logprob_diff"] = difference.max()
    return loss, metrics


def reference(args):
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl")
        set_tp_context(
            TPContext(
                rank, world, torch.device("cuda", rank), dist.group.WORLD, global_rank=rank, global_world_size=world
            )
        )
    hf = read_hf_config(args.model)
    if args.fp32:
        hf["torch_dtype"] = "float32"
        hf["dtype"] = "float32"
        torch.backends.cuda.matmul.allow_tf32 = False
    adapter = GptOssAdapter()
    config = adapter.config_from_hf(hf)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(config.dtype)
    try:
        with torch.device("cuda"):
            model = adapter.build(config)
    finally:
        torch.set_default_dtype(previous)
    adapter.load_weights(model, args.model)
    records = []
    for index in (0, 1):
        data = np.load(Path(args.reference) / f"reference-{index}.npz")
        ids = torch.tensor(data["tokens"], device="cuda")
        expected = torch.tensor(data["logits"], device="cuda")
        stage_file = Path(args.reference) / f"layers-{index}.npz"
        handles = []
        stage_records = []
        if stage_file.exists():
            stage_data = np.load(stage_file)

            def capture(name):
                def hook(module, inputs, output):
                    value = output[0] if isinstance(output, tuple) else output
                    reference = torch.tensor(stage_data[name], device=value.device)
                    error = (value.float() - reference).abs()
                    stage_records.append({"name": name, "mean": error.mean().item(), "max": error.max().item()})

                return hook

            for layer_id, layer in enumerate(model.layers):
                for suffix, module in (("attn", layer.self_attn), ("moe", layer.mlp), ("out", layer)):
                    handles.append(module.register_forward_hook(capture(f"{layer_id}:{suffix}")))
        with torch.set_grad_enabled(args.fp32):
            actual = all_gather_last_dim(model(ids).logits_shard).float()
        for handle in handles:
            handle.remove()
        print("NATIVE_STAGES", json.dumps(stage_records), flush=True)
        difference = (actual - expected).abs()
        record = {
            "index": index,
            "mean_logit_abs_diff": difference.mean().item(),
            "max_logit_abs_diff": difference.max().item(),
            "last_token": actual[:, -1].argmax(-1).tolist(),
            "expected_last_token": expected[:, -1].argmax(-1).tolist(),
        }
        records.append(record)
        print("NATIVE_REFERENCE", json.dumps(record), flush=True)
        assert torch.isfinite(actual).all()
        # BF16 eager HF rounds attention scores/activations differently from
        # fused FP32-accumulating attention. Tight FP32 tiny-model gates run first.
        mean_tolerance, max_tolerance = (1e-4, 1e-2) if args.fp32 else (0.05, 0.5)
        assert record["mean_logit_abs_diff"] < mean_tolerance and record["max_logit_abs_diff"] < max_tolerance
        assert record["last_token"] == record["expected_last_token"]
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (Path(args.output) / "reference.json").write_text(json.dumps(records, indent=2))
    if world > 1:
        dist.destroy_process_group()


def training(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        args.tp,
        args.model,
        custom_config=CudaConfig(
            tp_size=args.tp,
            devices=list(range(args.tp)),
            sequence_parallel=False,
            optimizer={"lr": 1e-6, "grad_clip_norm": 1.0, "adam_8bit": args.adam_8bit},
            runtime={"compile_model": False, "activation_checkpointing": True, "decode_graph_buckets": [1, 2]},
        ),
    )
    trainer.init()
    tokenizer = trainer.get_tokenizer()
    try:
        rows = [
            tokenizer.encode(text, add_special_tokens=True)
            for text in ("The capital of France is Paris.", "Two plus two equals four.")
        ]
        batch = [TrainSequence(tokens=row, prompt_mask=[False] * len(row), loss_mask=[True] * len(row)) for row in rows]
        records = []
        for step in range(0 if args.rollout_only else 2):
            loss_fn = sft_loss_fn
            if args.measure_rollout_diff:
                from areno.api.tokenizer import encode_generation_prompt

                prompts = ["What is the capital of France?", "What is 2 + 2?"]
                trainer.begin_rollout_session()
                try:
                    generated = trainer.rollout_batch(
                        prompts, 1, SamplingParams(greedy=True, max_new_tokens=8, ignore_eos=True)
                    )
                finally:
                    trainer.end_rollout_session()
                batch = []
                for prompt, result in zip(prompts, generated, strict=True):
                    prefix = encode_generation_prompt(tokenizer, prompt)
                    sequence = result.sequences[0]
                    batch.append(
                        TrainSequence(
                            tokens=prefix + sequence.resp_tokens,
                            prompt_mask=[True] * len(prefix) + [False] * len(sequence.resp_tokens),
                            loss_mask=[False] * len(prefix) + [True] * len(sequence.resp_tokens),
                            logprobs=[0.0] * len(prefix) + sequence.resp_logprobs,
                        )
                    )
                loss_fn = sft_with_logprob_diff
            metrics = trainer.train(batch, loss_fn, mini_bs=2)
            assert metrics and all(math.isfinite(float(x)) for x in metrics.values() if isinstance(x, float | int))
            records.append(metrics)
            print("TRAIN_STEP", step, json.dumps(metrics), flush=True)
        trainer.begin_rollout_session()
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
                tokenize=False,
                add_generation_prompt=True,
            )
            out = trainer.rollout_batch([prompt], 1, SamplingParams(greedy=True, max_new_tokens=32, ignore_eos=True))
            sequence = out[0].sequences[0]
            print("NATIVE_DECODE", tokenizer.decode(sequence.resp_tokens), flush=True)
            generation = {"tokens": sequence.resp_tokens, "logprobs": sequence.resp_logprobs}
            (output / "generation.json").write_text(json.dumps(generation, indent=2))
            assert all(math.isfinite(value) for value in sequence.resp_logprobs)
            if args.expected_generation:
                expected = json.loads(Path(args.expected_generation).read_text())
                assert generation["tokens"] == expected["tokens"], "reload changed greedy tokens"
                torch.testing.assert_close(
                    torch.tensor(generation["logprobs"]), torch.tensor(expected["logprobs"]), atol=1e-5, rtol=1e-5
                )
                print("RELOAD_GENERATION_OK", flush=True)
        finally:
            trainer.end_rollout_session()
        if not args.rollout_only:
            trainer.save_checkpoint(str(output / "trained"))
        (output / "training.json").write_text(json.dumps(records, indent=2))
        print("TRAIN_SAVE_OK", flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        trainer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--adam-8bit", action="store_true")
    p.add_argument("--rollout-only", action="store_true")
    p.add_argument("--expected-generation")
    p.add_argument("--measure-rollout-diff", action="store_true")
    args = p.parse_args()
    if args.reference:
        reference(args)
    else:
        training(args)


if __name__ == "__main__":
    main()
