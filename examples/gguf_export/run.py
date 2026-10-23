"""One real SFT update, native generation, and direct GGUF export.

Run from the source checkout with its matching installed AReno extension.
Models and outputs are explicit local paths; no downloads are performed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--file-type", choices=("f16", "q8_0"), default="f16")
    parser.add_argument("--lora-rank", type=int, default=0)
    args = parser.parse_args()

    import numpy as np
    import torch

    from areno import Trainer
    from areno.api import CudaConfig, LoraConfig, SamplingParams, TrainSequence, sft_loss_fn

    torch.manual_seed(123)
    trainer = Trainer(
        world_size=1,
        model_path=args.model_path,
        custom_config=CudaConfig(
            tp_size=1,
            max_running_prompts=1,
            lora=LoraConfig(rank=args.lora_rank, alpha=float(args.lora_rank * 2)) if args.lora_rank else None,
            optimizer={"lr": 1e-4, "grad_clip_norm": 1.0},
            runtime={
                "compile_model": False,
                "activation_checkpointing": False,
                "attn_backend": "native",
                "eager_decode": True,
                "keep_rollout_state": False,
            },
        ),
    )
    report = {"model_path": args.model_path, "file_type": args.file_type, "lora_rank": args.lora_rank, "steps": []}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def record(event, **values):
        print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    trainer.init()
    try:
        tokenizer = trainer.get_tokenizer()
        probe_messages = [{"role": "user", "content": "What is 1 + 1? Answer briefly."}]
        probe = tokenizer.apply_chat_template(
            probe_messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        report["probe_prompt"] = tokenizer.decode(probe, skip_special_tokens=False)
        sampling = SamplingParams(max_new_tokens=32, greedy=True, max_prompt_len=256, max_context_len=512)

        async def generate():
            async with trainer.rollout_session(sampling_params=sampling, proxy=False):
                sequence = trainer.rollout_token_batch([probe], n_samples=1, sampling_params=sampling)[0].sequences[0]
            return {
                "text": tokenizer.decode(sequence.resp_tokens, skip_special_tokens=True),
                "tokens": sequence.resp_tokens,
                "logprobs": sequence.resp_logprobs,
            }

        report["before"] = asyncio.run(generate())
        record("native_before", **report["before"])
        rows = [
            ("What is 2 + 3? Answer with the number only.", "5"),
            ("What is 4 + 4? Answer with the number only.", "8"),
        ]
        batch = []
        for question, answer in rows:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            response = tokenizer.encode(answer + tokenizer.eos_token, add_special_tokens=False)
            batch.append(
                TrainSequence(tokens=prompt + response, prompt_len=len(prompt), eos_token_id=tokenizer.eos_token_id)
            )
        record("dataset_inspected", ok=True, records=len(batch), token_lengths=[len(row.tokens) for row in batch])
        # Both examples form one minibatch and exactly one optimizer update.
        metrics = trainer.train(batch, sft_loss_fn, mini_bs=len(batch))
        if not math.isfinite(float(metrics["loss"])):
            raise RuntimeError("non-finite SFT loss")
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError(f"non-finite train metric {key}: {value}")
        report["steps"].append(metrics)
        record("train_step", step=1, metrics=metrics)
        report["after"] = asyncio.run(generate())
        record("native_after", **report["after"])
        report["gguf_path"] = trainer.export_gguf(args.output_dir, file_type=args.file_type)
        record("gguf_exported", path=report["gguf_path"])
    finally:
        trainer.close()

    import gguf
    from gguf.quants import dequantize
    from safetensors import safe_open

    reader = gguf.GGUFReader(report["gguf_path"])
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    index_file = Path(args.model_path) / "model.safetensors.index.json"
    weights = json.loads(index_file.read_text())["weight_map"] if index_file.exists() else None
    checks = []
    for name, exported_name in (
        ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
        ("model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
    ):
        source_file = (
            Path(args.model_path) / weights[name] if weights else next(Path(args.model_path).glob("*.safetensors"))
        )
        with safe_open(source_file, framework="pt", device="cpu") as source:
            original = source.get_tensor(name).float().numpy()
        tensor = tensors[exported_name]
        exported = dequantize(tensor.data, tensor.tensor_type)
        rounded_original = original.astype(np.float16).astype(np.float32)
        difference = np.abs(exported - rounded_original)
        checks.append(
            {
                "tensor": exported_name,
                "changed_elements": int(np.count_nonzero(difference)),
                "max_delta": float(difference.max()),
            }
        )
    if args.file_type == "f16" and not any(check["changed_elements"] for check in checks):
        raise RuntimeError("GGUF contains no observed training update in the checked projections")
    report["weight_update_checks"] = checks
    report["tensor_count"] = len(tensors)
    report["status"] = "native_training_and_export_passed"
    record("weight_updates_verified", checks=checks, tensor_count=len(tensors))


if __name__ == "__main__":
    main()
