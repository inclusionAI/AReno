"""Compare a full model's LoRA/QLoRA training memory in separate processes.

Includes managed optimizer bytes explicitly. This is a packed training-step
benchmark; rollout caches and loading-time peaks are not part of this metric.
"""

import argparse
import json
import time

import torch

from areno.adapters import LoraConfig
from areno.adapters.lora import initialize_lora
from areno.adapters.qlora import initialize_qlora
from areno.engine.config import EngineConfig, OptimizerConfig
from areno.engine.modeling import build_model_on_device, build_optimizer
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.logprobs import packed_next_token_logprobs
from areno.engine.runtime.metadata import TrainMeta
from areno.models.registry import config_from_hf, load_model_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local checkpoint path")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--adam-4bit", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(2026)
    model_config = config_from_hf(args.model)
    engine = EngineConfig(model=model_config, model_path=args.model, devices=[0])
    model = build_model_on_device(engine, torch.device("cuda", 0))
    load_model_weights(model, model_config, args.model)
    registry = initialize_lora(model, LoraConfig(qlora=args.qlora), seed=0)
    quantization = initialize_qlora(model) if args.qlora else None
    optimizer = build_optimizer(
        registry.parameters(),
        OptimizerConfig(
            lr=1e-5,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            adam_4bit=args.adam_4bit,
            paged=args.qlora,
        ),
        get_tp_context(),
    )
    length, batch = args.sequence_length, args.microbatch
    tokens = torch.randint(model_config.vocab_size, (1, batch * length), device="cuda")
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * length
    positions = torch.arange(length, device="cuda").repeat(batch)[None]
    meta = TrainMeta(cu_seqlens=cu, max_seqlen=length, packed=True, activation_checkpointing=True)
    results = []
    for step in range(3):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        logits = model(tokens, position_ids=positions, train_meta=meta).logits_shard
        loss = -packed_next_token_logprobs(logits, tokens, cu).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del logits
        torch.cuda.synchronize()
        managed = getattr(optimizer, "managed_memory_bytes", lambda: 0)()
        results.append(
            {
                "step": step,
                "loss": loss.item(),
                "seconds": time.perf_counter() - start,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "managed_optimizer_bytes": managed,
                "peak_allocated_plus_managed_bytes": torch.cuda.max_memory_allocated() + managed,
            }
        )
        del loss
    report = {
        "config": vars(args),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "quantization": quantization,
        "steps": results,
    }
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
