"""Measure checkpoint-backed training forward memory, without a backward pass.

Run the same script in separate baseline/candidate processes with PYTHONPATH
pointing at the checkout being measured. Model load and warmup are excluded.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--row-tokens", type=int, default=32768)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.row_tokens, args.rows, args.chunk_size, args.repeats) < 1:
        parser.error("token lengths, rows, chunk size and repeats must be positive")

    import areno
    from areno import _configure_torch_runtime
    from areno.adapters.config import LoraConfig
    from areno.adapters.lora import initialize_lora
    from areno.adapters.qlora import initialize_qlora
    from areno.engine.config import EngineConfig, OptimizerConfig, RuntimeConfig
    from areno.engine.modeling import build_model_on_device
    from areno.engine.parallel.context import TPContext, set_tp_context
    from areno.engine.runtime.device import configure_unified_memory_budget
    from areno.engine.runtime.logprobs import packed_next_token_logprobs_from_hidden
    from areno.engine.runtime.metadata import TrainMeta
    from areno.models.registry import config_from_hf, load_model_weights

    _configure_torch_runtime()
    torch.manual_seed(23)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_tp_context(TPContext(0, 1, device, None))
    configure_unified_memory_budget(device)
    checkpoint = str(args.checkpoint)
    config = EngineConfig(
        model=config_from_hf(checkpoint),
        model_path=checkpoint,
        lora=LoraConfig(rank=8, alpha=16, qlora=True),
        runtime=RuntimeConfig(compile_model=False, activation_checkpointing=True, keep_rollout_state=False),
        optimizer=OptimizerConfig(adam_4bit=True),
        devices=[0],
    )
    model = build_model_on_device(config, device)
    load_model_weights(model, config.model, checkpoint)
    initialize_lora(model, config.lora, seed=0)
    initialize_qlora(model)
    model.train()
    tokens = torch.randint(config.model.vocab_size, (1, args.rows * args.row_tokens), device=device)
    positions = torch.arange(args.row_tokens, device=device).repeat(args.rows).unsqueeze(0)
    cu = torch.arange(args.rows + 1, device=device, dtype=torch.int32) * args.row_tokens
    meta = TrainMeta(
        cu_seqlens=cu,
        max_seqlen=args.row_tokens,
        packed=True,
        activation_checkpointing=True,
        mst_chunk_size=args.chunk_size,
    )
    checkout = Path(areno.__file__).resolve().parents[1]
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
        "checkpoint": checkpoint,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(config.model.dtype),
        "qlora_rank": 8,
        "rows": args.rows,
        "row_tokens": args.row_tokens,
        "packed_tokens": tokens.numel(),
        "chunk_size": args.chunk_size,
        "lm_head_chunk_size": 256,
        "compile_model": False,
        "activation_checkpointing": True,
        "warmup_runs": 1,
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats + 1):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output = model(tokens, position_ids=positions, train_meta=meta, defer_lm_head=True)
        decoder_peak = torch.cuda.max_memory_allocated()
        logprobs = packed_next_token_logprobs_from_hidden(
            output.hidden_states, tokens, cu, model.lm_head, chunk_size=256
        )
        torch.cuda.synchronize()
        result = {
            "resident_bytes": resident,
            "decoder_peak_allocated_bytes": decoder_peak,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "seconds": time.perf_counter() - started,
            "finite": bool(torch.isfinite(logprobs).all()),
        }
        del logprobs, output
        if repeat:
            report["runs"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        assert result["finite"], "forward logprobs must be finite"


if __name__ == "__main__":
    main()
