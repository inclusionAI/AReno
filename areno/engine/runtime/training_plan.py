"""Record the resolved Ling training schedule from the executing worker."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from collections import Counter
from pathlib import Path

import torch

from areno.engine.modeling import unwrap_model
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.recompute_policy import RecomputePolicy, effective_recompute_plan

logger = logging.getLogger(__name__)


def record_training_recompute_plan(
    worker, model, meta, pack, *, defer_lm_head: bool, output_chunk_size: int, output_recompute: bool
) -> dict | None:
    """Persist once per worker after actual topology, modules and chunks resolve."""
    if worker.config.model.model_type != "bailing_moe_v3":
        return None
    model = unwrap_model(model)
    available = []
    for layer in model.layers:
        names = ["softmax_attention" if layer.attention_layer_type == "attention" else "kda_attention"]
        if hasattr(layer.mlp, "route"):
            names.append("routed_experts")
            if layer.mlp.shared_experts is not None:
                names.append("shared_experts")
        else:
            names.append("dense_mlp")
        available.append(names)
    plan = effective_recompute_plan(
        activation_checkpointing=meta.activation_checkpointing,
        policy=meta.recompute_policy or RecomputePolicy(),
        layer_modules=available,
        output_recompute=bool(defer_lm_head and output_recompute),
        output_chunk_size=output_chunk_size,
        mst_chunk_size=meta.mst_chunk_size,
    )
    plan["sft_output"]["path"] = "chunked_hidden" if defer_lm_head else "model_logits"
    plan["sft_output"]["chunk_size"] = output_chunk_size if defer_lm_head else None
    runtime = worker.config.runtime
    ctx = get_tp_context()
    if (
        ctx.device.type in ("cuda", "cpu")
        and ctx.world_size == 1
        and not meta.sequence_parallel
        and any(getattr(layer, "mst_consistent_tiles", False) for layer in model.layers)
    ):
        plan["decoder"]["compute_tiling"] = {
            "chunk_size": meta.mst_chunk_size or 256,
            "applies_to": "dense_mlp,routed_experts,shared_experts, including direct execution",
            "checkpoint_scope_when_selected": "per_chunk" if meta.mst_chunk_size else "whole_tiled_module",
            "low_precision_parameter_reduction": "FP32 accumulation of partials, then one final storage-dtype cast",
        }
    source = Path(__file__).resolve().parents[3]
    commit_file = source / "SOURCE_COMMIT"
    try:
        commit = (
            commit_file.read_text().strip()
            if commit_file.exists()
            else subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=3
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit = None
    mask = pack.get("packed_response_mask")
    digest = hashlib.sha256()
    for value in (pack["input_ids"], pack["train_cu_seqlens"], mask):
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    plan.update(
        schema_version=1,
        requested={
            "activation_checkpointing": runtime.activation_checkpointing,
            "recompute_modules": runtime.recompute_modules,
            "recompute_layers": runtime.recompute_layers,
            "sft_output_recompute": runtime.sft_output_recompute,
        },
        provenance={
            "code_commit": commit,
            "model_path": worker.config.model_path,
            "dtype": str(next(model.parameters()).dtype),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "torch_initial_seed": torch.initial_seed(),
            "lora_seed": worker.config.lora_seed,
        },
        topology={
            "global_rank": ctx.global_rank,
            "tp_size": ctx.world_size,
            "dp_size": ctx.dp_size,
            "sequence_parallel": meta.sequence_parallel,
        },
        first_batch={
            "input_tokens_per_dp_rank": pack["input_ids"].numel(),
            "supervised_tokens_per_dp_rank": int(mask.sum()) if isinstance(mask, torch.Tensor) else None,
            "packed_sequences": pack["train_cu_seqlens"].numel() - 1,
            "input_sha256": digest.hexdigest(),
        },
    )
    directory = getattr(runtime, "recompute_plan_dir", None)
    path = None
    if directory:
        path = Path(directory) / f"recompute-plan.{os.getpid()}.rank{ctx.global_rank}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if ctx.rank == 0 and ctx.dp_rank == 0:
        counts = Counter(strategy for layer in plan["decoder"]["layers"] for strategy in layer["modules"].values())
        logger.info("Recompute plan: decoder=%s SFT_output=%s saved=%s", dict(counts), plan["sft_output"], path)
    return plan
