"""Portable strict-deferred reproduction; distinct from the frozen study harness.

Run one method per fresh process, with caller-selected CUDA_VISIBLE_DEVICES and
CPU affinity. No checkpoint is saved and the test split is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method", choices=["adamw", "adam8", "strict", "deferred"], required=True)
    p.add_argument("--mode", choices=["performance", "quality"], required=True)
    p.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    p.add_argument("--steps", type=int, choices=[1024, 4096])
    p.add_argument(
        "--fused-scale-writeback", action="store_true", help="opt in to separately qualified deferred writeback fusion"
    )
    return p


def configure(args):
    if args.mode == "quality" and args.steps not in (1024, 4096):
        raise ValueError("quality requires --steps 1024 or 4096")
    if args.mode == "performance" and args.steps is not None:
        raise ValueError("performance fixes two warmup plus 400 measured updates")
    if args.seed not in (0, 1, 2) or args.method not in ("adamw", "adam8", "strict", "deferred"):
        raise ValueError("unsupported seed or method")
    if args.mode not in ("quality", "performance"):
        raise ValueError("unsupported mode")
    args.fused_scale_writeback = bool(getattr(args, "fused_scale_writeback", False))
    if args.fused_scale_writeback and args.method != "deferred":
        raise ValueError("--fused-scale-writeback requires --method deferred")
    args.context, args.batch, args.micro_batch = 512, 4, 4
    args.rank, args.gap, args.scale, args.retained = 128, 200, 1.0, False
    args.optimizer = "galore" if args.method in ("strict", "deferred") else args.method
    args.extra_config_json = '{"optimizer":{"galore_exclude_vocab":false}}'
    args.checkpoint_dir = args.output_dir / "unused-checkpoint"
    return args


def process_environment(args):
    for key in os.environ:
        if key.startswith("ARENO_EXPERIMENT_") and key not in {
            "ARENO_EXPERIMENT_DEFERRED_PROJECTION",
            "ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK",
        }:
            raise ValueError(f"unregistered private experiment flag: {key}")
    os.environ["ARENO_BENCH_SEED"] = str(args.seed)
    os.environ["ARENO_EXPERIMENT_DEFERRED_PROJECTION"] = str(int(args.method == "deferred"))
    os.environ["ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK"] = str(int(args.fused_scale_writeback))
    os.environ["ARENO_DEFERRED_REPRO_METHOD"] = args.method


def load_helper():
    name = "_areno_deferred_repro_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts/galore_sft/common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def install_observer():
    """Observe native annotations/state without tensor reads or added syncs."""
    from areno.engine.training import TrainingManager

    if getattr(TrainingManager.train, "_deferred_repro_observer", False):
        return
    original = TrainingManager.train

    def observed(self, payload):
        packs = payload.data_packs_by_dp
        if len(packs) != 1 or len(packs[0]) != 1:
            raise RuntimeError("fixed recipe requires one complete SFT pack")
        pack = packs[0][0]
        normalization = {
            key: pack.get(name)
            for key, name in (
                ("sft_total_target_tokens", "_sft_total_target_tokens"),
                ("sft_grad_scale", "_sft_grad_scale"),
            )
        }
        if any(type(value) is not int for value in normalization.values()):
            raise RuntimeError("missing native CPU SFT normalization annotations")
        optimizer = self.worker.optimizer
        if int(getattr(optimizer, "_fused_scale_writeback", False)) != int(
            os.environ.get("ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK", "0")
        ):
            raise RuntimeError("native fused-scale writeback policy does not match the explicit opt-in")
        parameters = list(optimizer.model_params)
        if len(parameters) != 226 or sum(p.numel() for p in parameters) != 1720574976:
            raise RuntimeError("model is not the fixed tied Qwen3-1.7B recipe")
        named = [p for _, p in self.worker.model.named_parameters() if p.requires_grad]
        if [id(p) for p in named] != [id(p) for p in parameters]:
            raise RuntimeError("optimizer owner order differs from native named parameters")
        dtype_elements = {}
        for parameter in parameters:
            dtype = str(parameter.dtype)
            dtype_elements[dtype] = dtype_elements.get(dtype, 0) + parameter.numel()
        if dtype_elements != {"torch.bfloat16": 1720451072, "torch.float32": 123904}:
            raise RuntimeError("native parameter precision policy differs from fixed recipe")
        results = original(self, payload)
        extra = {
            "optimizer_deferred_projection_enabled": int(getattr(optimizer, "_deferred_projection", False)),
            "optimizer_fused_scale_writeback_enabled": int(getattr(optimizer, "_fused_scale_writeback", False)),
            "optimizer_model_owned_master_enabled": int(getattr(optimizer, "_model_owned_master", False)),
            "deferred_cache_empty_after_step": int(not getattr(optimizer, "_deferred_pending", {})),
            "sft_pack_count": 1,
            "sft_dp_shards_per_pack": 1,
            **normalization,
            **getattr(optimizer, "_deferred_last", {}),
        }
        for row in results:
            if row is not None and row.get("metrics"):
                row["metrics"].update(extra)
        return results

    observed._deferred_repro_observer = True
    TrainingManager.train = observed


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(args, common):
    import torch

    import areno

    package = Path(areno.__file__).resolve().parent
    if package.parent != REPO:
        raise RuntimeError("set PYTHONPATH to this repository; imported AReno source differs")
    source = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        source.update(str(path.relative_to(package.parent)).encode())
        source.update(path.read_bytes())
    manifest = json.loads((HERE / "model-manifest.json").read_text())
    model_assets = {}
    for name, expected in manifest["files"].items():
        path = args.model / name
        actual = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if actual != expected:
            raise ValueError(f"model revision asset mismatch: {name}")
        model_assets[name] = actual
    splits = ["train", "validation"] if args.mode == "quality" else ["train"]
    data_hashes = {split: digest(args.data_dir / f"{split}.jsonl") for split in splits}
    data_manifest = json.loads((REPO / "scripts/galore_sft/data-manifest.json").read_text())
    for split, actual in data_hashes.items():
        if actual != data_manifest["files"][f"{split}.jsonl"]["sha256"]:
            raise ValueError(f"fixed data split digest mismatch: {split}")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": "new portable runner; not byte-identical to original formal measurement harness",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_root": str(package.parent),
        "source_sha256": source.hexdigest(),
        "native_extension_sha256": digest(next((package / "accel").glob("_areno_accel*.so"))),
        "wrapper_sha256": {"run.py": digest(__file__), "helper": digest(common.__file__)},
        "model_revision": manifest["revision"],
        "model_assets": model_assets,
        "data_sha256": data_hashes,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "deferred_flag": os.getenv("ARENO_EXPERIMENT_DEFERRED_PROJECTION"),
        "fused_scale_writeback_flag": os.getenv("ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK"),
        "read_test": False,
        "save_checkpoint": False,
    }


def batch_identity(batch):
    return hashlib.sha256(
        json.dumps([(s.tokens, s.prompt_len) for s in batch], separators=(",", ":")).encode()
    ).hexdigest()


def check_update(result, engine, args, index, targets):
    expected = {
        "update_optimizer_steps": 1,
        "update_worker_step": index + 1,
        "update_refresh_any": int(args.optimizer == "galore" and index % 200 == 0),
        "optimizer_deferred_projection_enabled": int(args.method == "deferred"),
        "optimizer_fused_scale_writeback_enabled": int(args.fused_scale_writeback),
        "optimizer_model_owned_master_enabled": 0,
        "sft_pack_count": 1,
        "sft_dp_shards_per_pack": 1,
        "sft_total_target_tokens": targets,
        "sft_grad_scale": 1,
        "logp_abs_diff_mean": 0,
    }
    if engine._ctx.global_step != index or not math.isfinite(result["loss"]):
        raise RuntimeError("native clock or finite loss guard failed")
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"native guard failed: {key}: {result.get(key)} != {value}")
    if args.optimizer == "galore" and not result.get("galore_master_bytes", 0) > 0:
        raise RuntimeError("strict FP32 authority missing")
    if args.method == "deferred":
        for key, expected_value in {
            "deferred_owner_count": 226,
            "deferred_owners_once": 1,
            "deferred_barrier_versions_unchanged": 1,
            "deferred_full_leaf_bytes_after_backward": 0,
            "deferred_cache_empty_after_step": 1,
        }.items():
            if result.get(key) != expected_value:
                raise RuntimeError(f"deferred lifecycle guard failed: {key}")


def summarize(rows):
    return {
        "updates": len(rows),
        "score_plus_train_seconds": sum(r["score_plus_train_seconds"] for r in rows),
        "update_cuda_time_s": sum(r["update_cuda_time_s"] for r in rows),
        "allocated_peak_bytes": max((r["update_peak_allocated_bytes"] for r in rows), default=0),
        "reserved_peak_bytes": max((r["update_peak_reserved_bytes"] for r in rows), default=0),
        "target_tokens": sum(r["target_tokens"] for r in rows),
    }


def execute(args, common, loss_fn, *, clock=time.perf_counter):
    """One fresh engine and a fixed update budget; injectable for CPU smoke tests."""
    warmup, updates = (2, 402) if args.mode == "performance" else (0, args.steps)
    engine = common.trainer(args, resume=False)
    rows, orders = [], {}
    try:
        engine.init()
        if engine._ctx.global_step != -1:
            raise RuntimeError("fresh engine required")
        training, stats = common.encode(engine.get_tokenizer(), args, "train")
        if len(training) != 13544 or stats["rows"] != 13544:
            raise ValueError("fixed encoded train split must contain 13544 examples")
        encoding = {"train": stats}
        common.write(args.output_dir / "encoding.json", encoding)
        with (args.output_dir / "steps.jsonl").open("x") as raw:
            for index in range(updates):
                indices = common.batch_indices(index, args.seed, len(training), args.batch, orders)
                batch = [training[i] for i in indices]
                targets = sum(len(s.tokens) - s.prompt_len for s in batch)
                begin = clock()
                scored = engine.score_logprobs("actor", [s.tokens for s in batch], microbatch_size=args.micro_batch)
                for sample, values in zip(batch, scored, strict=True):
                    sample.logprobs = list(values)
                scored_at = clock()
                result = engine.train(batch, loss_fn, mini_bs=args.micro_batch)
                elapsed = clock() - begin
                check_update(result, engine, args, index, targets)
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": index,
                    "sample_indices": indices,
                    "batch_sha256": batch_identity(batch),
                    "target_tokens": targets,
                    "warmup": index < warmup,
                    "scoring_seconds": scored_at - begin,
                    "score_plus_train_seconds": elapsed,
                    **result,
                }
                raw.write(json.dumps(row, allow_nan=False) + "\n")
                raw.flush()
                rows.append(row)
        measured = rows[warmup:]
        normal = [r for r in measured if not r["update_refresh_any"]]
        refresh = [r for r in measured if r["update_refresh_any"]]
        summary = {
            "scope": "new portable reproduction",
            "all": summarize(measured),
            "ordinary": summarize(normal),
            "refresh": summarize(refresh),
            "optimizer_updates": updates,
            "warmup_updates": warmup,
            "read_test": False,
            "saved_checkpoint": False,
        }
        if args.mode == "performance" and len(refresh) != (2 if args.optimizer == "galore" else 0):
            raise RuntimeError("measured refresh coverage must include absolute steps200/400")
        if args.mode == "quality":
            validation, val_stats = common.encode(engine.get_tokenizer(), args, "validation")
            if len(validation) != 708 or val_stats["rows"] != 708 or val_stats["targets"] != 52181:
                raise ValueError("fixed validation encoding must contain708 examples and52181 targets")
            encoding["validation"] = val_stats
            common.write(args.output_dir / "encoding.json", encoding)
            summary["validation"] = common.evaluate(engine, validation, args.micro_batch)
            if summary["validation"]["examples"] != 708 or summary["validation"]["target_tokens"] != 52181:
                raise RuntimeError("final response NLL coverage mismatch")
        common.write(args.output_dir / "summary.json", summary)
        return summary
    finally:
        engine.close()


def main():
    args = configure(parser().parse_args())
    process_environment(args)
    if not args.model.is_dir():
        raise ValueError("--model must be a local pinned Qwen3-1.7B directory")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    common = load_helper()
    install_observer()
    common.setup(args, "portable-deferred")
    common.write(args.output_dir / "metadata.json", metadata(args, common))
    from areno.api import sft_loss_fn

    execute(args, common, sft_loss_fn)


# Spawned native workers reimport this module with the already-set method/seed.
# Installing here ensures their TrainingManager gets the same observer as parent.
if os.environ.get("ARENO_DEFERRED_REPRO_METHOD"):
    load_helper()
    install_observer()

if __name__ == "__main__":
    main()
