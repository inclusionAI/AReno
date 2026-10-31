"""Frozen experiment harness: matched batches, response targets and public Trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import torch

import areno
from areno import Trainer
from areno.api import CudaConfig, TrainSequence
from areno.engine.training import TrainingManager

UTC = timezone.utc


ALIAS = Path(__file__).resolve().parent
if "ARENO_BENCH_SEED" in os.environ:
    torch.manual_seed(int(os.environ["ARENO_BENCH_SEED"]))
    random.seed(int(os.environ["ARENO_BENCH_SEED"]))

# Native profiles reset per microbatch. Preserve them and additionally expose
# sum(time)/max(peaks) per actual optimizer update, instead of averaging peaks.
if not getattr(TrainingManager.train, "_overnight_profile", False):
    _native_train = TrainingManager.train

    def _profile_update(self, payload):
        before = self.worker._global_step
        results = _native_train(self, payload)
        rows = [row["metrics"] for row in results if row is not None and row.get("metrics")]
        if rows:
            aggregate = {
                "update_cuda_time_s": sum(row.get("cuda_step_time_s", 0) for row in rows),
                "update_peak_allocated_bytes": max(row.get("cuda_peak_allocated_bytes", 0) for row in rows),
                "update_peak_reserved_bytes": max(row.get("cuda_peak_reserved_bytes", 0) for row in rows),
                "update_optimizer_steps": self.worker._global_step - before,
                "update_worker_step": self.worker._global_step,
                "update_refresh_any": float(any(row.get("galore_projection_refresh", False) for row in rows)),
            }
            for row in rows:
                row.update(aggregate)
        return results

    _profile_update._overnight_profile = True
    TrainingManager.train = _profile_update


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--optimizer", choices=["adamw", "adam8", "galore"], default="galore")
    p.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--gap", type=int, default=200)
    p.add_argument("--scale", type=float, default=4)
    p.add_argument("--retained", action="store_true")
    p.add_argument("--extra-config-json", default="{}")
    return p


def setup(args, phase):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(args.output_dir / f"{phase}.log"), logging.StreamHandler()],
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["ARENO_BENCH_SEED"] = str(args.seed)
    if args.context < 4 or args.batch < 1 or args.micro_batch < 1:
        raise ValueError("invalid context/batch")
    return logging.getLogger(phase)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(args):
    package = Path(areno.__file__).resolve().parent
    sha = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        sha.update(str(path.relative_to(package.parent)).encode())
        sha.update(path.read_bytes())
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_root": str(package.parent),
        "source_sha256": sha.hexdigest(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
        "native_extension_sha256": digest(next((package / "accel").glob("_areno_accel*.so"))),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "data_sha256": {
            s: digest(args.data_dir / f"{s}.jsonl")
            for s in ["train", "validation", "test"]
            if (args.data_dir / f"{s}.jsonl").exists()
        },
        "wrapper_sha256": {s: digest(ALIAS / s) for s in ["train.py", "eval.py", "common.py"]},
        "encoding_policy": "response-reserve-min-half; head-tail-prompt; response-prefix; EOS-if-fit-v1",
        "order_policy": "absolute-cursor-epoch-shuffle-seed-plus-1000003epoch-v1",
        "model_identity": "tied596049920-parameters; BF16-with-FP32-norms",
        "profiler_policy": "sum-per-microbatch-time-max-per-microbatch-peaks-v1",
    }


def trainer(args, *, resume=False, output=None):
    opt = dict(
        lr=1e-5,
        min_lr=1e-5,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        grad_clip_norm=1.0,
        save_training_state=True,
        resume_training_state=resume,
    )
    if args.optimizer == "adam8":
        opt["adam_8bit"] = True
    if args.optimizer == "galore":
        opt.update(
            galore_rank=args.rank,
            galore_update_proj_gap=args.gap,
            galore_scale=args.scale,
            galore_projector_backend="recycled",
            galore_projector_oversampling=16,
            galore_projector_iterations=1,
            galore_projector_seed=args.seed,
            galore_projector_final_qr=True,
            galore_compact_master=True,
            galore_tile_rows=16384,
            galore_release_gradients=True,
            galore_fused_writeback=True,
            galore_transport_moments=not args.retained,
        )
    runtime = dict(
        attn_backend="native",
        compile_model=False,
        activation_checkpointing=True,
        keep_rollout_state=True,
        profile_training=True,
    )
    extra = json.loads(args.extra_config_json)
    if set(extra) - {"optimizer", "runtime"}:
        raise ValueError("extra config may only contain optimizer/runtime")
    opt.update(extra.get("optimizer", {}))
    runtime.update(extra.get("runtime", {}))
    return Trainer(
        world_size=1,
        model_path=str(args.checkpoint_dir if resume else args.model),
        custom_config=CudaConfig(devices=[0], tp_size=1, optimizer=opt, runtime=runtime),
        metrics_log_dir=str((output or args.output_dir) / "events"),
    )


def encode(tokenizer, args, split):
    records = [json.loads(line) for line in (args.data_dir / f"{split}.jsonl").read_text().splitlines()]
    encoded, stats = [], dict(rows=len(records), prompt_truncated=0, response_truncated=0, tokens=0, targets=0)
    for record in records:
        if "text" in record:
            tokens = tokenizer.encode(record["text"], add_special_tokens=False)[: args.context]
            prompt_len = 1
        else:
            prompt = tokenizer.encode(record["prompt"], add_special_tokens=False)
            response = tokenizer.encode(record["response"], add_special_tokens=False)
            eos = tokenizer.eos_token_id
            if eos is not None and (not response or response[-1] != eos):
                response.append(eos)
            reserve = min(len(response), args.context // 2)
            cap = args.context - max(reserve, 1)
            if len(prompt) > cap:
                left = (cap + 1) // 2
                prompt = prompt[:left] + prompt[-(cap - left) :] if cap > left else prompt[:left]
                stats["prompt_truncated"] += 1
            if not prompt:
                prompt = [tokenizer.bos_token_id or tokenizer.eos_token_id or 0]
            space = args.context - len(prompt)
            stats["response_truncated"] += int(len(response) > space)
            tokens = prompt + response[:space]
            prompt_len = len(prompt)
        if len(tokens) <= prompt_len:
            raise ValueError("encoding produced a response-free example")
        stats["tokens"] += len(tokens)
        stats["targets"] += len(tokens) - prompt_len
        encoded.append(
            TrainSequence(
                tokens=tokens,
                prompt_len=prompt_len,
                logprobs=[],
                scalar_advantage=1.0,
                reward=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )
        )
    return encoded, stats


def batch_indices(step, seed, size, batch, orders):
    result = []
    for cursor in range(step * batch, (step + 1) * batch):
        epoch, offset = divmod(cursor, size)
        if epoch not in orders:
            order = list(range(size))
            random.Random(seed + 1000003 * epoch).shuffle(order)
            orders[epoch] = order
        result.append(orders[epoch][offset])
    return result


def evaluate(engine, examples, micro_batch):
    total, count = 0.0, 0
    for start in range(0, len(examples), micro_batch):
        batch = examples[start : start + micro_batch]
        scored = engine.score_logprobs("actor", [s.tokens for s in batch], microbatch_size=micro_batch)
        for sample, values in zip(batch, scored, strict=True):
            targets = values[sample.prompt_len :]
            if not targets or not all(math.isfinite(v) for v in targets):
                raise ValueError("nonfinite/empty evaluation")
            total -= sum(targets)
            count += len(targets)
    return {"loss": total / count, "target_tokens": count, "examples": len(examples)}
