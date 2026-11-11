"""Native CUDA decision head, joint backbone updates, and persistence."""

import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from areno.api.jev.checkpoint import save_head_config
from areno.api.jev.data import candidate_packs, group_questions
from areno.engine.modeling import unwrap_model
from areno.engine.parallel.context import get_tp_context


class DecisionHead(nn.Module):
    """Use the same parameter names, FP32 head, and GELU as MLX."""

    def __init__(self, hidden, device):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden, hidden, device=device, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(hidden, 1, device=device, dtype=torch.float32),
        )

    def forward(self, value):
        return self.layers(value.float())


def initialize_head(worker):
    config = worker.config.runtime.decision_head
    model = unwrap_model(worker.model)
    # Every rank starts with the same replicated head, independently of how
    # many random numbers the model's TP shard initialization consumed.
    devices = [worker.device.index] if worker.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(worker.config.lora_seed)
        model.score_head = DecisionHead(worker.config.model.hidden_size, worker.device)
    checkpoint = config.get("checkpoint")
    if checkpoint:
        from safetensors.torch import load_file

        model.score_head.load_state_dict(load_file(str(Path(checkpoint) / "score_head.safetensors")))
    for parameter in model.score_head.parameters():
        parameter._areno_lr_group = "head"
        parameter._areno_lr = config["head_lr"]
        parameter.tensor_model_parallel = False


def parameter_counts(worker):
    from areno.engine.runtime.train_step import _is_tp_norm_owner

    ctx = get_tp_context()
    counts = {"head": 0, "lora": 0, "backbone": 0}
    for name, value in unwrap_model(worker.model).named_parameters():
        if not value.requires_grad or not _is_tp_norm_owner(value, ctx):
            continue
        key = "head" if name.startswith("score_head.") else "lora" if "lora_" in name else "backbone"
        counts[key] += value.numel()
    values = torch.tensor(list(counts.values()), device=worker.device, dtype=torch.int64)
    if ctx.world_size > 1:
        dist.all_reduce(values, group=ctx.group)
    return dict(zip(counts, values.cpu().tolist(), strict=True))


def question_logits(worker, rows, *, model=None):
    from areno.api.backend.cuda.training import make_train_pack
    from areno.engine.data import to_device
    from areno.engine.runtime.heads import gather_packed_hidden
    from areno.engine.runtime.mst import training_chunk_size
    from areno.engine.runtime.train_step import _pack_train_data, _train_meta

    model = worker.model if model is None else model
    pack = make_train_pack(rows, include_routing_replay=False)
    # Candidate targets are objective metadata, not model input features.
    pack.pop("features", None)
    pack = _pack_train_data(pack)
    pack = to_device(pack, worker.device)
    pack["_activation_checkpointing_enabled"] = worker.config.runtime.activation_checkpointing
    pack["_mst_chunk_size"] = (
        training_chunk_size(worker.device.type, get_tp_context().world_size) if torch.is_grad_enabled() else 0
    )
    meta = _train_meta(pack, pack["input_ids"], sequence_parallel=worker.config.effective_sequence_parallel)
    output = model(input_ids=pack["input_ids"], position_ids=pack["position_ids"], train_meta=meta, defer_lm_head=True)
    if output.hidden_states is None:
        raise RuntimeError("JEV requires the model's normalized hidden_states output")
    hidden = gather_packed_hidden(output.hidden_states, meta).reshape(-1, output.hidden_states.shape[-1])
    # Ignore the singleton TP-alignment padding rows appended by the packer.
    positions = pack["train_cu_seqlens"][1 : len(rows) + 1].long() - 1
    return unwrap_model(model).score_head(hidden[positions]).squeeze(-1)


def backward_questions(worker, pack, *, grad_scale, model=None):
    """Same complete-question loss and bounded two-pass replay as MLX.

    Only forward/backward is owned here. TrainingManager owns the native
    gradient synchronization, clipping, optimizer, offload, and step schedule.
    """
    groups, mini_bs = pack["_jev_groups"], pack["_jev_mini_bs"]
    weight = pack["_jev_weight"]
    objective = pack["_jev_loss_fn"]
    model = worker.model if model is None else model
    model.train()
    score = partial(question_logits, worker, model=model)
    totals, total_loss, microbatches = {}, 0.0, 0

    def loss_for(questions, logits):
        losses, stats, offset = [], {}, 0
        for question in questions:
            target = logits.new_tensor([row.features["jev_target"] for row in question])
            value, values = objective(target, logits[offset : offset + len(question)])
            losses.append(value)
            for key, item in values.items():
                stats[key] = stats.get(key, 0.0) + item.detach()
            offset += len(question)
        return sum(losses), stats

    for candidates in candidate_packs(groups, mini_bs):
        rows = [row for question in candidates for row in question]
        if len(rows) <= mini_bs:
            logits = score(rows)
            loss, stats = loss_for(candidates, logits)
            (loss * weight / grad_scale).backward()
            microbatches += 1
        else:
            scores, rng_states = [], []
            devices = [worker.device.index] if worker.device.type == "cuda" else []
            # Replay the same RNG state, so stochastic backbones also see the
            # same forward when differentiating each scored candidate chunk.
            for start in range(0, len(rows), mini_bs):
                rng_states.append((torch.get_rng_state(), torch.cuda.get_rng_state(worker.device) if devices else None))
                with torch.no_grad():
                    scores.append(score(rows[start : start + mini_bs]))
            logits = torch.cat(scores).detach().requires_grad_()
            loss, stats = loss_for(candidates, logits)
            (score_grad,) = torch.autograd.grad(loss, logits)
            for index, start in enumerate(range(0, len(rows), mini_bs)):
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(rng_states[index][0])
                    if devices:
                        torch.cuda.set_rng_state(rng_states[index][1], worker.device)
                    scores = score(rows[start : start + mini_bs])
                    (scores * score_grad[start : start + len(scores)] * weight / grad_scale).sum().backward()
                microbatches += 1
        total_loss = total_loss + loss.detach() * weight
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value * weight
    if not bool(torch.isfinite(total_loss)):
        raise FloatingPointError("nonfinite JEV loss; update was not applied")
    return total_loss, {
        **totals,
        "candidate_microbatches": microbatches,
        "head_lr": worker.config.runtime.decision_head["head_lr"],
    }


def train_questions(backend, ctx, rows, *, loss_fn, mini_bs):
    started = time.perf_counter()
    groups = group_questions(rows)
    pack = {"_jev_groups": groups, "_jev_mini_bs": mini_bs, "_jev_weight": 1 / len(groups), "_jev_loss_fn": loss_fn}
    (stats,) = backend._require_train_engine().step([pack])
    return {
        "loss": stats.loss,
        **(stats.metrics or {}),
        "questions": len(groups),
        "policy_train_wall_time_s": time.perf_counter() - started,
    }


def score_candidates(worker, rows):
    worker._prepare_actor_onloaded()
    worker.model.eval()
    with torch.no_grad():
        # Bound inference activations, while batching candidates across requests
        # to amortize IPC, packing and TP launches. Preserve original ordering.
        values = []
        for start in range(0, len(rows), 32):
            values.extend(question_logits(worker, rows[start : start + 32]).float().cpu().tolist())
        return values


def save_head(worker, path):
    from safetensors.torch import save_file

    ctx = get_tp_context()
    if ctx.rank != 0 or ctx.dp_rank != 0:
        return
    model = unwrap_model(worker.model)
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in model.score_head.state_dict().items()},
        str(destination / "score_head.safetensors"),
    )
    save_head_config(
        destination,
        base_model=worker.config.base_model_name_or_path or worker.config.model_path,
        max_seq_len=worker.config.runtime.decision_head["max_seq_len"],
    )
