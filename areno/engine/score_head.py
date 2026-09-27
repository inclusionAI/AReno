"""Per-sequence scoring head for classification-style actor training.

When `RuntimeConfig.score_head` is enabled the actor carries a small MLP
(`Linear -> GELU -> Linear(1)`) as the `score_head` submodule. Training reads
the final-norm hidden state at each packed sequence's last token and turns it
into one scalar score; the LM head is skipped entirely. Group-wise objectives
(softmax over the candidates of one question, as in JevForge-style decision
models) are left to the loss function.

The head is replicated on every TP and DP rank. It is initialized from a fixed
seed so replicas start identical, and it is saved next to the HF backbone as
`score_head.safetensors` (keys `0.weight`, `0.bias`, `2.weight`, `2.bias`).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch import nn

from areno.engine.parallel.collectives import gather_from_sequence_parallel_region
from areno.engine.parallel.context import get_tp_context

SCORE_HEAD_FILENAME = "score_head.safetensors"
SCORE_HEAD_LR_GROUP = "score_head"


class _ScaleGradient(torch.autograd.Function):
    """Keep forward values unchanged while scaling the input gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad * ctx.scale, None


def build_score_head(hidden_size: int, *, seed: int = 0) -> nn.Sequential:
    """Build the fp32 CPU head with a seed shared by every rank."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1))


def attach_score_head(
    model: nn.Module,
    *,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    model_path: str | None,
) -> nn.Module:
    """Register `model.score_head`, resuming from `model_path` when it has one."""

    forward = inspect.signature(model.forward).parameters
    if "defer_lm_head" not in forward:
        raise ValueError(f"score_head requires a model whose forward accepts defer_lm_head; got {type(model).__name__}")
    head = build_score_head(hidden_size)
    saved = Path(model_path) / SCORE_HEAD_FILENAME if model_path is not None else None
    if saved is not None and saved.is_file():
        from safetensors.torch import load_file

        head.load_state_dict(load_file(str(saved), device="cpu"), strict=True)
    head.to(device=device, dtype=dtype)
    # The worker schedules this LR group like the multimodal tower/projector.
    for param in head.parameters():
        param._areno_lr_group = SCORE_HEAD_LR_GROUP
    model.score_head = head
    return head


def packed_sequence_scores(
    head: nn.Module,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_sequences: int,
    *,
    sequence_parallel: bool,
) -> torch.Tensor:
    """Score the last token of each of the first `num_sequences` packed rows."""

    if sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)
        # Every TP rank applies the same replicated head and back-propagates
        # the same hidden gradient; the gather's reduce-scatter sums those
        # copies, so average them before they reach the TP backbone.
        hidden_states = _ScaleGradient.apply(hidden_states, 1.0 / get_tp_context().world_size)
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    ends = cu_seqlens[1 : num_sequences + 1].to(device=flat.device, dtype=torch.long) - 1
    return head(flat.index_select(0, ends)).squeeze(-1).float()


@torch.no_grad()
def save_score_head(head: nn.Module, output_dir: str | Path) -> str:
    """Write the head as fp32 safetensors next to the saved backbone."""

    from safetensors.torch import save_file

    path = Path(output_dir) / SCORE_HEAD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {name: tensor.detach().float().cpu().contiguous() for name, tensor in head.state_dict().items()}
    save_file(tensors, str(path))
    return str(path)


__all__ = [
    "SCORE_HEAD_FILENAME",
    "SCORE_HEAD_LR_GROUP",
    "attach_score_head",
    "build_score_head",
    "packed_sequence_scores",
    "save_score_head",
]
