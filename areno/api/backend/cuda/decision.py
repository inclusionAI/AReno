"""CUDA grouped-softmax decision loss for JEV training.

The CUDA engine hands the loss one scalar per packed row (the actor's score head
reads each candidate's last token). This module turns those scores into the same
per-question objective as `areno.api.jev.loss.jev_loss_fn` (cross-entropy plus an
optional Brier term), using the numeric `sequence_labels` that
`areno.api.jev.data.encode_question` stamps on every candidate row (`group`,
`target`, `weight`).

One JEV question must reach the loss whole: its candidates share a single
softmax. `JevTrainer` therefore assigns one `trajectory_id` per question (so the
engine's grouped DP split keeps candidates together) and stamps the per-question
loss weight that makes the step gradient the mean over the step's questions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# sequence_labels keys used by the JEV CUDA loss.
GROUP_LABEL = "group"
TARGET_LABEL = "target"
WEIGHT_LABEL = "weight"


def jev_loss(pack: dict, scores) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Worker-side entry: `scores` holds one logit per packed sequence.

    Each row's `weight` is the per-question loss weight the trainer chose so the
    DP- and accumulation-averaged gradient equals the mean over the step's
    questions; the loss is the weighted sum over this rank's questions.
    """

    import torch

    labels = pack.get("sequence_labels")
    if not isinstance(labels, dict):
        raise ValueError("JEV loss requires `sequence_labels` from encode_question")
    missing = {GROUP_LABEL, TARGET_LABEL, WEIGHT_LABEL} - set(labels)
    if missing:
        raise ValueError(f"JEV loss requires sequence_labels {sorted(missing)}")
    group = labels[GROUP_LABEL].to(device=scores.device, dtype=torch.long)
    target = labels[TARGET_LABEL].to(device=scores.device, dtype=torch.float32)
    weight = labels[WEIGHT_LABEL].to(device=scores.device, dtype=torch.float32)
    scores = scores.float().reshape(-1)
    if scores.numel() != group.numel():
        raise ValueError(f"expected {group.numel()} scores, got {scores.numel()}")

    # Remap group ids to a dense [0, G) range so `index_add` can segment them.
    unique, index = torch.unique(group, sorted=True, return_inverse=True)
    groups = unique.numel()

    def segment_max(values: torch.Tensor) -> torch.Tensor:
        out = values.new_full((groups,), float("-inf"))
        return out.scatter_reduce(0, index, values, reduce="amax", include_self=True)

    def segment_sum(values: torch.Tensor) -> torch.Tensor:
        return values.new_zeros(groups).index_add(0, index, values)

    shifted = scores - segment_max(scores.detach())[index]
    log_probs = shifted - torch.log(segment_sum(torch.exp(shifted)))[index]
    probabilities = torch.exp(log_probs)
    ce = segment_sum(-(target * log_probs))
    brier = segment_sum((probabilities - target).square())
    # Every row of a question carries the same weight; `amax` reads it once.
    group_weight = weight.new_zeros(groups).scatter_reduce(0, index, weight, reduce="amax", include_self=False)
    loss = (group_weight * (ce + brier * _brier_weight(pack))).sum()
    with torch.no_grad():
        return loss, {
            "jev_ce": ce.detach().mean(),
            "jev_brier": brier.detach().mean(),
            "jev_questions": torch.tensor(float(groups), device=scores.device),
        }


def _brier_weight(pack: dict) -> float:
    """Read the Brier weight the algorithm loss bound through its partial."""

    settings = pack.get("_loss_settings")
    if isinstance(settings, dict) and "brier_weight" in settings:
        return float(settings["brier_weight"])
    return 0.5


class SequenceScorer:
    """Score token rows with a JEV checkpoint: one logit per row.

    A JEV checkpoint is an HF-layout backbone plus `score_head.safetensors`.
    The backbone is rebuilt through AReno's model adapter (never
    `trust_remote_code`), runs the packed varlen training forward without the LM
    head, and reads the score head at each row's last token: the exact forward
    used in training, with no padding and every candidate path in one packed
    batch. Runs on one GPU without starting the worker cluster.
    """

    def __init__(self, checkpoint, *, attn_backend: str = "flash", max_tokens: int = 16384):
        import torch

        from areno.engine.config import EngineConfig, RuntimeConfig
        from areno.engine.data.tokenizer import load_tokenizer
        from areno.engine.modeling import build_model_on_device
        from areno.engine.parallel.context import get_tp_context
        from areno.engine.score_head import SCORE_HEAD_FILENAME, attach_score_head
        from areno.models.registry import config_from_hf, load_model_weights

        root = Path(checkpoint).expanduser().resolve(strict=True)
        if not (root / SCORE_HEAD_FILENAME).is_file():
            raise FileNotFoundError(f"{root} has no {SCORE_HEAD_FILENAME}; is this a JEV checkpoint?")
        ctx = get_tp_context()
        if ctx.world_size != 1:
            raise RuntimeError("SequenceScorer runs single-process (TP=1)")
        self.torch = torch
        self.device = ctx.device
        config = EngineConfig(
            model=config_from_hf(root),
            model_path=str(root),
            runtime=RuntimeConfig(attn_backend=attn_backend, compile_model=False, activation_checkpointing=False),
            tp_size=1,
            devices=[self.device.index or 0],
            role="rollout",
        )
        self.model = build_model_on_device(config, self.device)
        load_model_weights(self.model, config.model, str(root))
        self.model.onload_train_weights(self.device)
        self.head = attach_score_head(
            self.model,
            hidden_size=config.model.hidden_size,
            dtype=config.model.dtype,
            device=self.device,
            model_path=str(root),
        )
        self.model.eval()
        self.tokenizer = load_tokenizer(str(root))
        self.max_tokens = int(max_tokens)

    def score(self, rows: list[list[int]]):
        """Return a float32 tensor with one score per row (same order)."""

        torch = self.torch
        scores = []
        chunk: list[list[int]] = []
        used = 0
        for row in rows:
            if chunk and used + len(row) > self.max_tokens:
                scores.append(self._score_packed(chunk))
                chunk, used = [], 0
            chunk.append(row)
            used += len(row)
        if chunk:
            scores.append(self._score_packed(chunk))
        return torch.cat(scores) if scores else torch.empty(0)

    def _score_packed(self, rows: list[list[int]]):
        torch = self.torch
        from areno.engine.runtime.train_step import _pack_train_data, _train_meta
        from areno.engine.score_head import packed_sequence_scores

        width = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long, device=self.device)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=self.device)
        shape = input_ids.shape
        packed = _pack_train_data(
            {
                "input_ids": input_ids,
                "lengths": torch.tensor([len(row) for row in rows], device=self.device),
                "prompt_mask": torch.ones(shape, dtype=torch.bool, device=self.device),
                "advantages": torch.zeros(shape, dtype=torch.float32, device=self.device),
                "logprobs": torch.zeros(shape, dtype=torch.float32, device=self.device),
            }
        )
        tokens = packed["input_ids"]
        meta = _train_meta(packed, tokens, sequence_parallel=False)
        with torch.inference_mode():
            out = self.model(input_ids=tokens, position_ids=packed["position_ids"], train_meta=meta, defer_lm_head=True)
            return packed_sequence_scores(
                self.head, out.hidden_states, packed["train_cu_seqlens"], len(rows), sequence_parallel=False
            )


__all__ = ["GROUP_LABEL", "SequenceScorer", "TARGET_LABEL", "WEIGHT_LABEL", "jev_loss"]