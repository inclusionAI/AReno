"""Single-process inference for classify checkpoints on AReno's own models.

A classify checkpoint is an HF-layout backbone plus `score_head.safetensors`.
`SequenceScorer` rebuilds the backbone through AReno's model adapter (no
`trust_remote_code` model code, so e.g. Ling's transformers-4.45 modeling file
is never imported), runs the packed varlen training forward without the LM
head, and reads the score head at each row's last token. That is the exact
forward used in training, and rows never see padding: every candidate path of
a request is scored in one packed batch.

Runs on one GPU (TP=1) without starting the worker cluster.
"""

from __future__ import annotations

from pathlib import Path


class SequenceScorer:
    """Score token rows with a classify checkpoint: one logit per row."""

    def __init__(self, checkpoint: str | Path, *, attn_backend: str = "flash", max_tokens: int = 16384):
        import torch

        from areno.engine.config import EngineConfig, RuntimeConfig
        from areno.engine.data.tokenizer import load_tokenizer
        from areno.engine.modeling import build_model_on_device
        from areno.engine.parallel.context import get_tp_context
        from areno.engine.score_head import SCORE_HEAD_FILENAME, attach_score_head
        from areno.models.registry import config_from_hf, load_model_weights

        root = Path(checkpoint).expanduser().resolve(strict=True)
        if not (root / SCORE_HEAD_FILENAME).is_file():
            raise FileNotFoundError(f"{root} has no {SCORE_HEAD_FILENAME}; is this a classify checkpoint?")
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


__all__ = ["SequenceScorer"]
