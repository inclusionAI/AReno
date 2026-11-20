"""Small CPU reference for multi-sample shared-prompt KV semantics.

This module is an executable specification for issue #592, not the production
rollout path.  ``TinyCausalLM.forward`` is the independent-full-forward oracle;
``SharedPromptKVCache`` keeps one immutable prompt prefix and a private KV
suffix per sample.  No production scheduler or attention backend is imported.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

KVPair = tuple[torch.Tensor, torch.Tensor]
PrefixKV = tuple[KVPair, ...]
SuffixKV = tuple[KVPair | None, ...]


def _split_heads(projection: torch.Tensor, num_heads: int) -> torch.Tensor:
    tokens, width = projection.shape
    return projection.view(tokens, num_heads, width // num_heads).transpose(0, 1).contiguous()


class _TinyBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.attention_out = nn.Linear(hidden_size, hidden_size)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        prefix_kv: KVPair | None = None,
        suffix_kv: KVPair | None = None,
    ) -> tuple[torch.Tensor, KVPair]:
        residual = hidden
        normalized = self.attention_norm(hidden)
        query, key, value = (_split_heads(part, self.num_heads) for part in self.qkv(normalized).chunk(3, dim=-1))
        new_kv = (key, value)

        past = tuple(pair for pair in (prefix_kv, suffix_kv) if pair is not None)
        past_length = sum(pair[0].shape[1] for pair in past)
        if past:
            key = torch.cat([*(pair[0] for pair in past), key], dim=1)
            value = torch.cat([*(pair[1] for pair in past), value], dim=1)

        token_count = hidden.shape[0]
        query_positions = past_length + torch.arange(token_count, device=hidden.device)
        key_positions = torch.arange(past_length + token_count, device=hidden.device)
        visible = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=visible)
        attended = attended.transpose(0, 1).reshape(token_count, -1)
        hidden = residual + self.attention_out(attended)
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden, new_kv


class TinyCausalLM(nn.Module):
    """Two-layer unbatched Transformer with learned absolute positions."""

    def __init__(
        self,
        *,
        vocab_size: int = 64,
        hidden_size: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.blocks = nn.ModuleList(_TinyBlock(hidden_size, num_heads) for _ in range(num_layers))
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 1 or tokens.numel() == 0:
            raise ValueError("tokens must be a non-empty rank-1 tensor")
        if tokens.dtype != torch.long:
            raise TypeError("tokens must have dtype torch.long")

    def _run(
        self,
        tokens: torch.Tensor,
        *,
        start: int = 0,
        prefix: PrefixKV | None = None,
        suffix: SuffixKV | None = None,
    ) -> tuple[torch.Tensor, PrefixKV]:
        self._validate_tokens(tokens)
        end = start + tokens.numel()
        if end > self.max_seq_len:
            raise ValueError(f"sequence length {end} exceeds max_seq_len={self.max_seq_len}")
        positions = torch.arange(start, end, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        new_kv = []
        for index, block in enumerate(self.blocks):
            hidden, layer_kv = block(
                hidden,
                prefix_kv=None if prefix is None else prefix[index],
                suffix_kv=None if suffix is None else suffix[index],
            )
            new_kv.append(layer_kv)
        return self.lm_head(self.final_norm(hidden)), tuple(new_kv)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the independent full-sequence oracle and return all logits."""

        return self._run(tokens)[0]

    def prefill_with_kv(self, prompt: torch.Tensor) -> tuple[torch.Tensor, PrefixKV]:
        """Compute a prompt once and retain its per-layer KV pairs."""

        return self._run(prompt)

    def decode_with_shared_kv(
        self,
        token: int,
        prefix: PrefixKV,
        suffix: SuffixKV,
    ) -> tuple[torch.Tensor, SuffixKV]:
        """Decode one token without modifying or copying the shared prefix."""

        if len(prefix) != len(self.blocks) or len(suffix) != len(self.blocks):
            raise ValueError("one KV entry per Transformer layer is required")
        prefix_lengths = {pair[0].shape[1] for pair in prefix}
        suffix_lengths = {pair[0].shape[1] for pair in suffix if pair is not None}
        if len(prefix_lengths) != 1 or len(suffix_lengths) > 1:
            raise ValueError("inconsistent KV lengths across layers")
        prefix_length = prefix_lengths.pop()
        suffix_length = suffix_lengths.pop() if suffix_lengths else 0
        device = self.token_embedding.weight.device
        tokens = torch.tensor([token], dtype=torch.long, device=device)
        logits, new_kv = self._run(
            tokens,
            start=prefix_length + suffix_length,
            prefix=prefix,
            suffix=suffix,
        )
        updated = tuple(
            new if old is None else (torch.cat((old[0], new[0]), dim=1), torch.cat((old[1], new[1]), dim=1))
            for old, new in zip(suffix, new_kv, strict=True)
        )
        return logits[0], updated


class SharedPromptBranch:
    """One sample's private continuation owned by ``SharedPromptKVCache``."""

    def __init__(self, cache: SharedPromptKVCache) -> None:
        self._cache = cache
        self._suffix: SuffixKV = tuple(None for _ in cache.prefix)
        self._suffix_tokens: tuple[int, ...] = ()
        self._active = True

    @property
    def suffix_tokens(self) -> tuple[int, ...]:
        return self._suffix_tokens

    @property
    def shared_prefix(self) -> PrefixKV:
        self._require_active()
        return self._cache.prefix

    @property
    def private_suffix(self) -> SuffixKV:
        return self._suffix

    @property
    def tokens(self) -> torch.Tensor:
        self._require_active()
        suffix = torch.tensor(self._suffix_tokens, dtype=torch.long, device=self._cache.prompt.device)
        return torch.cat((self._cache.prompt, suffix))

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("branch belongs to a reset shared-KV generation")

    def _release(self) -> None:
        self._active = False
        self._suffix = ()
        self._suffix_tokens = ()

    def decode(self, token: int) -> torch.Tensor:
        """Append one private token and return logits for the following token."""

        self._require_active()
        if not 0 <= token < self._cache.model.vocab_size:
            raise ValueError("token is outside the model vocabulary")
        logits, self._suffix = self._cache.model.decode_with_shared_kv(token, self._cache.prefix, self._suffix)
        self._suffix_tokens += (token,)
        return logits


class SharedPromptKVCache:
    """Minimal reusable owner for one shared prompt and its sample branches."""

    def __init__(self, model: TinyCausalLM) -> None:
        self.model = model
        self.prompt = torch.empty(0, dtype=torch.long)
        self.prefix: PrefixKV = ()
        self.prefill_logits = torch.empty(0)
        self._branches: list[SharedPromptBranch] = []

    @property
    def active(self) -> bool:
        return bool(self.prefix)

    @property
    def branches(self) -> tuple[SharedPromptBranch, ...]:
        return tuple(self._branches)

    def start(self, prompt: torch.Tensor, num_samples: int) -> tuple[SharedPromptBranch, ...]:
        """Prefill a fresh generation; callers must reset before reuse."""

        if self.active:
            raise RuntimeError("reset is required before starting another shared prompt")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.prompt = prompt.clone()
        self.prefill_logits, self.prefix = self.model.prefill_with_kv(self.prompt)
        self._branches = [SharedPromptBranch(self) for _ in range(num_samples)]
        return self.branches

    def reset(self) -> None:
        """Invalidate branches and release every prompt and private-KV reference."""

        for branch in self._branches:
            branch._release()
        self._branches = []
        self.prompt = torch.empty(0, dtype=torch.long)
        self.prefix = ()
        self.prefill_logits = torch.empty(0)
