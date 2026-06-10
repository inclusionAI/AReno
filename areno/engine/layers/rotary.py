"""Rotary positional embeddings.

Three flavors are provided: full RoPE (rotates every channel pair), partial
RoPE (rotates only the first ``partial_rotary_factor`` fraction of channels
and passes the rest through), and a Gemma4 variant that places zero-frequency
pairs after the rotary pairs so the same cos/sin tables can be reused on the
full head. All flavors precompute cos/sin tables for every position up to
``max_position`` at construction time and gather them at forward.
"""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dim by 90 degrees: (-x2, x1)."""

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Standard RoPE that rotates every (i, i+head_dim/2) pair.

    Precomputes cos/sin tables of shape ``(max_position, head_dim)`` (the
    half-frequencies are doubled by concatenation so the table aligns with
    `rotate_half`). At forward, positions index into the cached tables and
    the resulting cos/sin are broadcast across the heads dimension.
    """

    def __init__(self, head_dim: int, max_position: int, theta: float):
        super().__init__()
        # Inverse frequencies of even channels in [0, head_dim).
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_position, dtype=torch.float32)
        # Outer product yields per-position phase per frequency.
        freqs = torch.outer(positions, inv_freq)
        # Doubled phase table aligns with rotate_half's (x1, x2) layout.
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Gather positional cos/sin and broadcast over heads (unsqueeze(2)).
        cos = self.cos_cached[position_ids].unsqueeze(2).to(dtype=q.dtype)
        sin = self.sin_cached[position_ids].unsqueeze(2).to(dtype=q.dtype)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class PartialRotaryEmbedding(nn.Module):
    """RoPE applied to only the first ``partial_rotary_factor`` of each head.

    The remaining "pass" channels are left untouched and concatenated back.
    Supports both neox-style (paired by half) and gpt-j-style (paired by
    adjacent channels) rotation via ``is_neox_style``.
    """

    def __init__(self, head_dim: int, max_position: int, theta: float, partial_rotary_factor: float, *, is_neox_style: bool):
        super().__init__()
        self.is_neox_style = is_neox_style
        # Only this many leading channels get rotated.
        self.rope_dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim))
        positions = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[position_ids].unsqueeze(2).to(dtype=q.dtype)
        sin = self.sin_cached[position_ids].unsqueeze(2).to(dtype=q.dtype)
        # Split each head into rotated and pass-through segments.
        q_rot, q_pass = q[..., : self.rope_dim], q[..., self.rope_dim :]
        k_rot, k_pass = k[..., : self.rope_dim], k[..., self.rope_dim :]
        q_rot = apply_rotary(q_rot, cos, sin, self.is_neox_style)
        k_rot = apply_rotary(k_rot, cos, sin, self.is_neox_style)
        return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


class Gemma4RotaryEmbedding(nn.Module):
    """Gemma4 RoPE rotates across the two halves of each attention head.

    Builds inv_freq for the rotary half and pads with zero frequencies for
    the non-rotated channels, so the cos/sin table covers the full head dim
    and channels in the no-rope slice end up with cos=1, sin=0 (identity).
    """

    def __init__(self, head_dim: int, max_position: int, theta: float, partial_rotary_factor: float):
        super().__init__()
        rotary_dim = int(head_dim * partial_rotary_factor)
        # Number of (real) frequency pairs to rotate.
        rope_angles = rotary_dim // 2
        # Remaining pairs that should pass through with the identity rotation.
        nope_angles = (head_dim // 2) - rope_angles
        inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim))
        if nope_angles > 0:
            # Append zero-freq pairs so the table spans the full head dim
            # and the non-rotary slice degenerates to (cos=1, sin=0).
            inv_freq = torch.cat((inv_freq, torch.zeros(nope_angles, dtype=torch.float32)))
        positions = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[position_ids].unsqueeze(2).to(dtype=q.dtype)
        sin = self.sin_cached[position_ids].unsqueeze(2).to(dtype=q.dtype)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox_style: bool) -> torch.Tensor:
    """Apply RoPE to ``x`` using neox or gpt-j channel pairing."""

    if is_neox_style:
        # Pair channel i with channel i + head_dim/2 (rotate_half layout).
        return (x * cos) + (rotate_half(x) * sin)
    # gpt-j style: pair adjacent channels (i, i+1) and rotate per pair.
    cos = cos[..., : x.shape[-1] // 2]
    sin = sin[..., : x.shape[-1] // 2]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out1 = (x1 * cos) - (x2 * sin)
    out2 = (x2 * cos) + (x1 * sin)
    # Re-interleave the rotated pairs back to the original channel order.
    return torch.stack((out1, out2), dim=-1).flatten(-2)
