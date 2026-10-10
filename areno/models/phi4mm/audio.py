"""Native Phi-4-Multimodal Conformer audio encoder and projector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True, slots=True)
class Phi4MMAudioConfig:
    input_size: int = 80
    attention_dim: int = 1024
    attention_heads: int = 16
    linear_units: int = 1536
    num_blocks: int = 24
    kernel_size: int = 3
    time_reduction: int = 8
    relative_attention_max_distance: int = 500

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Phi4MMAudioConfig:
        relative = values.get("relative_attention_bias_args") or {}
        normalized = dict(values)
        normalized["relative_attention_max_distance"] = int(relative.get("t5_bias_max_distance", 500))
        return cls(**{name: normalized[name] for name in cls.__dataclass_fields__ if name in normalized})


class _Swish(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(value)


class _GLULinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dtype: torch.dtype):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2, dtype=dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        left, gate = self.linear(value).chunk(2, dim=-1)
        return left * (gate * torch.sigmoid(gate))


class _FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, dtype=dtype)
        self.net = nn.Sequential(
            _GLULinear(hidden_size, intermediate_size, dtype),
            nn.Dropout(0.0),
            nn.Linear(intermediate_size, hidden_size, dtype=dtype),
            nn.Dropout(0.0),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(self.layer_norm(value))


class _GLUPointWiseConv(nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.output_dim = hidden_size
        self.ext_pw_conv_1d = nn.Conv1d(hidden_size, hidden_size * 2, 1, padding=0, dtype=dtype)
        self.b1 = nn.Parameter(torch.zeros(1, hidden_size, 1, dtype=dtype))
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size, 1, dtype=dtype))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.ext_pw_conv_1d(value.transpose(1, 2))
        left = value[:, : self.output_dim] + self.b1
        gate = value[:, self.output_dim :] + self.b2
        return (left * (gate * torch.sigmoid(gate))).transpose(1, 2)


class _DepthWiseSeperableConv1d(nn.Module):
    # Keep the official misspelling in this private class because it defines
    # checkpoint-compatible attribute names.
    def __init__(self, hidden_size: int, kernel_size: int, dtype: torch.dtype):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            padding=kernel_size - 1,
            groups=hidden_size,
            dtype=dtype,
        )
        self.pw_conv = nn.Conv1d(hidden_size, hidden_size, 1, dtype=dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.pw_conv(self.dw_conv(value))


class _ConvModule(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, dtype: torch.dtype):
        super().__init__()
        self.kernel_size = kernel_size
        self.layer_norm = nn.LayerNorm(hidden_size, dtype=dtype)
        self.glu = _GLUPointWiseConv(hidden_size, dtype)
        self.dw_sep_conv_1d = _DepthWiseSeperableConv1d(hidden_size, kernel_size, dtype)
        self.ext_pw_conv_1d = nn.Conv1d(hidden_size, hidden_size, 1, dtype=dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.glu(self.layer_norm(value)).transpose(1, 2)
        value = self.dw_sep_conv_1d(value)
        value = value[:, :, : -(self.kernel_size - 1)]
        value = value * torch.sigmoid(value)
        return self.ext_pw_conv_1d(value).transpose(1, 2)


class _T5RelativeAttentionLogitBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int, dtype: torch.dtype):
        super().__init__()
        self.max_distance = max_distance
        self.bias_values = nn.Embedding(max_distance * 2, num_heads, dtype=dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        length = value.shape[1]
        positions = torch.arange(length, device=value.device, dtype=torch.long)
        relative = positions[None, :] - positions[:, None]
        indices = relative.clamp(-self.max_distance, self.max_distance - 1) + self.max_distance
        return self.bias_values(indices).permute(2, 0, 1).unsqueeze(0)


class _MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dtype: torch.dtype):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("Phi4MM audio attention dimension must be divisible by its head count")
        self.h = num_heads
        self.d_k = hidden_size // num_heads
        self.inv_sqrt_d_k = self.d_k**-0.5
        self.linear_q = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.linear_k = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.linear_v = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.linear_out = nn.Linear(hidden_size, hidden_size, dtype=dtype)

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        relative_attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        batch = value.shape[0]
        query = self.linear_q(value).view(batch, -1, self.h, self.d_k).transpose(1, 2) * self.inv_sqrt_d_k
        key = self.linear_k(value).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        projected = self.linear_v(value).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) + relative_attention_bias
        if mask is not None:
            invalid = ~mask.unsqueeze(1)
            attention = torch.softmax(scores.masked_fill(invalid, -torch.inf), dim=-1).masked_fill(invalid, 0.0)
        else:
            attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention.to(projected.dtype), projected)
        output = output.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)
        return self.linear_out(output)


class _ConformerEncoderLayer(nn.Module):
    def __init__(self, config: Phi4MMAudioConfig, dtype: torch.dtype):
        super().__init__()
        self.feed_forward_in = _FeedForward(config.attention_dim, config.linear_units, dtype)
        self.self_attn = _MultiHeadedAttention(config.attention_dim, config.attention_heads, dtype)
        self.conv = _ConvModule(config.attention_dim, config.kernel_size, dtype)
        self.feed_forward_out = _FeedForward(config.attention_dim, config.linear_units, dtype)
        self.layer_norm_att = nn.LayerNorm(config.attention_dim, dtype=dtype)
        self.layer_norm = nn.LayerNorm(config.attention_dim, dtype=dtype)

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        relative_attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        value = value + 0.5 * self.feed_forward_in(value)
        normalized = self.layer_norm_att(value)
        value = value + self.self_attn(normalized, mask, relative_attention_bias)
        value = value + self.conv(value)
        value = value + 0.5 * self.feed_forward_out(value)
        return self.layer_norm(value)


class _MeanVarianceNormLayer(nn.Module):
    def __init__(self, input_size: int, dtype: torch.dtype):
        super().__init__()
        self.register_buffer("global_mean", torch.zeros(input_size, dtype=dtype))
        self.register_buffer("global_invstd", torch.ones(input_size, dtype=dtype))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.global_mean) * self.global_invstd


class _NemoConvSubsampling(nn.Module):
    def __init__(self, config: Phi4MMAudioConfig, dtype: torch.dtype):
        super().__init__()
        if config.time_reduction != 8 or config.input_size != 80:
            raise ValueError("Phi4MM audio requires 80-bin features and time_reduction=8")
        hidden = config.attention_dim
        self.subsampling_factor = config.time_reduction
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden, 3, stride=2, padding=1, dtype=dtype),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, groups=hidden, dtype=dtype),
            nn.Conv2d(hidden, hidden, 1, dtype=dtype),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, groups=hidden, dtype=dtype),
            nn.Conv2d(hidden, hidden, 1, dtype=dtype),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden * 10, hidden, dtype=dtype)

    def forward(
        self, value: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        value = self.conv(value.unsqueeze(1))
        batch, channels, time, frequency = value.shape
        value = self.out(value.transpose(1, 2).reshape(batch, time, channels * frequency))
        if attention_mask is None:
            return value, None
        lengths = torch.ceil(attention_mask.sum(1) / self.subsampling_factor).to(dtype=torch.long)
        valid = torch.arange(time, device=value.device).unsqueeze(0) < lengths.unsqueeze(1)
        return value, valid.unsqueeze(1)


class _ConformerEncoder(nn.Module):
    def __init__(self, config: Phi4MMAudioConfig, dtype: torch.dtype):
        super().__init__()
        self.embed = _NemoConvSubsampling(config, dtype)
        self.relative_attention_bias_layer = _T5RelativeAttentionLogitBias(
            config.attention_heads, config.relative_attention_max_distance, dtype
        )
        self.encoders = nn.ModuleList([_ConformerEncoderLayer(config, dtype) for _ in range(config.num_blocks)])
        self.encoder_embedding = _MeanVarianceNormLayer(config.input_size, dtype)

    def _encode_chunk(self, value: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        relative_bias = self.relative_attention_bias_layer(value)
        # chunk_size=-1 in the released checkpoint denotes full-utterance
        # attention; padding still masks invalid keys for variable-length input.
        attention_mask = None
        if padding_mask is not None:
            attention_mask = padding_mask.expand(-1, value.shape[1], -1)
        for layer in self.encoders:
            value = layer(value, attention_mask, relative_bias)
        return value

    def forward(
        self, value: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        value = self.encoder_embedding(value)
        value, padding_mask = self.embed(value, attention_mask)
        original_length = value.shape[1]
        if original_length <= 500:
            return self._encode_chunk(value, padding_mask), padding_mask
        padded_length = ((original_length + 499) // 500) * 500
        value = F.pad(value, (0, 0, 0, padded_length - original_length))
        chunks = value.reshape(-1, 500, value.shape[-1])
        chunk_masks = None
        if padding_mask is not None:
            valid = F.pad(padding_mask.squeeze(1), (0, padded_length - original_length), value=False)
            chunk_masks = valid.reshape(-1, 1, 500)
        output = self._encode_chunk(chunks, chunk_masks)
        return output.reshape(value.shape[0], padded_length, -1)[:, :original_length], padding_mask


class Phi4MMAudioEmbedding(nn.Module):
    """Checkpoint-compatible Phi-4 audio tower with both official projectors."""

    def __init__(
        self,
        config: Phi4MMAudioConfig,
        language_hidden_size: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.encoder = _ConformerEncoder(config, dtype)
        self.audio_projection = nn.ModuleDict(
            {
                mode: nn.Sequential(
                    nn.Linear(config.attention_dim, language_hidden_size, dtype=dtype),
                    nn.GELU(),
                    nn.Linear(language_hidden_size, language_hidden_size, dtype=dtype),
                )
                for mode in ("speech", "vision")
            }
        )

    def forward(
        self,
        input_audio_embeds: torch.Tensor,
        audio_attention_mask: torch.Tensor | None,
        projection_mode: str = "speech",
    ) -> torch.Tensor:
        if projection_mode not in self.audio_projection:
            raise ValueError(f"unsupported Phi4MM audio projection mode {projection_mode!r}")
        dtype = self.audio_projection[projection_mode][0].weight.dtype
        features, _ = self.encoder(input_audio_embeds.to(dtype=dtype), audio_attention_mask)
        return self.audio_projection[projection_mode](features)
