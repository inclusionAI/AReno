"""Native Phi-4-Multimodal SigLIP vision tower and HD projector."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True, slots=True)
class Phi4MMVisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 448
    patch_size: int = 14
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "gelu_pytorch_tanh"
    feature_layer: int = -2
    crop_size: int = 448
    hd_transform_order: str = "sub_glb"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Phi4MMVisionConfig:
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


class Phi4MMVisionEmbeddings(nn.Module):
    def __init__(self, config: Phi4MMVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches_per_side = config.image_size // config.patch_size
        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            dtype=dtype,
        )
        self.position_embedding = nn.Embedding(
            self.num_patches_per_side**2,
            config.hidden_size,
            dtype=dtype,
        )

    def forward(self, pixel_values: torch.Tensor, patch_attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = (
            self.patch_embedding(pixel_values.to(dtype=self.patch_embedding.weight.dtype)).flatten(2).transpose(1, 2)
        )
        batch, patch_height, patch_width = patch_attention_mask.shape
        boundaries = torch.arange(
            1 / self.num_patches_per_side,
            1.0,
            1 / self.num_patches_per_side,
            device="cpu",
        )
        position_ids = torch.zeros((batch, patch_height * patch_width), dtype=torch.long, device="cpu")
        for row, mask in enumerate(patch_attention_mask.detach().to(device="cpu", dtype=torch.bool)):
            valid_height = int(mask[:, 0].sum().item())
            valid_width = int(mask[0].sum().item())
            if valid_height <= 0 or valid_width <= 0:
                raise ValueError("Phi4MM image attention mask must contain at least one valid patch")
            height_coords = torch.arange(valid_height, dtype=torch.float32) / valid_height
            width_coords = torch.arange(valid_width, dtype=torch.float32) / valid_width
            height_buckets = torch.bucketize(height_coords, boundaries, right=True)
            width_buckets = torch.bucketize(width_coords, boundaries, right=True)
            ids = (height_buckets[:, None] * self.num_patches_per_side + width_buckets).flatten()
            position_ids[row, mask.reshape(-1)] = ids
        return embeddings + self.position_embedding(position_ids.to(self.position_embedding.weight.device))


class Phi4MMVisionAttention(nn.Module):
    def __init__(self, config: Phi4MMVisionConfig, dtype: torch.dtype):
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("Phi4MM vision hidden_size must be divisible by num_attention_heads")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, dtype=dtype)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, dtype=dtype)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, dtype=dtype)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        batch, seqlen, hidden_size = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=query.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        output = torch.matmul(probabilities, value).transpose(1, 2).reshape(batch, seqlen, hidden_size)
        return self.out_proj(output)


class Phi4MMVisionMLP(nn.Module):
    def __init__(self, config: Phi4MMVisionConfig, dtype: torch.dtype):
        super().__init__()
        if config.hidden_act != "gelu_pytorch_tanh":
            raise ValueError(f"unsupported Phi4MM vision activation {config.hidden_act!r}")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, dtype=dtype)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class Phi4MMVisionEncoderLayer(nn.Module):
    def __init__(self, config: Phi4MMVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.self_attn = Phi4MMVisionAttention(config, dtype)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype)
        self.mlp = Phi4MMVisionMLP(config, dtype)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states), attention_mask)
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class Phi4MMVisionEncoder(nn.Module):
    def __init__(self, config: Phi4MMVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.layers = nn.ModuleList([Phi4MMVisionEncoderLayer(config, dtype) for _ in range(config.num_hidden_layers)])


class Phi4MMVisionPoolingHead(nn.Module):
    def __init__(self, config: Phi4MMVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.probe = nn.Parameter(torch.empty(1, 1, config.hidden_size, dtype=dtype))
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            batch_first=True,
            dtype=dtype,
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype)
        self.mlp = Phi4MMVisionMLP(config, dtype)


class Phi4MMVisionTransformer(nn.Module):
    """SigLIP NaViT module with checkpoint-compatible parameter names."""

    def __init__(self, config: Phi4MMVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.embeddings = Phi4MMVisionEmbeddings(config, dtype)
        self.encoder = Phi4MMVisionEncoder(config, dtype)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype)
        self.head = Phi4MMVisionPoolingHead(config, dtype)

    def patch_features(self, pixel_values: torch.Tensor, patch_attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values, patch_attention_mask)
        flat_mask = patch_attention_mask.reshape(patch_attention_mask.shape[0], -1).to(dtype=torch.bool)
        attention_mask = None if bool(flat_mask.all()) else flat_mask
        hidden_states_by_layer = [hidden_states]
        for layer in self.encoder.layers:
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states_by_layer.append(hidden_states)
        return hidden_states_by_layer[self.config.feature_layer]


class Phi4MMImageEmbedding(nn.Module):
    """Project processor crops into the language model's expanded image slots."""

    def __init__(self, config: Phi4MMVisionConfig, language_hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.img_processor = Phi4MMVisionTransformer(config, dtype)
        self.glb_GN = nn.Parameter(torch.zeros(1, 1, config.hidden_size, dtype=dtype))
        self.sub_GN = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size, dtype=dtype))
        self.img_projection = nn.Sequential(
            nn.Linear(config.hidden_size, language_hidden_size, dtype=dtype),
            nn.GELU(),
            nn.Linear(language_hidden_size, language_hidden_size, dtype=dtype),
        )

    def forward(
        self,
        input_image_embeds: torch.Tensor,
        image_sizes: torch.Tensor,
        image_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_image_embeds.ndim != 5:
            raise ValueError("Phi4MM input_image_embeds must have shape (images, crops, 3, H, W)")
        if image_sizes.numel() == 0 or image_attention_mask.numel() == 0:
            raise ValueError("Phi4MM vision inputs require image_sizes and image_attention_mask")
        image_count, max_crops = input_image_embeds.shape[:2]
        masks = image_attention_mask.to(device=input_image_embeds.device, dtype=torch.bool)
        features = self.img_processor.patch_features(input_image_embeds.flatten(0, 1), masks.flatten(0, 1))
        side = math.isqrt(int(features.shape[1]))
        if side * side != int(features.shape[1]):
            raise ValueError("Phi4MM vision patch count must form a square grid")
        features = features.view(image_count * max_crops, side, side, self.config.hidden_size)
        features = F.avg_pool2d(features.permute(0, 3, 1, 2), kernel_size=2, stride=2).permute(0, 2, 3, 1)
        pooled_side = int(features.shape[1])
        features = features.reshape(image_count, max_crops, pooled_side, pooled_side, self.config.hidden_size)

        projected = []
        sizes = image_sizes.reshape(-1, 2).detach().to(device="cpu", dtype=torch.long)
        for image_idx, (height_value, width_value) in enumerate(sizes.tolist()):
            crop_rows = int(height_value) // self.config.crop_size
            crop_cols = int(width_value) // self.config.crop_size
            local_crop_count = crop_rows * crop_cols
            if local_crop_count + 1 > max_crops:
                raise ValueError("Phi4MM image size requires more crops than input_image_embeds provides")

            global_image = features[image_idx, 0:1]
            global_separators = self.sub_GN.expand(1, pooled_side, 1, -1)
            global_image = torch.cat((global_image, global_separators), dim=2).reshape(1, -1, self.config.hidden_size)

            local_image = features[image_idx, 1 : local_crop_count + 1]
            local_image = (
                local_image.reshape(crop_rows, crop_cols, pooled_side, pooled_side, self.config.hidden_size)
                .permute(0, 2, 1, 3, 4)
                .reshape(1, crop_rows * pooled_side, crop_cols * pooled_side, self.config.hidden_size)
            )
            local_mask = masks[image_idx, 1 : local_crop_count + 1, 0::2, 0::2]
            local_mask = (
                local_mask.reshape(crop_rows, crop_cols, pooled_side, pooled_side)
                .permute(0, 2, 1, 3)
                .reshape(crop_rows * pooled_side, crop_cols * pooled_side)
            )
            useful_height = int(local_mask[:, 0].sum().item())
            useful_width = int(local_mask[0].sum().item())
            local_image = local_image[:, :useful_height, :useful_width]
            local_separators = self.sub_GN.expand(1, useful_height, 1, -1)
            local_image = torch.cat((local_image, local_separators), dim=2).reshape(1, -1, self.config.hidden_size)

            if self.config.hd_transform_order != "sub_glb":
                raise ValueError(f"unsupported Phi4MM hd_transform_order {self.config.hd_transform_order!r}")
            image_features = torch.cat((local_image, self.glb_GN, global_image), dim=1)
            projected.append(self.img_projection(image_features))
        return torch.cat(projected, dim=1).squeeze(0)


class Phi4MMExtendedEmbedding(nn.Module):
    """Checkpoint-compatible container for Phi multimodal embedding modules."""

    def __init__(
        self,
        config: Phi4MMVisionConfig | None,
        language_hidden_size: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        if config is not None:
            self.image_embed = Phi4MMImageEmbedding(config, language_hidden_size, dtype)
