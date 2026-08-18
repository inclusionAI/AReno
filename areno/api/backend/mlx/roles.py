"""MLX reference, critic, and reward role models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areno.api.backend.mlx.provider import MlxModelProvider, load_provider

_HEAD_WEIGHT_KEYS = (
    "score.weight",
    "v_head.weight",
    "classifier.weight",
    "reward_head.weight",
    "value_head.weight",
    "model.score.weight",
    "model.v_head.weight",
    "model.classifier.weight",
    "model.reward_head.weight",
    "model.value_head.weight",
)


@dataclass(slots=True)
class MlxRole:
    """One backend-owned auxiliary model role."""

    provider: MlxModelProvider
    module: Any | None = None
    optimizer: Any | None = None

    @property
    def model(self) -> Any:
        return self.provider.model

    @property
    def tokenizer(self) -> Any:
        return self.provider.tokenizer


def load_language_role(path: str) -> MlxRole:
    """Load a frozen language-model role."""

    provider = load_provider(path)
    provider.model.eval()
    return MlxRole(provider=provider)


def load_value_role(path: str, *, trainable: bool, learning_rate: float | None, reward: bool) -> MlxRole:
    """Load a language backbone plus scalar/classification value head."""

    import mlx.core as mx
    import mlx.optimizers as optim

    provider = load_provider(path)
    config = provider.config
    head_state = _load_head_state(path)
    if reward and head_state is None:
        raise KeyError("reward checkpoint must contain one of: " + ", ".join(_HEAD_WEIGHT_KEYS))
    hidden_size = _hidden_size(config)
    output_size = 1 if head_state is None else int(head_state[0].shape[0])
    module = _make_value_module(provider, hidden_size, output_size, head_state)
    optimizer = optim.AdamW(learning_rate=float(learning_rate or 1e-5)) if trainable else None
    if trainable:
        module.train()
        mx.eval(module.parameters(), optimizer.state)
    else:
        module.eval()
        mx.eval(module.parameters())
    return MlxRole(provider=provider, module=module, optimizer=optimizer)


def hidden_states(provider: MlxModelProvider, batch: dict[str, Any]):
    """Run a provider's text or multimodal body without the LM projection."""

    output = provider.forward_hidden_states(batch)
    if not hasattr(output, "shape") or len(output.shape) != 3:
        raise RuntimeError("MLX role model body must return [batch, sequence, hidden] states")
    return output


def role_output(role: MlxRole, batch: dict[str, Any]):
    """Return critic/reward logits from a value-bearing role."""

    if role.module is None:
        raise RuntimeError("MLX role does not contain a value head")
    return role.module(batch)


def _make_value_module(provider: MlxModelProvider, hidden_size: int, output_size: int, head_state):
    import mlx.core as mx
    import mlx.nn as nn

    class ValueModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = provider.model
            self.head = nn.Linear(hidden_size, output_size, bias=head_state is not None and head_state[1] is not None)
            if head_state is None:
                self.head.weight = mx.zeros_like(self.head.weight)
                bias = getattr(self.head, "bias", None)
                if bias is not None:
                    self.head.bias = mx.zeros_like(bias)
            else:
                weight, bias = head_state
                expected = tuple(self.head.weight.shape)
                if tuple(weight.shape) == expected:
                    self.head.weight = weight
                elif tuple(weight.T.shape) == expected:
                    self.head.weight = weight.T
                else:
                    raise ValueError(f"value head shape {tuple(weight.shape)} does not match {expected}")
                if getattr(self.head, "bias", None) is not None:
                    if bias is None:
                        raise KeyError("value head checkpoint is missing its bias")
                    self.head.bias = bias.reshape(self.head.bias.shape)

        def __call__(self, batch):
            return self.head(hidden_states(provider, batch))

    return ValueModel()


def _hidden_size(config: dict[str, Any]) -> int:
    for candidate in (config, config.get("text_config", {}), config.get("model_config", {})):
        for key in ("hidden_size", "dim", "model_dim"):
            value = candidate.get(key) if isinstance(candidate, dict) else None
            if value is not None:
                return int(value)
    raise KeyError("model config does not define hidden_size")


def _load_head_state(path: str):
    import mlx.core as mx
    from mlx_lm.utils import hf_repo_to_path

    model_path = Path(path)
    if not model_path.exists():
        model_path = hf_repo_to_path(path)
    index_path = model_path / "model.safetensors.index.json"
    weight_map = {}
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
    for key in _HEAD_WEIGHT_KEYS:
        files = [model_path / weight_map[key]] if key in weight_map else sorted(model_path.glob("*.safetensors"))
        for file in files:
            weights = mx.load(str(file))
            if key not in weights:
                continue
            weight = weights[key]
            if len(weight.shape) == 1:
                weight = weight.reshape(1, -1)
            bias = weights.get(key.removesuffix(".weight") + ".bias")
            return weight, bias
    return None


__all__ = ["MlxRole", "hidden_states", "load_language_role", "load_value_role", "role_output"]
