"""Model-family providers for the integrated MLX backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_FEATURE_METADATA = {
    "image_token_id",
    "modality_token_ids",
    "merge_size",
    "spatial_merge_size",
    "mrope_position_ids",
}
_TOWER_PARTS = {
    "audio_encoder",
    "audio_model",
    "audio_tower",
    "vision_encoder",
    "vision_model",
    "vision_tower",
}
_PROJECTOR_PARTS = {
    "audio_projector",
    "embed_audio",
    "embed_vision",
    "merger",
    "mm_projector",
    "multi_modal_projector",
    "multimodal_projector",
    "projector",
    "resampler",
    "vision_embedder",
    "vision_projector",
}


def load_provider(model_path: str, *, adapter_path: str | None = None) -> MlxModelProvider:
    """Load one text or multimodal MLX policy from a HuggingFace checkpoint."""

    config = _checkpoint_config(model_path)
    if _is_multimodal_config(config):
        return MlxVlmProvider.load(model_path, config=config, adapter_path=adapter_path)
    return MlxLmProvider.load(model_path, config=config, adapter_path=adapter_path)


class MlxModelProvider:
    """Uniform model operations shared by text and multimodal policies."""

    is_multimodal = False

    def __init__(self, model: Any, tokenizer: Any, processor: Any, config: dict[str, Any]) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

    @property
    def generation_model(self):
        return self.model

    def configure_trainability(self, optimizer_config: dict[str, Any]) -> None:
        del optimizer_config

    def prepare_generation_prompt(self, tokens: list[int], features: dict[str, Any] | None) -> dict[str, Any]:
        if features is not None:
            raise ValueError("text-only MLX checkpoints cannot consume multimodal prompt features")
        return {}

    def prepare_train_batch(self, batch: dict[str, Any], rows: list[Any]) -> dict[str, Any]:
        if any(row.features is not None for row in rows):
            raise ValueError("text-only MLX checkpoints cannot consume multimodal training features")
        return batch

    def prepare_score_batch(
        self,
        input_ids: Any,
        lengths: list[int],
        features: list[dict | None] | None,
    ) -> dict[str, Any]:
        if features is not None and any(feature is not None for feature in features):
            raise ValueError("text-only MLX checkpoints cannot consume multimodal prompt features")
        return {"input_ids": input_ids}

    def forward_logits(self, batch: dict[str, Any]):
        output = self.model(batch["input_ids"])
        logits = output.logits if hasattr(output, "logits") else output
        return logits[:, :-1, :]

    def selected_token_logprobs(self, batch: dict[str, Any], targets: Any, *, chunk_size: int):
        from areno.api.backend.mlx.numerics import selected_token_logprobs

        del chunk_size
        return selected_token_logprobs(self.forward_logits(batch), targets)

    def forward_hidden_states(self, batch: dict[str, Any]):
        body = getattr(self.model, "model", None)
        if body is None or not callable(body):
            raise RuntimeError(f"{type(self.model).__name__} does not expose a callable hidden-state model body")
        output = body(batch["input_ids"])
        return output[0] if isinstance(output, tuple) else output

    def save(self, destination: Path) -> None:
        from mlx_lm.utils import save_config, save_model

        save_model(destination, self.model)
        save_config(dict(self.config), config_path=destination / "config.json")
        self.tokenizer.save_pretrained(destination)


class MlxLmProvider(MlxModelProvider):
    """MLX-LM provider for text-only checkpoints."""

    @classmethod
    def load(
        cls,
        model_path: str,
        *,
        config: dict[str, Any],
        adapter_path: str | None,
    ) -> MlxLmProvider:
        from mlx_lm import load

        model, tokenizer, loaded_config = load(model_path, adapter_path=adapter_path, return_config=True)
        return cls(model, tokenizer, None, loaded_config or config)


class MlxVlmProvider(MlxModelProvider):
    """MLX-VLM provider for image, audio, video, and omni checkpoints."""

    is_multimodal = True

    @classmethod
    def load(
        cls,
        model_path: str,
        *,
        config: dict[str, Any],
        adapter_path: str | None,
    ) -> MlxVlmProvider:
        try:
            from mlx_vlm import load
        except ImportError as exc:
            raise RuntimeError("multimodal MLX checkpoints require mlx-vlm>=0.6.14") from exc

        _expose_transformers_pil_processor(config)
        model, processor = load(model_path, adapter_path=adapter_path, backend="pil")
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        try:
            processor._areno_return_tensors = "np"
        except (AttributeError, TypeError):
            pass
        return cls(model, tokenizer, processor, config)

    @property
    def generation_model(self):
        model = getattr(self.model, "language_model", None)
        if model is None:
            raise RuntimeError(f"{type(self.model).__name__} does not expose language_model for decode")
        return model

    def configure_trainability(self, optimizer_config: dict[str, Any]) -> None:
        unfreeze_tower = bool(optimizer_config.get("unfreeze_multimodal_tower", False))
        unfreeze_projector = bool(optimizer_config.get("unfreeze_multimodal_projector", False))

        def configure(path: str, module: Any) -> None:
            group = parameter_group(path)
            if group == "tower" and not unfreeze_tower:
                module.freeze()
            elif group == "projector" and not unfreeze_projector:
                module.freeze()

        self.model.apply_to_modules(configure)

    def prepare_generation_prompt(self, tokens: list[int], features: dict[str, Any] | None) -> dict[str, Any]:
        import mlx.core as mx

        input_ids = mx.array([tokens], dtype=mx.int32)
        prepared = _prepare_feature_dict(features)
        pixel_values = prepared.pop("pixel_values", None)
        attention_mask = prepared.pop("attention_mask", None)
        embeddings = self.model.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            mask=attention_mask,
            **prepared,
        )
        embedding_values = embeddings.to_dict() if hasattr(embeddings, "to_dict") else {"inputs_embeds": embeddings}
        prompt_kwargs = {**prepared, **{key: value for key, value in embedding_values.items() if value is not None}}
        arrays = [value for value in prompt_kwargs.values() if isinstance(value, mx.array)]
        if arrays:
            mx.eval(*arrays)
        return prompt_kwargs

    def prepare_train_batch(self, batch: dict[str, Any], rows: list[Any]) -> dict[str, Any]:
        import mlx.core as mx

        width = int(batch["input_ids"].shape[1])
        lengths = mx.array([len(row.tokens) for row in rows], dtype=mx.int32)
        batch["attention_mask"] = mx.arange(width)[None, :] < lengths[:, None]
        feature_rows = [row.features for row in rows]
        if not any(feature is not None for feature in feature_rows):
            return batch
        if any(not isinstance(feature, dict) for feature in feature_rows):
            raise ValueError("multimodal MLX microbatches cannot mix media and text-only rows")
        batch.update(_collate_feature_rows(feature_rows))
        return batch

    def prepare_score_batch(
        self,
        input_ids: Any,
        lengths: list[int],
        features: list[dict | None] | None,
    ) -> dict[str, Any]:
        import mlx.core as mx

        width = int(input_ids.shape[1])
        batch = {
            "input_ids": input_ids,
            "attention_mask": mx.arange(width)[None, :] < mx.array(lengths, dtype=mx.int32)[:, None],
        }
        if features is None or not any(feature is not None for feature in features):
            return batch
        if len(features) != len(lengths) or any(not isinstance(feature, dict) for feature in features):
            raise ValueError("multimodal MLX scoring requires one feature dictionary per row")
        batch.update(_collate_feature_rows(features))
        return batch

    def forward_logits(self, batch: dict[str, Any]):
        import mlx.core as mx

        input_ids = batch["input_ids"][:, :-1]
        attention_mask = batch["attention_mask"][:, :-1]
        pixel_values = batch.get("pixel_values")
        kwargs = {
            key: value
            for key, value in batch.items()
            if key
            not in {
                "input_ids",
                "pixel_values",
                "attention_mask",
                "prompt_mask",
                "loss_mask",
                "response_mask",
                "old_logprobs",
                "advantages",
                "ref_logprobs",
                "returns",
                "values",
            }
        }
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        model_mask = None if model_type == "gemma4_unified" else attention_mask
        output = self.model(input_ids, pixel_values, model_mask, **kwargs)
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits.astype(mx.float32)
        target_length = int(input_ids.shape[1])
        if logits.shape[1] < target_length:
            logits = mx.pad(logits, ((0, 0), (0, target_length - logits.shape[1]), (0, 0)))
        elif logits.shape[1] > target_length:
            logits = logits[:, -target_length:, :]
        return logits

    def forward_hidden_states(self, batch: dict[str, Any]):
        input_ids = batch["input_ids"][:, :-1]
        attention_mask = batch["attention_mask"][:, :-1]
        prepared = {
            key: value for key, value in batch.items() if key not in {"input_ids", "attention_mask", "pixel_values"}
        }
        embedding_output = self.model.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=batch.get("pixel_values"),
            mask=attention_mask,
            **prepared,
        )
        inputs_embeds = (
            embedding_output.inputs_embeds if hasattr(embedding_output, "inputs_embeds") else embedding_output
        )
        kwargs = (
            {
                key: value
                for key, value in embedding_output.to_dict().items()
                if key != "inputs_embeds" and value is not None
            }
            if hasattr(embedding_output, "to_dict")
            else {}
        )
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        model_mask = None if model_type == "gemma4_unified" else attention_mask
        language_model = self.generation_model
        output = language_model(
            input_ids,
            inputs_embeds=inputs_embeds,
            mask=model_mask,
            skip_logits=True,
            return_hidden=True,
            **kwargs,
        )
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is not None:
            return hidden_states[-1] if isinstance(hidden_states, list | tuple) else hidden_states
        body = getattr(language_model, "model", None)
        if body is None or not callable(body):
            raise RuntimeError(f"{type(language_model).__name__} does not expose hidden states for training")
        output = body(input_ids, inputs_embeds=inputs_embeds, **kwargs)
        return output[0] if isinstance(output, tuple) else output

    def selected_token_logprobs(self, batch: dict[str, Any], targets: Any, *, chunk_size: int):
        import mlx.core as mx

        from areno.api.backend.mlx.numerics import (
            chunked_linear_selected_token_logprobs,
            selected_token_logprobs,
        )

        hidden_states = self.forward_hidden_states(batch)
        language_model = self.generation_model
        head = getattr(language_model, "lm_head", None)
        embedding = getattr(getattr(language_model, "model", None), "embed_tokens", None)
        weight = getattr(head, "weight", None)
        bias = getattr(head, "bias", None)
        if weight is None:
            weight = getattr(embedding, "weight", None)
            bias = None
        if weight is not None:
            return chunked_linear_selected_token_logprobs(
                hidden_states,
                targets,
                weight,
                bias=bias,
                chunk_size=chunk_size,
            )

        project = getattr(language_model, "speculative_logits_from_hidden", None)
        if not callable(project):
            if callable(head):
                project = head
            else:
                project = getattr(embedding, "as_linear", None)
        if not callable(project):
            return super().selected_token_logprobs(batch, targets, chunk_size=chunk_size)

        def checkpointed_logprobs(target_chunk):
            def chunk_logprobs(hidden_chunk):
                return selected_token_logprobs(project(hidden_chunk), target_chunk)

            return mx.checkpoint(chunk_logprobs)

        selected = []
        step = 16
        for start in range(0, int(hidden_states.shape[1]), step):
            end = min(start + step, int(hidden_states.shape[1]))
            target_chunk = mx.stop_gradient(targets[:, start:end])
            selected.append(checkpointed_logprobs(target_chunk)(hidden_states[:, start:end, :]))
        return mx.concatenate(selected, axis=1)

    def save(self, destination: Path) -> None:
        from mlx_vlm.utils import save_config, save_weights

        save_weights(destination, self.model)
        save_config(dict(self.config), destination / "config.json")
        self.processor.save_pretrained(destination)


def parameter_group(path: str) -> str:
    """Classify one flattened MLX parameter/module path."""

    parts = set(path.lower().split("."))
    if parts & _PROJECTOR_PARTS:
        return "projector"
    if parts & _TOWER_PARTS:
        return "tower"
    return "model"


def _checkpoint_config(model_path: str) -> dict[str, Any]:
    path = Path(model_path) / "config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path, trust_remote_code=True).to_dict()


def _is_multimodal_config(config: dict[str, Any]) -> bool:
    return any(config.get(key) is not None for key in ("vision_config", "audio_config"))


def _expose_transformers_pil_processor(config: dict[str, Any]) -> None:
    """Work around the missing Idefics3 PIL lazy export in Transformers 5.15."""

    if config.get("model_type") != "idefics3":
        return
    import transformers.models.idefics3 as idefics3
    from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil

    if getattr(idefics3, "Idefics3ImageProcessorPil", None) is not Idefics3ImageProcessorPil:
        idefics3.Idefics3ImageProcessorPil = Idefics3ImageProcessorPil


def _prepare_feature_dict(features: dict[str, Any] | None) -> dict[str, Any]:
    if not features:
        return {}
    return {
        key: _to_mlx(value) for key, value in features.items() if key not in _FEATURE_METADATA and value is not None
    }


def _to_mlx(value: Any):
    import mlx.core as mx

    if isinstance(value, mx.array):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray | int | float | bool):
        return mx.array(value)
    if isinstance(value, list):
        return [_to_mlx(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_mlx(item) for item in value)
    return value


def _squeeze_leading_batch(value: Any):
    value = _to_mlx(value)
    if isinstance(value, list | tuple) and len(value) == 1:
        return value[0]
    if hasattr(value, "ndim") and value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


def _collate_feature_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import mlx.core as mx

    keys = set(rows[0]) - _FEATURE_METADATA
    for row in rows[1:]:
        if (set(row) - _FEATURE_METADATA) != keys:
            raise ValueError("multimodal MLX rows in one microbatch must expose the same feature keys")
    result: dict[str, Any] = {}
    for key in keys:
        values = [_squeeze_leading_batch(row[key]) for row in rows]
        if not isinstance(values[0], mx.array):
            result[key] = values[0]
            continue
        if key in {"image_grid_thw", "video_grid_thw"}:
            values = [value[None, :] if value.ndim == 1 else value for value in values]
            result[key] = mx.concatenate(values, axis=0)
            continue
        try:
            result[key] = mx.stack(values)
        except ValueError:
            first = values[0]
            if first.ndim > 1 and all(
                value.ndim == first.ndim and value.shape[1:] == first.shape[1:] for value in values
            ):
                result[key] = mx.concatenate(values, axis=0)
            else:
                raise
    return result


__all__ = ["MlxModelProvider", "load_provider", "parameter_group"]
