"""Helpers for multimodal token/feature alignment."""

from __future__ import annotations

import base64
import io
import threading
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.request import urlopen

import torch

from areno.api.tokenizer import apply_chat_template_with_options, normalize_token_ids

_MODALITIES = ("image", "video", "audio")
_VIDEO_FPS_PATCH_LOCK = threading.Lock()
_VIDEO_FPS_PATCHED = False


def record_has_multimodal(record: dict[str, Any]) -> bool:
    """Return true when a loader row contains raw image, video, or audio input."""

    if any(
        record.get(f"{modality}_base64") is not None or record.get(f"{modality}s_base64") is not None
        for modality in _MODALITIES
    ):
        return True
    messages = record.get("messages")
    if not isinstance(messages, list):
        return False
    return any(_content_has_multimodal(message.get("content")) for message in messages if isinstance(message, dict))


def record_has_image(record: dict[str, Any]) -> bool:
    """Backward-compatible alias for loaders that previously checked image rows."""

    return record_has_multimodal(record)


def encode_multimodal_prompt(
    tokenizer: Any,
    processor: Any,
    record: dict[str, Any],
    *,
    prompt_key: str = "prompt",
) -> tuple[list[int], dict[str, Any] | None]:
    """Encode a loader row with base64 image fields into tokens and features.

    Dataset loaders stay model-agnostic and return raw ``image_base64`` plus
    text fields. This helper is the model boundary: it uses the current
    checkpoint processor to produce token ids, image grids, pixel values, and
    Qwen-style expanded image-token slots.
    """

    if processor is None:
        raise ValueError("multimodal rows require a checkpoint processor")
    if _record_requires_native_multimodal(record):
        messages = record.get("messages")
        if not isinstance(messages, list):
            prompt = str(record.get(prompt_key, ""))
            content = _record_multimodal_parts(record)
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
        return encode_processor_messages(processor, messages, tools=record.get("tools"))
    images = _load_record_images(record)
    if isinstance(record.get("messages"), list):
        messages = _normalize_image_messages(record["messages"])
    else:
        prompt = str(record.get(prompt_key, ""))
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": prompt}],
            }
        ]
    text = _processor_chat_text(processor, messages, tools=record.get("tools"))
    encoded = _encode_text_and_images(tokenizer, processor, text, images)
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise ValueError("processor did not return input_ids for image row")
    features = {
        key: value
        for key, value in dict(encoded).items()
        if key not in {"input_ids", "attention_mask", "token_type_ids"}
    }
    image_token_id = _image_token_id(tokenizer, processor)
    if image_token_id is not None:
        features["image_token_id"] = image_token_id
    if "target_sizes" in features and "num_patches_per_image" in features:
        # MiniCPM-V processors replace each source `<image>` marker with the
        # complete visual-token span, including optional high-resolution slices.
        features["processor_expanded_image_tokens"] = True
        image_processor = getattr(processor, "image_processor", None)
        downsample_mode = getattr(image_processor, "downsample_mode", None)
        if downsample_mode is not None:
            features["downsample_mode"] = str(downsample_mode)
    tokens = normalize_token_ids(input_ids[0].tolist())
    counts = image_token_counts_from_features(features)
    if counts:
        if image_token_id is None:
            raise ValueError("image rows require an image token id from tokenizer or processor")
        tokens, _ = expand_image_tokens(tokens, image_token_id=image_token_id, image_token_counts=counts)
        mrope_position_ids = mrope_position_ids_from_image_grid(
            tokens,
            image_token_id=image_token_id,
            features=features,
        )
        if mrope_position_ids is not None:
            features["mrope_position_ids"] = mrope_position_ids
    return tokens, features or None


def encode_processor_messages(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    tools: Any = None,
) -> tuple[list[int], dict[str, Any] | None]:
    """Use a native multimodal processor to load media and expand soft-token slots."""

    normalized = _normalize_multimodal_messages(messages)
    identity = f"{type(processor).__module__}.{type(processor).__name__}".lower()
    if "phi4mm" in identity:
        return _encode_phi4mm_messages(processor, normalized, tools=tools)
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": getattr(processor, "_areno_return_tensors", "pt"),
    }
    if tools:
        kwargs["tools"] = tools
    _ensure_gemma4_torchvision_video_fps(processor)
    encoded = apply_chat_template_with_options(processor, normalized, **kwargs)
    if not isinstance(encoded, Mapping) or encoded.get("input_ids") is None:
        raise ValueError("multimodal processor did not return input_ids")
    input_ids = encoded["input_ids"]
    tokens = normalize_token_ids(input_ids[0].tolist())
    features = {
        key: value for key, value in encoded.items() if key not in {"input_ids", "attention_mask", "token_type_ids"}
    }
    token_ids = modality_token_ids(processor)
    if token_ids:
        features["modality_token_ids"] = token_ids
        if token_ids.get("image") is not None:
            features["image_token_id"] = token_ids["image"]
    return tokens, features or None


def _encode_phi4mm_messages(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    tools: Any = None,
) -> tuple[list[int], dict[str, Any] | None]:
    """Bridge structured API messages to the released Phi-4 processor API."""

    images = []
    audios = []
    rendered_messages = []
    for message in messages:
        rendered = dict(message)
        content = rendered.get("content")
        if isinstance(content, list):
            pieces = []
            for part in content:
                if not isinstance(part, dict):
                    pieces.append(str(part))
                    continue
                kind = str(part.get("type", ""))
                if kind == "text":
                    pieces.append(str(part.get("text", "")))
                elif kind == "image":
                    images.append(_load_phi4mm_image(part.get("url")))
                    pieces.append(f"<|image_{len(images)}|>")
                elif kind == "audio":
                    audios.append(_load_phi4mm_audio(part.get("url")))
                    pieces.append(f"<|audio_{len(audios)}|>")
                else:
                    raise ValueError(f"Phi4MM does not support multimodal content type {kind!r}")
            rendered["content"] = "".join(pieces)
        rendered_messages.append(rendered)
    template_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        template_kwargs["tools"] = tools
    prompt = apply_chat_template_with_options(processor.tokenizer, rendered_messages, **template_kwargs)
    if prompt.endswith("<|endoftext|>"):
        prompt = prompt.removesuffix("<|endoftext|>")
    encoded = processor(
        text=prompt,
        images=images or None,
        audios=audios or None,
        return_tensors=getattr(processor, "_areno_return_tensors", "pt"),
    )
    input_ids = encoded["input_ids"]
    tokens = normalize_token_ids(input_ids[0].tolist())
    features = {
        key: value for key, value in encoded.items() if key not in {"input_ids", "attention_mask", "token_type_ids"}
    }
    token_ids = modality_token_ids(processor)
    features["modality_token_ids"] = token_ids
    features["image_token_id"] = token_ids["image"]
    features["audio_token_id"] = token_ids["audio"]
    return tokens, features


def _load_phi4mm_image(reference: Any) -> Any:
    if not isinstance(reference, str) or not reference:
        raise ValueError("Phi4MM image content requires a URL or data URI")
    if reference.startswith("data:"):
        return _load_base64_image(reference)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("Phi4MM image input requires Pillow") from exc
    if reference.startswith(("http://", "https://")):
        with urlopen(reference, timeout=30) as response:  # noqa: S310
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    return Image.open(reference).convert("RGB")


def _load_phi4mm_audio(reference: Any) -> tuple[Any, int]:
    if not isinstance(reference, str) or not reference:
        raise ValueError("Phi4MM audio content requires a URL or data URI")
    try:
        import soundfile
    except ImportError as exc:
        raise ValueError("Phi4MM audio input requires soundfile") from exc
    if reference.startswith("data:"):
        _, _, payload = reference.partition(",")
        return soundfile.read(io.BytesIO(base64.b64decode(payload)))
    if reference.startswith(("http://", "https://")):
        with urlopen(reference, timeout=30) as response:  # noqa: S310
            return soundfile.read(io.BytesIO(response.read()))
    return soundfile.read(reference)


def _ensure_gemma4_torchvision_video_fps(processor: Any) -> None:
    """Backfill FPS metadata omitted by torchvision for some browser videos."""

    global _VIDEO_FPS_PATCHED
    identity = f"{type(processor).__module__}.{type(processor).__name__}".lower()
    if "gemma4" not in identity or _VIDEO_FPS_PATCHED:
        return
    with _VIDEO_FPS_PATCH_LOCK:
        if _VIDEO_FPS_PATCHED:
            return
        try:
            from transformers import video_utils
        except (ImportError, AttributeError):
            return
        torchvision_io = getattr(video_utils, "torchvision_io", None)
        original = getattr(torchvision_io, "read_video", None)
        if original is None or getattr(original, "_areno_video_fps_fallback", False):
            _VIDEO_FPS_PATCHED = original is not None
            return

        def read_video_with_fps(video_path, *args, **kwargs):
            video, audio, info = original(video_path, *args, **kwargs)
            info = dict(info or {})
            if not info.get("video_fps"):
                info["video_fps"] = _read_torchvision_fps(torchvision_io, video_path)
            return video, audio, info

        read_video_with_fps._areno_video_fps_fallback = True
        torchvision_io.read_video = read_video_with_fps
        _VIDEO_FPS_PATCHED = True


def _read_torchvision_fps(torchvision_io: Any, video_path: Any) -> float:
    try:
        _, fps = torchvision_io.read_video_timestamps(video_path, pts_unit="sec")
        if fps is not None and float(fps) > 0:
            return float(fps)
    except Exception:  # noqa: BLE001
        pass
    return 30.0


def modality_token_ids(processor: Any) -> dict[str, int]:
    """Return the soft-token id for each modality exposed by a processor."""

    result: dict[str, int] = {}
    for modality in _MODALITIES:
        value = getattr(processor, f"{modality}_token_id", None)
        if isinstance(value, int) and value >= 0:
            result[modality] = int(value)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        for modality, token in (("image", "<|endoftext10|>"), ("audio", "<|endoftext11|>")):
            if modality in result:
                continue
            identity = f"{type(processor).__module__}.{type(processor).__name__}".lower()
            if "phi4mm" not in identity:
                continue
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                result[modality] = token_id
    return result


def _record_requires_native_multimodal(record: dict[str, Any]) -> bool:
    if any(record.get(key) is not None for key in ("audio_base64", "audios_base64", "video_base64", "videos_base64")):
        return True
    messages = record.get("messages")
    return isinstance(messages, list) and any(
        _content_has_audio_or_video(message.get("content")) for message in messages if isinstance(message, dict)
    )


def _record_multimodal_parts(record: dict[str, Any]) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = []
    mime_defaults = {"image": "image/png", "audio": "audio/wav", "video": "video/mp4"}
    for modality in _MODALITIES:
        values = record.get(f"{modality}s_base64")
        if values is None:
            values = record.get(f"{modality}_base64")
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        for value in values:
            ref = str(value)
            if not ref.startswith("data:"):
                ref = f"data:{mime_defaults[modality]};base64,{ref}"
            parts.append({"type": modality, "url": ref})
    return parts


def _normalize_multimodal_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = [_normalize_multimodal_content_part(part) for part in content]
        normalized.append(item)
    return normalized


def _normalize_multimodal_content_part(part: Any) -> Any:
    if not isinstance(part, dict):
        return part
    kind = str(part.get("type", ""))
    if kind == "input_audio":
        value = part.get("input_audio")
        if not isinstance(value, dict) or not value.get("data"):
            raise ValueError("input_audio content must include base64 data")
        fmt = str(value.get("format") or "wav").lower()
        mime = {"wav": "audio/wav", "mp3": "audio/mpeg"}.get(fmt, f"audio/{fmt}")
        ref = str(value["data"])
        if not ref.startswith("data:"):
            ref = f"data:{mime};base64,{ref}"
        return {"type": "audio", "url": ref}
    if kind in _MODALITIES and part.get("url") is None and part.get(kind) is not None:
        normalized = dict(part)
        normalized["url"] = normalized.pop(kind)
        return normalized
    if not kind.endswith("_url"):
        return part
    modality = kind.removesuffix("_url")
    if modality not in _MODALITIES:
        return part
    value = part.get(kind)
    ref = value.get("url") if isinstance(value, dict) else value
    if not isinstance(ref, str) or not ref:
        raise ValueError(f"{kind} content must include a url")
    normalized = dict(part)
    normalized.pop(kind, None)
    normalized["type"] = modality
    normalized["url"] = ref
    return normalized


def _content_has_multimodal(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type", ""))
        if kind == "input_audio" or kind.removesuffix("_url") in _MODALITIES:
            return True
    return False


def _content_has_audio_or_video(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type", ""))
        if kind == "input_audio" or kind.removesuffix("_url") in {"audio", "video"}:
            return True
    return False


def _load_record_images(record: dict[str, Any]) -> list[Any]:
    values = record.get("images_base64")
    if values is None:
        values = record.get("image_base64")
    if values is None:
        raise ValueError("multimodal row must contain image_base64 or images_base64")
    if isinstance(values, str):
        values = [values]
    return [_load_base64_image(value) for value in values]


def _load_base64_image(value: str) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("image_base64 rows require Pillow") from exc
    payload = str(value)
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _normalize_image_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = [_normalize_image_content_part(part) for part in content]
        normalized.append(item)
    return normalized


def _normalize_image_content_part(part: Any) -> Any:
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return part
    image_url = part.get("image_url")
    if not isinstance(image_url, dict) or "url" not in image_url:
        raise ValueError("image_url content must be an object with a url field")
    normalized = dict(part)
    normalized["type"] = "image"
    normalized["image"] = image_url["url"]
    normalized.pop("image_url", None)
    return normalized


def _processor_chat_text(processor: Any, messages: list[dict[str, Any]], *, tools: Any = None) -> str:
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if callable(apply_chat_template):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        rendered = apply_chat_template_with_options(processor, messages, **kwargs)
        if isinstance(rendered, str):
            return rendered
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        return apply_chat_template_with_options(
            tokenizer,
            _messages_for_text_fallback(messages),
            **kwargs,
        )
    if tools:
        raise ValueError("image input with tools requires a processor or tokenizer chat template that supports tools")
    return _messages_fallback_text(_messages_for_text_fallback(messages))


def _messages_for_text_fallback(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image":
                        parts.append({"type": "image"})
                    elif part.get("type") == "text":
                        parts.append({"type": "text", "text": str(part.get("text", ""))})
                else:
                    parts.append({"type": "text", "text": str(part)})
            item["content"] = parts
        out.append(item)
    return out


def _messages_fallback_text(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text = "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        else:
            text = str(content or "")
        lines.append(f"{message['role']}: {text}")
    lines.append("assistant:")
    return "\n".join(lines)


def _encode_text_and_images(tokenizer: Any, processor: Any, text: str, images: list[Any]) -> dict[str, Any]:
    return_tensors = getattr(processor, "_areno_return_tensors", "pt")
    try:
        return dict(processor(text=[text], images=images, return_tensors=return_tensors))
    except TypeError as exc:
        if "images" not in str(exc):
            raise
    image_processor = _image_processor_from_processor(processor)
    text_encoded = tokenizer([text], return_tensors=return_tensors)
    image_encoded = image_processor(images=images, return_tensors=return_tensors)
    encoded = dict(image_encoded)
    encoded["input_ids"] = text_encoded["input_ids"]
    if text_encoded.get("attention_mask") is not None:
        encoded["attention_mask"] = text_encoded["attention_mask"]
    return encoded


def _image_processor_from_processor(processor: Any):
    nested = getattr(processor, "image_processor", None)
    if nested is not None:
        return nested
    try:
        from transformers import AutoImageProcessor
    except ImportError as exc:
        raise ValueError("image_base64 rows require transformers AutoImageProcessor") from exc
    name_or_path = getattr(processor, "name_or_path", None)
    if not name_or_path:
        raise ValueError("image_base64 rows require an image processor")
    return AutoImageProcessor.from_pretrained(name_or_path, trust_remote_code=True)


def _image_token_id(tokenizer: Any, processor: Any) -> int | None:
    for obj in (processor, tokenizer):
        for attr in ("image_token_id", "image_token_index", "special_image_token_id"):
            value = getattr(obj, attr, None)
            if isinstance(value, int):
                return int(value)
        token = getattr(obj, "image_token", None)
        if isinstance(token, str):
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(convert):
                token_id = convert(token)
                if isinstance(token_id, int) and token_id >= 0:
                    return int(token_id)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        for token in ("<|image_pad|>", "<|image|>", "<image>", "<|endoftext10|>"):
            token_id = convert(token)
            if isinstance(token_id, int) and token_id >= 0:
                return int(token_id)
    return None


def image_token_counts_from_features(features: dict[str, Any] | None) -> list[int]:
    """Return merged visual-token counts for each image in ``features``.

    Qwen-style image processors place one placeholder token per image in the
    rendered text, while the vision tower emits one embedding per merged patch
    group. The language input must therefore repeat each image placeholder by
    the corresponding merged visual-token count before the model can replace
    those token embeddings with visual embeddings.
    """

    if not features:
        return []
    if features.get("processor_expanded_image_tokens"):
        return []
    grid = features.get("image_grid_thw")
    if grid is None:
        target_sizes = features.get("target_sizes")
        patches_per_image = features.get("num_patches_per_image")
        if target_sizes is None or patches_per_image is None:
            return []
        if not isinstance(target_sizes, torch.Tensor):
            target_sizes = torch.as_tensor(target_sizes)
        target_sizes = target_sizes.detach().cpu().to(dtype=torch.long).reshape(-1, 2)
        divisor = 4 if str(features.get("downsample_mode", "16x")) == "4x" else 16
        counts = []
        offset = 0
        for patch_count in patches_per_image:
            patch_count = int(patch_count)
            rows = target_sizes[offset : offset + patch_count]
            if rows.shape[0] != patch_count:
                raise ValueError("MiniCPM target_sizes does not match num_patches_per_image")
            image_count = 0
            for row in rows:
                patches = int(row[0]) * int(row[1])
                if patches % divisor:
                    raise ValueError(
                        f"MiniCPM target_sizes patches={patches} is not divisible by downsample divisor {divisor}"
                    )
                image_count += patches // divisor
            counts.append(image_count)
            offset += patch_count
        if offset != int(target_sizes.shape[0]):
            raise ValueError("MiniCPM num_patches_per_image does not cover target_sizes")
        return counts
    if not isinstance(grid, torch.Tensor):
        grid = torch.as_tensor(grid)
    grid = grid.detach().cpu().to(dtype=torch.long).reshape(-1, 3)
    grid_rows = _array_to_numpy(grid).astype("int64", copy=False).reshape(-1, 3)
    merge = int(features.get("spatial_merge_size", features.get("merge_size", 2)) or 2)
    merge_unit = merge * merge
    counts: list[int] = []
    for t, h, w in grid_rows.tolist():
        patches = int(t) * int(h) * int(w)
        if patches % merge_unit:
            raise ValueError(f"image_grid_thw patches={patches} is not divisible by spatial_merge_size**2={merge_unit}")
        counts.append(patches // merge_unit)
    return counts


def expand_image_tokens(
    tokens: Sequence[int],
    *,
    image_token_id: int | None,
    image_token_counts: Sequence[int],
    aligned_sequences: dict[str, Sequence[Any]] | None = None,
) -> tuple[list[int], dict[str, list[Any]]]:
    """Expand one image placeholder per image into merged visual-token slots.

    ``aligned_sequences`` may contain masks or per-token arrays with the same
    length as ``tokens``; each value at an expanded image-token position is
    repeated by the same visual-token count.
    """

    out_tokens: list[int] = []
    out_aligned = {name: [] for name in (aligned_sequences or {})}
    count_idx = 0
    image_token_id = int(image_token_id) if image_token_id is not None else None
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        repeat = 1
        if image_token_id is not None and int(token) == image_token_id and count_idx < len(image_token_counts):
            count = int(image_token_counts[count_idx])
            if _has_existing_image_span(tokens, image_token_id, idx, count):
                repeat = count
                count_idx += 1
                out_tokens.extend([int(token)] * repeat)
                for name, values in (aligned_sequences or {}).items():
                    out_aligned[name].extend(values[idx : idx + repeat])
                idx += repeat
                continue
            repeat = count
            count_idx += 1
        out_tokens.extend([int(token)] * repeat)
        for name, values in (aligned_sequences or {}).items():
            out_aligned[name].extend([values[idx]] * repeat)
        idx += 1
    if count_idx != len(image_token_counts):
        raise ValueError(
            "image feature count does not match prompt image token count: "
            f"features={len(image_token_counts)} prompt_tokens={count_idx}"
        )
    return out_tokens, out_aligned


def _has_existing_image_span(tokens: Sequence[int], image_token_id: int, start: int, count: int) -> bool:
    """Return true if a processor already expanded this image token span."""

    end = start + count
    if count <= 1 or end > len(tokens):
        return False
    return all(int(token) == image_token_id for token in tokens[start:end])


def mrope_position_ids_from_image_grid(
    tokens: Sequence[int],
    *,
    image_token_id: int | None,
    features: dict[str, Any] | None,
) -> Any | None:
    """Build Qwen3.5-VL MRoPE ids for tokens after image-token expansion.

    This follows the image-only fast path used by SGLang: text spans advance a
    scalar position, image spans use (t, h, w) grid positions at merged-patch
    resolution, and the following text resumes at max(t, h, w) after the image.
    """

    if image_token_id is None or not features:
        return None
    grid = features.get("image_grid_thw")
    if grid is None:
        return None
    grid_rows = _array_to_numpy(grid).astype("int64", copy=False).reshape(-1, 3)
    merge = int(features.get("spatial_merge_size", features.get("merge_size", 2)) or 2)
    image_token_id = int(image_token_id)
    token_list = [int(token) for token in tokens]
    import numpy as np

    segments: list[np.ndarray] = []
    st = 0
    next_pos = 0
    for t, h, w in grid_rows.tolist():
        t, h, w = int(t), int(h), int(w)
        if h % merge or w % merge:
            raise ValueError("image_grid_thw height/width must be divisible by spatial_merge_size")
        llm_t, llm_h, llm_w = t, h // merge, w // merge
        count = llm_t * llm_h * llm_w
        try:
            start = _find_image_span(token_list, image_token_id, st, count)
        except ValueError as exc:
            raise ValueError("image_grid_thw count does not match expanded image tokens") from exc
        text_len = start - st
        if text_len > 0:
            segments.append(np.broadcast_to(np.arange(text_len, dtype=np.int64), (3, text_len)) + next_pos)
            next_pos += text_len
        end = start + count
        t_index = np.broadcast_to(np.arange(llm_t, dtype=np.int64)[:, None], (llm_t, llm_h * llm_w)).reshape(-1)
        h_index = np.broadcast_to(np.arange(llm_h, dtype=np.int64)[None, :, None], (llm_t, llm_h, llm_w)).reshape(-1)
        w_index = np.broadcast_to(np.arange(llm_w, dtype=np.int64)[None, None, :], (llm_t, llm_h, llm_w)).reshape(-1)
        segments.append(np.stack([t_index, h_index, w_index]) + next_pos)
        next_pos += max(llm_t, llm_h, llm_w)
        st = end
    if st < len(token_list):
        text_len = len(token_list) - st
        segments.append(np.broadcast_to(np.arange(text_len, dtype=np.int64), (3, text_len)) + next_pos)
    if not segments:
        return None
    positions = np.concatenate(segments, axis=1)
    new_tensor = getattr(grid, "new_tensor", None)
    return new_tensor(positions) if callable(new_tensor) else positions


def _array_to_numpy(value: Any):
    import numpy as np

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    return np.asarray(value)


def _find_image_span(tokens: list[int], image_token_id: int, start: int, count: int) -> int:
    if count <= 0:
        raise ValueError("image token count must be positive")
    for idx in range(start, len(tokens) - count + 1):
        if all(token == image_token_id for token in tokens[idx : idx + count]):
            return idx
    raise ValueError("expanded image token span does not match image_grid_thw")
