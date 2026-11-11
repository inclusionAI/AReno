"""Tokenizer loading helpers compatible with HuggingFace AutoTokenizer.

Newer HF tokenizers may pass `extra_special_tokens` as a list, which trips a
known attribute error inside `AutoTokenizer.from_pretrained` for some
checkpoints. We fall back to an explicit empty mapping so loading is robust to
this mismatch. Generic serialized fast tokenizers load without resolving a
model config, which also supports native architectures unknown to Transformers.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_tokenizer(model_path: str | Path):
    """Load a HF tokenizer, retrying once with safe special-token defaults."""

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    path = Path(model_path)
    tokenizer_config_path = path / "tokenizer_config.json"
    if tokenizer_config_path.is_file() and (path / "tokenizer.json").is_file():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        auto_map = tokenizer_config.get("auto_map")
        tokenizer_auto_map = auto_map.get("AutoTokenizer") if isinstance(auto_map, dict) else auto_map
        if (
            tokenizer_config.get("tokenizer_class") in {"PreTrainedTokenizerFast", "TokenizersBackend"}
            and not tokenizer_auto_map
        ):
            # A serialized generic fast tokenizer needs no model config.
            # Transformers 5 AutoTokenizer resolves the model first; its
            # generic-config fallback cannot initialize Phi's LongRoPE.
            return PreTrainedTokenizerFast.from_pretrained(model_path)

    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except AttributeError as exc:
        # Retry only for the known "list has no attribute 'keys'" path inside
        # transformers; re-raise anything else unchanged.
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            extra_special_tokens={},
        )


def load_processor(model_path: str | Path):
    """Load a HF processor when the checkpoint provides one, otherwise return None."""

    try:
        from transformers import AutoProcessor
    except ImportError:
        return None
    try:
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return None
