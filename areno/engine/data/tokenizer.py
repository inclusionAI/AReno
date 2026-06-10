"""Tokenizer loading helpers compatible with HuggingFace AutoTokenizer.

Newer HF tokenizers may pass `extra_special_tokens` as a list, which trips a
known attribute error inside `AutoTokenizer.from_pretrained` for some
checkpoints. We fall back to an explicit empty mapping so loading is robust to
this mismatch.
"""

from __future__ import annotations

from pathlib import Path


def load_tokenizer(model_path: str | Path):
    """Load a HF tokenizer, retrying once with safe special-token defaults."""

    from transformers import AutoTokenizer

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
