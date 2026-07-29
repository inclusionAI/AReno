"""Clamp a checkpoint's serving context length before ``areno serve``.

Why this exists
---------------
``areno serve`` sizes its paged-KV allocation from ``max_position_embeddings``
in the checkpoint's ``config.json`` (``serve.py``: ``max_model_len =
engine.config.model.max_position_embeddings``). Qwen3.5 ships
``text_config.max_position_embeddings = 262144`` (256K native context), and the
checkpoint writer copies that verbatim into every ``step_*/config.json``.
Honouring 262144 pre-allocates a KV pool no single GPU can hold -> OOM at
startup, long before any 2048 prompt is served.

The 2048 demo only needs a tiny context (a 4x4 board prompt plus at most
``--default-max-tokens`` of generation), so clamping the checkpoint's
advertised context to a small serving budget removes the OOM without touching
the served model's weights or the areno source.

This is the demo-local equivalent of the manual ``config.json`` edit:
idempotent, only ever *lowers* the value, and safe to run before every serve.

Usage
-----
    python examples/agentic/2048/patch_serve_context.py \
        --ckpt ./2048/step_000100 --max-model-len 2048

Then launch serve as usual (no ``--max-model-len`` flag exists on the served
side; the patched ``config.json`` is what carries the budget).

    areno serve --model-path ./2048/step_000100 --tp-size 1 --world-size 1 \
        --port 8000 --max-running-prompts 1 --default-max-tokens 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The 2048 board prompt is short and ``--default-max-tokens`` is small, so 2048
# is a comfortable serving budget for the demo. Raise it here if you serve with
# a larger generation cap.
DEFAULT_MAX_MODEL_LEN = 2048


def patch_config(config: dict, max_model_len: int) -> list[str]:
    """Clamp every ``max_position_embeddings`` in ``config`` to ``max_model_len``.

    Only lowers; a value already at or below the target is left untouched so the
    script is idempotent and never widens a model's advertised context. Returns
    a list of human-readable change descriptions (empty when nothing changed).
    """

    changes: list[str] = []

    def clamp(holder: dict, key: str, *, where: str) -> None:
        if not isinstance(holder, dict):
            return
        current = holder.get(key)
        if not isinstance(current, int):
            return
        if current <= max_model_len:
            return
        holder[key] = max_model_len
        changes.append(f"{where}: {current} -> {max_model_len}")

    # Qwen3.5 nests the text config under "text_config"; some configs put the
    # field at the top level. Clamp both, plus any rope_scaling/rope_parameters
    # copy, so every path the loader reads agrees.
    text = config.get("text_config")
    if isinstance(text, dict):
        clamp(text, "max_position_embeddings", where="text_config")
        for rope_key in ("rope_scaling", "rope_parameters"):
            clamp(text.get(rope_key), "max_position_embeddings", where=f"text_config.{rope_key}")
    clamp(config, "max_position_embeddings", where="top-level")

    return changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clamp a checkpoint's config.json context length before `areno serve`."
    )
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory containing config.json.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"Serving context budget (prompt + generated tokens). Default {DEFAULT_MAX_MODEL_LEN}.",
    )
    args = parser.parse_args()

    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")

    config_path = Path(args.ckpt).expanduser() / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    changes = patch_config(config, args.max_model_len)
    if not changes:
        print(f"[patch_serve_context] {config_path} already <= {args.max_model_len}; no change.")
        return

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[patch_serve_context] patched {config_path}:")
    for line in changes:
        print(f"  {line}")
    print(
        "[patch_serve_context] ready to serve. KV budget is now bounded by this "
        "value; launch `areno serve` against this checkpoint."
    )


if __name__ == "__main__":
    main()