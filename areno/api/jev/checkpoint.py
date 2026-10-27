"""Decision metadata shared by training and serving backends."""

import json
from pathlib import Path


def save_head_config(path, *, base_model, max_seq_len):
    (Path(path) / "jev_config.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "head": "linear-gelu-linear",
                "pooling": "last_valid_token",
                "temperature": 1.0,
                "base_model": base_model,
                "max_seq_len": max_seq_len,
            },
            indent=2,
        )
        + "\n"
    )
