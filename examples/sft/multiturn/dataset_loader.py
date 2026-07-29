"""多轮对话数据集 + loader，用于测试 --split-conversations。

用法:
    areno train --algo sft --ckpt Qwen/Qwen3-0.6B \
        --dataset-path examples/sft/multiturn/conversations.jsonl \
        --dataset-loader-fn examples/sft/multiturn/dataset_loader.py \
        --max-prompt-tokens 256 \
        --split-conversations
"""

from __future__ import annotations

import json
from pathlib import Path


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """读取 JSONL 格式的多轮对话数据集。

    每行一个 JSON 对象，格式:
        {"messages": [{"role": "...", "content": "..."}, ...]}
    可选字段:
        "system": [{"role": "system", "content": "..."}]
    """

    path = Path(dataset_path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
    return records