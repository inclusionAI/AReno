"""Prepare MLX-formatted training and validation datasets for QLoRA.

Reads `data/sft_payment_train.jsonl` (1,500 samples), converts into standard
multi-role `messages` format compatible with `mlx_lm.lora`, and partitions into:
- `data/mlx_sft/train.jsonl` (1,400 samples)
- `data/mlx_sft/valid.jsonl` (100 samples)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RECIPE_ROOT = Path(__file__).resolve().parent

from recipes.payment.payment_service.prompt_templates import (
    COMPACT_SYSTEM_PROMPT,
    format_compact_solution,
    format_compact_user_request,
)


import argparse
import statistics


def prepare_mlx_data(
    input_file: Path | str = RECIPE_ROOT / "data/sft_payment_train.jsonl",
    output_dir: Path | str = RECIPE_ROOT / "data/mlx_sft_compact",
    val_count: int = 100,
    model_path: Path | str = Path.home() / "models/Ling-3.0-tiny-4bit",
    max_seq_length: int | None = 1024,
) -> None:
    input_path = Path(input_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            raw_input = item.get("input", {})
            solution = item.get("solution", "")

            compact_user = format_compact_user_request(raw_input)
            compact_sol = format_compact_solution(solution)

            messages = [
                {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
                {"role": "user", "content": compact_user},
                {"role": "assistant", "content": compact_sol},
            ]
            rows.append({"messages": messages})

    # 若指定分词器与 max_seq_length，执行严格长度校验与过滤
    filtered_rows = rows
    if max_seq_length is not None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            valid_rows = []
            lengths = []
            dropped = 0
            for r in rows:
                text = tokenizer.apply_chat_template(r["messages"], tokenize=False)
                tok_len = len(tokenizer.encode(text))
                if tok_len <= max_seq_length:
                    valid_rows.append(r)
                    lengths.append(tok_len)
                else:
                    dropped += 1

            filtered_rows = valid_rows
            lengths.sort()
            p95 = lengths[int(len(lengths) * 0.95)]
            lte_900 = sum(1 for l in lengths if l <= 900)

            print(f"\n[DataPrep] 📊 序列过滤完成 (硬阈值: ≤ {max_seq_length} tokens):")
            print(f"  - 原始样本: {len(rows)} | 保留: {len(filtered_rows)} | 剔除超长: {dropped}")
            print(f"  - 保留集最小: {min(lengths)} | 中位数: {statistics.median(lengths)} | 均值: {statistics.mean(lengths):.1f}")
            print(f"  - P95: {p95} | 最大 Token: {max(lengths)}")
            print(f"  - ≤ 900 占比: {lte_900}/{len(lengths)} ({lte_900/len(lengths)*100:.1f}%)")
        except Exception as e:
            print(f"[DataPrep] ⚠️ 无法完成 Token 过滤: {e}")

    total = len(filtered_rows)
    if total <= val_count:
        raise ValueError(f"过滤后样本总数 ({total}) 小于等于验证集数量 ({val_count})")

    train_rows = filtered_rows[:-val_count]
    val_rows = filtered_rows[-val_count:]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "valid.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[DataPrep] ✔ 训练集导出: {train_path} ({len(train_rows)} 条)")
    print(f"[DataPrep] ✔ 验证集导出: {val_path} ({len(val_rows)} 条)")



def main() -> None:
    parser = argparse.ArgumentParser(description="切分 MLX SFT 紧凑数据集")
    parser.add_argument(
        "--input",
        type=str,
        default=str(RECIPE_ROOT / "data/sft_payment_train.jsonl"),
        help="原始 SFT 样本路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RECIPE_ROOT / "data/mlx_sft_compact"),
        help="切分产物目标目录",
    )
    parser.add_argument("--val-count", type=int, default=100, help="验证集样本数")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="序列长度硬阈值，超出将被剔除以保证显存安全",
    )
    args = parser.parse_args()

    prepare_mlx_data(
        input_file=args.input,
        output_dir=args.output_dir,
        val_count=args.val_count,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()

