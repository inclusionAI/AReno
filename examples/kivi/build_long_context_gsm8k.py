"""Build long-context rollout prompts: GSM8K test problems after WikiText-2 passages.

Each row holds one GSM8K test problem in the math example's prompt format,
preceded by a distinct ``--context``-token passage of the WikiText-2 text, so
every rollout attends over a long prefix while the reward stays a 0/1 math
check. Rows already use the trainer schema (``prompt``, ``solutions``) and
pass through ``examples/math/dataset_loader.py`` unchanged.

    python examples/kivi/build_long_context_gsm8k.py --tokenizer /path/to/Qwen3-0.6B \
        --gsm8k /path/to/gsm8k_main_test --text /path/to/wikitext2_test.txt --output /path/to/gsm8k_test512_ctx8k
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="Model directory whose tokenizer sizes the passages.")
    parser.add_argument("--gsm8k", required=True, help="GSM8K test split saved with `save_to_disk`.")
    parser.add_argument("--text", required=True, help="UTF-8 text file the passages are cut from.")
    parser.add_argument("--context", type=int, default=8000, help="Passage length in tokens.")
    parser.add_argument("--limit", type=int, default=512, help="Number of GSM8K problems.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import Dataset, load_from_disk
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.text, encoding="utf-8") as handle:
        ids = tokenizer(handle.read(), add_special_tokens=False)["input_ids"]
    span = len(ids) - args.context
    rows = []
    for index, record in enumerate(load_from_disk(args.gsm8k).select(range(args.limit))):
        # A prime stride spreads the passages over the whole text.
        start = (index * 7919) % span
        passage = tokenizer.decode(ids[start : start + args.context])
        final = str(record["answer"]).rsplit("####", 1)[-1].strip()
        prompt = (
            f"{passage}\n\n"
            "Solve the following grade-school math problem. Show your reasoning "
            "and put the final answer in \\boxed{}.\n\n"
            f"Problem: {record['question']}\nSolution:"
        )
        rows.append({"prompt": prompt, "solutions": [final], "solution": final, "passage_start": start})
    Dataset.from_list(rows).save_to_disk(args.output)
    lengths = [len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]) for row in rows]
    print(f"{len(rows)} rows; prompt tokens min {min(lengths)} max {max(lengths)}")


if __name__ == "__main__":
    main()
