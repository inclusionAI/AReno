# Alpaca SFT Example

This example shows the required SFT dataset-loader contract using Alpaca-style
rows with `instruction`, optional `input`, and `output` fields.

Recommended public dataset:

```text
yahma/alpaca-cleaned
```

The loader normalizes raw rows to the SFT trainer schema:

- `prompt`: source text used as context
- `response`: supervised target suffix

Run SFT with the recommended Hugging Face dataset:

```bash
areno train \
  --algo sft \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path yahma/alpaca-cleaned \
  --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 2 \
  --mini-bs 1 \
  --max-prompt-tokens 128 \
  --max-new-tokens 64
```
