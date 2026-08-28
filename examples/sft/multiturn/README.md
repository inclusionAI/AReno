# Multi-turn Chat SFT Example

This example shows the multi-turn SFT dataset-loader contract. The loader
accepts rows with a `messages` field (OpenAI/HF chat format) or a
`conversations` field (ShareGPT format) and normalizes them to the SFT
trainer's `messages` schema.

## Sample data

Create a JSONL file with multi-turn conversations:

```jsonl
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "And 3+3?"}, {"role": "assistant", "content": "6"}]}
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

## Train on all assistant turns (default)

```bash
areno train \
  --algo sft \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /path/to/multiturn.jsonl \
  --dataset-loader-fn examples/sft/multiturn/dataset_loader.py \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 2 \
  --mini-bs 1
```

## Train only on the final assistant turn

Use `--sft-assistant-turns last` to train only on the last assistant response
in each conversation. Earlier assistant turns are treated as context. This is
useful for focused evaluation of end-to-end multi-turn behavior.

```bash
areno train \
  --algo sft \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /path/to/multiturn.jsonl \
  --dataset-loader-fn examples/sft/multiturn/dataset_loader.py \
  --sft-assistant-turns last \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 2 \
  --mini-bs 1
```

## Observable output

- The training config summary printed at startup shows the resolved
  `sft_assistant_turns` value under the **Rollout** section.
- The dashboard run-config JSON includes `sft_assistant_turns` in its settings.
- Training logs show `sft_dataset_filter` counts for rows skipped due to
  empty messages, missing assistant turns, or budget limits.

## Mask semantics

- `user`, `system`, and `tool` tokens are always excluded from the loss.
- With `--sft-assistant-turns all` (default): every `assistant` turn is
  trainable.
- With `--sft-assistant-turns last`: only the final `assistant` turn is
  trainable; earlier `assistant` turns are context.
- EOS is appended after the last message and is trainable in both modes.
