# Data Contracts

- SFT normalizes supervised messages, prompt/response, instruction/output, question/answer, or text.
- DPO preserves one prompt and both chosen and rejected continuations or conversations.
- Rollout rows provide a prompt and source fields consumed by `reward_fn`.
- Agentic rows may contain OpenAI-style content and tool metadata. Assistant tool calls must precede matching tool results.
- Dataset loaders return records and must not own tokenizer or processor internals.
- Multimodal loaders return public image representations such as image URLs or data URLs; processor work remains in the runtime.

Inspect normalization in `areno/api/data_utils.py` and the concrete trainer before changing schemas.

When `--dataset-cache-path` is set, the rollout path caches tokenized prompt
samples keyed on dataset content, tokenizer assets, chat template, and
preprocessing options; later epochs log `stage=dataset_cache_hit` and skip
re-tokenization. It is off by default and only covers the rollout path; use
`areno dataset-cache inspect/clean` for size reporting and explicit removal.
