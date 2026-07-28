# Reward Hooks

AReno reward functions score each rollout completion against its prompt. This
page describes the single-record contract, the optional batched contract, and
the CLI for inspecting both.

## Input contract

Every reward hook receives a `RewardRecord` with at least these fields:

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `str` | The prompt text fed to the rollout. |
| `completion` | `str` | The decoded model response. |
| `answer` | `Any \| None` | Gold answer from the dataset, if any. |
| `messages` | `list[dict]` | Full chat messages (agentic mode). |
| `tool_calls` | `list[dict]` | Tool calls made during rollout. |
| `tool_results` | `list[dict]` | Tool results returned to the model. |
| `tokens` / `logprobs` / `loss_mask` | `list[...]` | Per-token tensors for the full sequence. |
| `source_record` | `dict` | The original dataset row. |
| `metadata` | `dict` | Free-form metadata (e.g. prompt_index). |

## Single-record hook (default)

```python
def reward_fn(record: RewardRecord) -> float:
    return 1.0 if record.completion.strip() == str(record.answer) else 0.0
