# Calendar Scheduling Agentic RL Demo

A self-contained calendar scheduling agent for AReno. The agent must schedule
meetings across participants in different time zones by calling three tools:
**query_availability**, **propose_slot**, and **confirm_slot**.

No external calendars, databases, or network services are used. All
availability, time zone conversion, and conflict detection are computed
locally in `game.py`.

## Files

| File | Purpose |
|------|---------|
| `game.py` | Calendar scheduling engine: time zones, availability intersection, conflict detection, scoring |
| `dataset_generator.py` | Generate reproducible calendar scenarios as JSONL |
| `dataset_loader.py` | Load JSONL and convert to AReno prompt records |
| `reward.py` | Reward function: constraint satisfaction (0.7) + tool-call efficiency (0.3) |
| `run_agent.py` | Agent entrypoint with three tool definitions |

## Quick Start

### 1. Generate the dataset

```bash
python3 examples/agentic/calendar/dataset_generator.py \
  --output /tmp/calendar.jsonl --count 128 --seed 2026
```

This also creates `/tmp/calendar.held_out.jsonl` for held-out evaluation.

### 2. Train with AReno

The multi-turn agent flow produces longer trajectories than single-turn agents.
On GPUs with limited VRAM (e.g. Tesla T4 16 GB), use `--disable-thinking` to
suppress Qwen3 reasoning tokens, keep `--n-samples` small, and set
`--max-context-len` conservatively. See [Known Issues](#known-issues) below.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/calendar.jsonl \
  --dataset-loader-fn examples/agentic/calendar/dataset_loader.py \
  --reward-fn-path examples/agentic/calendar/reward.py \
  --agent-fn examples/agentic/calendar/run_agent.py \
  --algo gspo \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --n-samples 2 \
  --max-running-prompts 2 \
  --max-new-tokens 256 \
  --max-context-len 4096 \
  --mini-bs 1 \
  --max-steps 20 \
  --disable-thinking \
  --drop-rollout-state
```

## Scenario Structure

Each JSONL record contains:

```json
{
  "id": "meeting-0000",
  "participants": {
    "Alice": {"name": "Alice", "timezone": "UTC+8", "available_slots": [{"start_hour": 9, "end_hour": 17}]},
    "Bob": {"name": "Bob", "timezone": "UTC-5", "available_slots": [{"start_hour": 8, "end_hour": 14}]}
  },
  "meetings": [
    {"id": "meeting-0000", "duration_hours": 1, "required_participants": ["Alice", "Bob"]}
  ],
  "confirmed": {},
  "target_meeting_id": "meeting-0000"
}
```

## Reward Function

The reward combines two dimensions:

- **Constraint satisfaction (0.7 weight):** correct time zone conversion, all
  required participants available, no conflicts with confirmed meetings,
  correct meeting duration.
- **Tool-call efficiency (0.3 weight):** minimal redundant queries, single
  propose, single confirm.

Score range: -1.0 (invalid proposal) to 1.0 (perfect scheduling).

## Testing

```bash
python3 -m pytest tests/test_calendar_agent_cpu.py -v
```

## Limitations

- Fixed-offset time zones only (no DST).
- Hour-level granularity.
- No recurring meetings.
- No optional participants.

## Known Issues

### CUDA OOM on limited-VRAM GPUs (e.g. Tesla T4)

**Symptom:** Training crashes with `torch.OutOfMemoryError` during the logprobs
computation (`exp_logits = torch.exp(...)`) in the train step, even though
rollout succeeds.

**Root cause:** The multi-turn agent flow (query_availability per participant →
propose_slot → confirm_slot) produces trajectories with 3–5 model calls per
scenario. Each assistant response is appended to the conversation, so total
token counts per trajectory reach 3 000–7 000 tokens. Qwen3-0.6B has a ~152k
vocabulary; the logprobs layer allocates
`total_tokens × vocab_size × 4 bytes × 2` (logits + exp_logits), which for
8 000 tokens is already ~9.7 GB. Combined with model weights, gradients, and
Adam optimizer state (~9.6 GB), this exceeds T4's 15 GB VRAM.

Qwen3's **thinking mode** (enabled by default) exacerbates the problem: the
model generates 200–500 tokens of chain-of-thought reasoning before each tool
call, inflating trajectory length by 3–5×.

**Mitigations (in order of impact):**

1. **`--disable-thinking`** — suppresses Qwen3 thinking tokens, reducing each
   assistant response from ~500 to ~50 tokens. This is the single most
   effective flag for VRAM-constrained GPUs.
2. **Reduce `--n-samples`** — fewer samples per prompt means fewer trajectories
   in each train batch. `--n-samples 2` is the minimum for GSPO (group-relative
   advantage requires at least 2 samples to produce non-zero gradients).
3. **Lower `--max-context-len`** — filters out trajectories whose total token
   count exceeds the limit before they enter the train batch. Values of 3072–4096
   are practical on T4.
4. **Reduce `--max-new-tokens`** — caps the length of each model response during
   rollout, indirectly limiting trajectory length.
5. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — reduces VRAM
   fragmentation.

**Trade-off:** With `--disable-thinking`, the model loses chain-of-thought
reasoning ability and may produce lower-quality tool calls, leading to all
trajectories receiving the same reward (e.g. -1.0). When all samples in a group
share the same reward, GSPO computes zero advantage and zero gradient, so no
learning occurs. If this persists across many steps, consider:

- Increasing `--n-samples` (more diversity → higher chance of reward variance).
- Increasing `--lr` (e.g. `1e-5` instead of `1e-6`).
- Increasing `--max-steps` (more opportunities for reward variance to emerge).
- Using a larger GPU (A100 40 GB+ allows `--n-samples 4+` with thinking enabled).