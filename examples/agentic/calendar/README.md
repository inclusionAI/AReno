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

```bash
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
  --n-samples 8 \
  --max-new-tokens 512 \
  --max-context-len 2048 \
  --mini-bs 1
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