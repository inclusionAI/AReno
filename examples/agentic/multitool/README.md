# Multi-Tool Agentic RL Example

This example trains a policy on a multi-tool task-calling benchmark. Each task
requires two or more tool calls executed in the correct order. All tools are
side-effect-free and operate on in-memory data — no network or database is
needed.

## Tools

| Tool | Description |
|------|-------------|
| `lookup_contact` | Look up a contact by partial name |
| `read_note` | Read a note by key |
| `calculate` | Evaluate a safe arithmetic expression |
| `unit_convert` | Convert between length/weight units |
| `lookup_parcel` | Look up parcel tracking info |
| `search_notes` | Search notes by keyword (returns matching keys + snippets) |
| `list_contacts_by_city` | List all contacts in a given city |

## Task Types

- **contact-meeting**: Find Alice's phone, then check the meeting note
- **budget-shipping**: Read the budget note, then read the shipping note
- **parcel-city**: Look up parcel P002, then find a contact in the same city
- **calc-shipping**: Calculate `3 * 15`, then read the shipping note
- **convert-parcel**: Convert 100 cm to m, then look up parcel P003
- **search-meeting-contact** (3 steps): Search notes for 'meeting', read the meeting note, then list contacts in Shanghai
- **parcel-calc-note** (3 steps): Look up parcel P002, calculate `7 - 6`, then read the shipping note
- **convert-search-contact-parcel** (4 steps): Convert 1000 mm to m, search notes for 'shipping', list contacts in Shanghai, then look up parcel P001

## Reward Dimensions

The reward function scores each trajectory across four dimensions:

| Dimension | What it checks |
|-----------|---------------|
| `tool_selection` | All required tools appear in the trajectory |
| `arguments` | Tool call arguments match expected values |
| `order` | Tools appear in the correct relative order |
| `final_answer` | The last tool call produces the expected result |

Failure classes are tracked separately: `tool_selection`, `arguments`, `order`,
`final_answer`.

## Generate Tasks

```bash
python examples/agentic/multitool/dataset_generator.py \
  --output /tmp/areno-multitool.jsonl \
  --count 2048 \
  --seed 2026
```

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-multitool.jsonl \
  --dataset-loader-fn examples/agentic/multitool/dataset_loader.py \
  --reward-fn-path examples/agentic/multitool/reward.py \
  --agent-fn examples/agentic/multitool/run_agent.py \
  --algo gspo \
  --batch-size 8 \
  --n-samples 4 \
  --max-new-tokens 128
```

## Observable Output

- **Logs**: The agent logs task count and max running prompts at start.
- **Metrics**: Reward is emitted per sample; per-dimension breakdown is
  available in `score_task` return value.
- **Artifacts**: Trajectories and tool results are captured in the standard
  AReno rollout artifacts.

## Limitations

- All data is in-memory and deterministic — no external state.
- The calculator supports `+`, `-`, `*`, `/`, and parentheses only.
- Unit conversion supports length (`m`, `cm`, `mm`, `km`) and weight (`g`, `kg`,
  `mg`).