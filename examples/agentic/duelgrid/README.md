# DuelGrid Agentic Example

DuelGrid is a small turn-based grid tactics example for agentic RLVR. The user
controls `U`; the LLM controls `A`. Each LLM step receives a text map, state
values, and a strict list of legal JSON actions.

## Training effect

Before GSPO/RLVR post-training, Gemma-E2B-it performs poorly in DuelGrid: it
often oscillates between nearby tiles instead of building a plan. After
training, the agent becomes much more purposeful. It actively collects health
and energy pickups, chases the user, attacks when it has position, and avoids
trap tiles while spending its turn energy.

The reward curve shows the same transition quantitatively: reward improves
quickly early in training and then stabilizes after the policy has learned the
game loop.

| Train before | Reward | Train after |
| --- | --- | --- |
| <img src="images/train_before.gif" alt="Gemma-E2B-it before DuelGrid training, repeatedly moving without progress" width="260"> | <img src="images/train_reward.jpg" alt="DuelGrid training reward curve" width="260"> | <img src="images/train_after.gif" alt="Gemma-E2B-it after DuelGrid training, pursuing the user and using pickups" width="260"> |

Generate prompt states:

```bash
python examples/agentic/duelgrid/dataset_generator.py --count 256 --output /tmp/duelgrid_states.jsonl
```

Run the browser UI:

```bash
python examples/agentic/duelgrid/web_ui.py
```

Then open `http://127.0.0.1:8765`. The browser UI uses the same rule engine as
training, but presents the map, status bars, legal actions, keyboard controls,
and short instructions in a richer layout.

Run the browser UI with an OpenAI-compatible LLM controlling `A`:

```bash
python examples/agentic/duelgrid/web_ui.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model policy
```

Add `--debug-llm` to show the raw OpenAI-compatible chat completion in the
browser event panel after each LLM move.

During browser play, `w/a/s/d` and arrow keys move, `f` attacks, `r` ranged
attacks, `e` picks up, `h` shields, and `x` waits. The agent moves when your
energy is spent.

The action space is intentionally structured:

```json
{"actions":[{"action":"MOVE","direction":"RIGHT"},{"action":"ATTACK","direction":"UP"}]}
{"actions":[{"action":"RANGED_ATTACK","direction":"LEFT"}]}
{"actions":[{"action":"PICKUP"},{"action":"SHIELD"}]}
{"actions":[{"action":"WAIT"}]}
```

For tool-call mode, the function name is always `choose_action`; `MOVE`,
`ATTACK`, and `RANGED_ATTACK` are action arguments, not tool names.
Energy costs are `MOVE=1`, `ATTACK=1`, `RANGED_ATTACK=2`, `SHIELD=1`,
`PICKUP=0`, and `WAIT=0`; energy refreshes to that player's current max energy
at the end of the turn. Picking up `E` increases max energy for future turns and
refills to the new max.

Rewards are computed by the rule engine, so the same generated states can be
used for supervised warmup, rollout collection, or GSPO/RLVR training.
