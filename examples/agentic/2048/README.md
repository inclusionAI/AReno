# Agentic 2048 Example

This example trains a policy to play full **2048** episodes. Each prompt is one
seeded starting board; the policy emits a bounded sequence of directions in a
single `choose_moves` tool call, and the engine replays the whole episode
deterministically at reward time. It includes both an OpenAI tool-call variant
and an XML no-tool variant. The environment is deterministic and self-contained.

The game, engine, dataset generator, and reward scoring run on pure Python, so
they are convenient to test on CPU; training and serving follow the same GPU
workflow as the other agentic examples.

## Files

- `game.py` — the 2048 engine: 4x4 board, four directions, seeded tile spawns,
  merge scoring, no-op detection, episode replay, the random-action baseline,
  and the shared move scoring used by both reward variants.
- `dataset_generator.py` — writes reproducible JSONL starting boards + a
  precomputed random baseline per board.
- `dataset_loader.py`, `run_agent.py`, and `reward.py` define the tool-call
  variant.
- `dataset_loader_no_tool.py`, `run_agent_no_tool.py`, and `reward_no_tool.py`
  define the XML no-tool variant.
- `baseline.py` — CPU-only random-baseline / trained-policy evaluation harness.
- `web_ui.py` — local browser demo (Human / Random / LLM modes), offline-capable.

## Generate Boards

```bash
python examples/agentic/2048/dataset_generator.py \
  --output /tmp/areno-2048-boards.jsonl \
  --count 2048 \
  --seed 2026
```

Each line is `{"id", "board", "seed", "random_baseline": {score, max_tile, invalid_rate, trials}}`.

## Run with Tool Calls

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-2048-boards.jsonl \
  --dataset-loader-fn examples/agentic/2048/dataset_loader.py \
  --reward-fn-path examples/agentic/2048/reward.py \
  --agent-fn examples/agentic/2048/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 256
```

## Play in the Web UI

Serve the trained checkpoint, then point the UI at its OpenAI-compatible
endpoint and switch to **LLM** mode:

```bash
areno serve --model-path /path/to/2048-policy --tp-size 1 --world-size 1 --port 8000
python examples/agentic/2048/web_ui.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key token --model policy --agent-mode llm
```

Open `http://127.0.0.1:8768`. Modes:

- **Human** — arrow keys or WASD; plays one move at a time. No server needed.
- **Random** — a uniform-random *direction* (all four, not legal-only) plays one
  move per *Agent Step*; no server needed.
- **LLM** — the policy plays a whole episode; the random baseline on the same
  board+seed and the trained-vs-baseline improvement are printed in the events
  panel. Requires `--base-url`.

## Evaluate against the Random Baseline

```bash
# CPU-only random baseline over the dataset
python examples/agentic/2048/baseline.py \
  --boards /tmp/areno-2048-boards.jsonl --cap 32 --json

# Optional: also evaluate the served trained policy and print the delta
python examples/agentic/2048/baseline.py \
  --boards /tmp/areno-2048-boards.jsonl --cap 32 \
  --base-url http://127.0.0.1:8000/v1 --api-key token --model policy
```

With `--json`, the structured output nests metrics under `random_baseline` (and
`trained_policy` / `improvement` when `--base-url` is given) and also mirrors
the headline means at top-level `summary` (`mean_score`, `mean_max_tile`,
`mean_invalid_rate`) for simple parsers. When `--cap`/`--trials` match the
values baked into the dataset, `random_baseline.baseline_source` is `stored`
(reusing the per-board baselines written by the generator); otherwise it is
`recomputed` (or `mixed`) and `recomputed` counts how many were rerun.

## Run without Tool Calls

The XML no-tool variant asks the model to answer with a moves tag such as
`<moves>up,left,down</moves>`.

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-2048-boards.jsonl \
  --dataset-loader-fn examples/agentic/2048/dataset_loader_no_tool.py \
  --reward-fn-path examples/agentic/2048/reward_no_tool.py \
  --agent-fn examples/agentic/2048/run_agent_no_tool.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 256
```

## Input & Output Contract

- **Board**: 4x4 list of ints, `0` == empty; tiles are powers of two.
- **Action space**: `up`, `down`, `left`, `right`.
- **Agent tool** (tool-call variant): `choose_moves` with `moves: string[]`
  (`enum: [up,down,left,right]`, `maxItems: 32`).
- **Agent output** (no-tool variant): one `<moves>up,left,down</moves>` tag.
- **Episode**: `game.play_episode(board, moves, seed=board_seed, cap=32)` replays
  the sequence under `random.Random(seed)`; identical `(board, moves, seed, cap)`
  reproduces the episode exactly (seeded replay). No-op moves are counted and
  penalized but do not advance the RNG.
- **Reward scalar**:
  `episode_score − random_baseline_score − INVALID_PENALTY * invalid_moves`.
- **Observable fields** (logged by `reward_fn` and printed by `baseline.py` /
  the web UI): episode score, max tile, invalid-move rate, trained-vs-baseline
  improvement, move count.

## Defaults & Limitations

- Default episode cap is 32 moves (`game.DEFAULT_EPISODE_CAP`); raise it for
  longer episodes.
- Tile spawns are seeded only — replay is deterministic, but live training
  spawns are stochastic per board seed.
- No external database, sandbox, or new runtime dependency; `openai`/`httpx` are
  imported lazily inside the agent, web UI, and baseline files only when an
  LLM/policy is used.
- `areno train` and `areno serve` need CUDA + a checkpoint; the engine,
  generator, loader, reward, baseline, and web UI run on CPU without a GPU.
- The random baseline (in `reward_fn`, `baseline.py`, and the web UI's Random
  mode) is a uniform-random direction over all four directions, not a
  legal-only policy — so it has a nonzero invalid-move rate, and any policy
  that picks legal directions should beat it on invalid-rate.