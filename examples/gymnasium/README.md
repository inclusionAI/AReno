# Gymnasium Rollout Example

Drives any Gymnasium-style environment through the agentic rollout API with
`areno.integrations.gymnasium.make_gymnasium_rollout`. `run_agent` plays one
episode per prompt/sample item and returns trajectory turns; the paired
`reward_fn` scores each episode by its cumulative return.

Requires `gymnasium` (only the task files import it; `areno` itself stays
dependency-free):

```bash
pip install gymnasium
```

## Files

- `cartpole_task.py` defines `run_agent` and `reward_fn` for `CartPole-v1`.
- `crafter_task.py` drives Crafter's RGB observations through the same API
  (requires `crafter` and `Pillow`); it adapts Crafter's classic Gym API to
  the Gymnasium contract and renders each frame as a base64 PNG image part.
- `dataset_generator.py` writes reproducible episode records (JSONL).
- `dataset_loader.py` normalizes episode records for the training CLI.

## Generate Episodes

```bash
python examples/gymnasium/dataset_generator.py \
  --output /tmp/areno-gymnasium-episodes.jsonl \
  --count 2048 \
  --seed 2026
```

## Train

Both hooks point at the same task file:

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/areno-gymnasium-episodes.jsonl \
  --dataset-loader-fn examples/gymnasium/dataset_loader.py \
  --agent-fn examples/gymnasium/cartpole_task.py \
  --reward-fn-path examples/gymnasium/cartpole_task.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 8
```

## Crafter RGB

The same factory drives Crafter's RGB observations through a vision-language
model. `crafter_task.py` renders each frame as a base64 PNG image part and
adapts Crafter's classic Gym API to the Gymnasium contract.

```bash
pip install crafter Pillow

python examples/gymnasium/dataset_generator.py \
  --output /tmp/areno-gymnasium-episodes.jsonl \
  --count 64 \
  --seed 2026 \
  --prompt "Play Crafter and maximize survival rewards."

areno train \
  --ckpt Qwen/Qwen3.5-0.8B \
  --dataset-path /tmp/areno-gymnasium-episodes.jsonl \
  --dataset-loader-fn examples/gymnasium/dataset_loader.py \
  --agent-fn examples/gymnasium/crafter_task.py \
  --reward-fn-path examples/gymnasium/crafter_task.py \
  --algo gspo \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --n-samples 4 \
  --mini-bs 4 \
  --epochs 1 \
  --max-running-prompts 4 \
  --max-new-tokens 16 \
  --max-prompt-tokens 8192 \
  --attn-backend native \
  --drop-rollout-state
```

Note: the CLI preflight locates task hooks as top-level `def` nodes, so the
task files expose the factory pair through explicit `run_agent` / `reward_fn`
wrappers instead of plain assignment unpacking.

## Episode Semantics

- Each record's integer `seed` fixes the episode: the env is reset with
  `env.reset(seed=seed)`, so episodes reproduce exactly and samples stay
  isolated (every episode gets its own env instance).
- The episode ends on `terminated`, on `truncated`, on `max_episode_steps`
  (recorded as truncation), or on an invalid action.
- An invalid action — `decode_action` raises, or the env rejects the action —
  ends the episode immediately; the sample keeps the partial return accumulated
  so far and its stats record `invalid_action: true`.
- The score is the cumulative return including the final step's reward.
- `encode_observation` may return a multimodal content-part list (for example a
  text instruction plus a base64 image part) for RGB observations.

Episode statistics (`episode_return`, `length`, `seed`, `terminated`,
`truncated`, `invalid_action`) travel on the sample record under
`gymnasium_episode`, so `--agent-fn` and `--reward-fn-path` can load the same
task file independently. The `n_samples` items of one prompt share the trainer's
record dict, so each episode writes its stats onto a per-episode copy of the
record; samples stay isolated even inside one prompt group.
