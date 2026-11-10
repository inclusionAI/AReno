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
task file independently.
