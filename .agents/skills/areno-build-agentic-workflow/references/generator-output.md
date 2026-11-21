# Agentic Project Generator Output

`scripts/generate_agentic_project.py` scaffolds a runnable, dependency-light
AReno agentic project. It uses only AReno's existing public contracts and
introduces no external database or mandatory sandbox.

## Command

```bash
python .agents/skills/areno-build-agentic-workflow/scripts/generate_agentic_project.py \
  --name my-gridworld \
  --out examples/agentic/my-gridworld \
  [--force] [--seed 2026]
```

## Input contract

| Option   | Required | Default                       | Description                                            |
| -------- | -------- | ----------------------------- | ------------------------------------------------------ |
| `--name` | yes      | —                             | Project name, lowercase hyphen-case (`^[a-z0-9]+(-[a-z0-9]+)*$`). |
| `--out`  | no       | `examples/agentic/<name>`     | Output directory.                                      |
| `--force`| no       | off                           | Overwrite a non-empty output directory.                |
| `--seed` | no       | `2026`                        | Deterministic seed for generated fixtures.             |

## Defaults and safety

- Default behavior is backward compatible: the generator only adds files; it
  never deletes or modifies unrelated project state.
- A non-empty `--out` is refused without `--force`, so existing user edits are
  never silently overwritten.
- Output is deterministic for a fixed `--name` and `--seed`.

## Output fields (structured summary)

The script prints human-readable progress lines followed by a single JSON line:

```json
{"ok": true, "name": "my-gridworld", "seed": 2026, "path": "...", "files": ["..."], "episode_cmd": "python .../run_episode.py"}
```

## Generated files

| File                  | Contract                                                        |
| --------------------- | --------------------------------------------------------------- |
| `game.py`             | `Env` with `reset` / `step` / `legal_actions` / `render`; clear error and reward. |
| `dataset_generator.py`| CLI `--output --count --seed`, reproducible JSONL.              |
| `dataset_loader.py`   | `load_training_dataset(dataset_path, *, default_loader=None, **_) -> list[dict]`. |
| `tool_defs.py`        | `tools() -> list[dict]`, `tool_choice() -> dict`.               |
| `run_agent.py`        | `async def run_agent(ctx, batch) -> AgentTrajectory`.           |
| `reward.py`           | `reward_fn(record) -> float`.                                   |
| `run_episode.py`      | No-model smoke; runs one fixed episode, prints `episode_total_reward=`. |
| `README.md`           | Minimal runnable example and observable output.                 |

## Observable output

`run_episode.py` prints reset/step/reward lines ending with
`episode_total_reward=<float>`, so the scaffold can be verified on CPU without
a model, network, or GPU.

## Limitations

- The generated environment is a placeholder gridwalk; replace `game.py` with
  the real task domain.
- `run_agent.py` requires `pip install openai` and a reachable
  OpenAI-compatible endpoint; `run_episode.py` does not.