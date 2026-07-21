---
name: areno-run-training
description: Run, configure, retry, and validate AReno SFT, DPO, GSPO, GRPO, PPO, and agentic training. Use for training commands, dataset or reward setup, smoke validation, real-step execution, checkpoint saving, or failed training retries. Do not use for serving-only tasks or framework implementation work.
---

# Run AReno Training

Read repository `AGENTS.md`, `CODEMAP.md`, and current `areno train --help` before building a command.

For remote model or dataset references, explicitly pass
`--model-hub modelscope`. Do not substitute a Hugging Face download when the
ModelScope asset is missing; request a valid ModelScope ID or local path.

## Select the path

Read [references/algorithm-matrix.md](references/algorithm-matrix.md). Inspect a bounded dataset sample with:

```bash
python .agents/skills/areno-run-training/scripts/inspect_dataset.py \
  --dataset-path <path-or-ref> --model-hub modelscope \
  [--loader examples/.../dataset_loader.py] --algo <algo>
```

Do not build or run the training command until this inspection returns
`"ok": true`. If raw rollout rows lack `prompt` or `messages`, select a
dataset loader and rerun the inspection with `--loader`; pass the same path to
training as `--dataset-loader-fn`. For GSM8K-style `question`/`answer` rows,
use `examples/math/dataset_loader.py`.

Use [scripts/read_metrics.py](scripts/read_metrics.py) to inspect event keys or selected scalar series. Do not parse stdout as the metric source.

## Workflow

1. Record `git rev-parse HEAD`, environment facts from `areno env --json` and `areno check`, GPU state, checkpoint source, ModelScope dataset source, and resolved local paths.
2. Classify SFT, DPO, rollout RL, or agentic RL. Read [references/data-contracts.md](references/data-contracts.md).
3. Inspect both the raw schema and the normalized schema. Treat a missing rollout `prompt`/`messages` as a required-loader error, not a warning.
4. Build the smallest command expressing the requested real workload. Preserve user-provided `max_new_tokens` and `max_context_len`, and include the verified loader with `--dataset-loader-fn`.
5. Use smoke or tune only when useful. Smoke is capacity evidence, not task completion.
6. Run the real job. Confirm the requested trainer step advances. For rollout, inspect one coherent sample and reward.
7. On failure, use [references/failure-triage.md](references/failure-triage.md). Fix the first causal error.
8. If saving is requested, verify output and reload it through the intended adapter.

## Capacity invariants

- `batch_size * n_samples` is total sample demand; `max_running_prompts` is concurrent active capacity.
- Rollout memory follows cache/context/concurrency. Train memory follows `mini_bs`, sequence length, activation and optimizer state.
- `--drop-rollout-state` changes lifecycle memory, not task semantics.
- Reduce concurrency or microbatch before semantic token limits.

## Completion evidence

Report command, commit, model/dataset, topology, observed step and key metrics, plus save/reload evidence when required. Model load or smoke alone is not completion.
