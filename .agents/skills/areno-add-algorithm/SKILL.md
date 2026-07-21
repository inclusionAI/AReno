---
name: areno-add-algorithm
description: Add or modify an AReno algorithm, trainer, loss, advantage calculation, role model, or algorithm-specific configuration. Use for framework-level SFT, DPO, GSPO, GRPO, PPO, or new optimization method development. Do not use merely to run an existing algorithm.
---

# Add an AReno Algorithm

Start from `AlgorithmSpec` and registration in `areno/api/algorithms.py`; do not add factory branches.

Develop on a dedicated local branch. Commit changes locally, then update the
remote GPU checkout by fetching and pulling that branch; never patch source on
the remote host. Use ModelScope for any model or dataset references used by the
validation workload.

```bash
python .agents/skills/areno-add-algorithm/scripts/inspect_algorithms.py
```

## Workflow

1. Define whether the algorithm is offline, rollout policy-only, or multi-role. Read [references/ownership.md](references/ownership.md).
2. Specify input records, sequence construction, masks, role models, loss inputs, and metrics before coding.
3. Add the narrowest config type and preserve public defaults/compatibility.
4. Put batch/materialization logic in the trainer and tensor mathematics in `areno/api/loss_fns/` or advantage helpers.
5. Register one `AlgorithmSpec`; load experimental implementations through `areno/experimental/` when appropriate.
6. Add CPU tests for registration, config, masks, exact small-tensor math, and trainer dispatch.
7. Run the new algorithm end to end for at least two consecutive successful training steps using a real model and representative data. Verify finite losses, metrics, and gradients on both steps. For rollout algorithms, also verify bounded GPU rollout/train logprob consistency.

Completion requires registry discovery, deterministic mathematical tests, evidence from at least two successful end-to-end training steps, and role lifecycle checks where applicable. A one-step smoke train is useful for diagnosis but does not complete algorithm validation.
