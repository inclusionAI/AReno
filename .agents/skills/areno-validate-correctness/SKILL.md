---
name: areno-validate-correctness
description: Compare an AReno branch, model, checkpoint, algorithm, scheduler, or kernel against a baseline. Use for regression validation, logprob or metric comparison, checkpoint round trips, and correctness gates. Keep performance reporting separate.
---

# Validate AReno Correctness

Create an isolated baseline worktree. Keep checkpoint, input, seed, algorithm, topology, dtype, and bounded step count identical.

## Primitives

```bash
python .agents/skills/areno-validate-correctness/scripts/compare_metrics.py BASE_DIR CANDIDATE_DIR --metric train/loss
python .agents/skills/areno-validate-correctness/scripts/compare_arrays.py base.json candidate.json --atol 1e-5 --rtol 1e-4
python .agents/skills/areno-model-adaptation/scripts/inspect_checkpoint.py /path/to/checkpoint
```

Read [references/validation-matrix.md](references/validation-matrix.md). Select invariants based on ownership; final loss alone is insufficient for model, algorithm, scheduler, or kernel changes.

## Workflow

1. State baseline commit and candidate commit.
2. Capture environment and workload metadata.
3. Run baseline and candidate independently. Do not switch the user's dirty worktree.
4. Compare the earliest meaningful boundary, then downstream outputs.
5. Explain numerical scale before choosing tolerance. Never loosen it merely to pass.
6. Report correctness separately from throughput, step time, and memory.

Completion requires reproducible commands, artifacts, comparison output, and a nonzero exit on failed invariants.
