---
name: areno-tune-capacity
description: Fit an AReno training or rollout workload to available GPUs by tuning TP, mini-batch, batch, sample count, and rollout concurrency. Use for OOM prevention, memory headroom, smoke-infer, smoke-train, or tune-params requests. Do not change semantic token limits unless requested.
---

# Tune AReno Capacity

Inspect current train help and `areno/cli/auto_tune.py`. Validate relationships first:

```bash
python .agents/skills/areno-tune-capacity/scripts/check_capacity.py \
  --batch-size N --n-samples N --max-running-prompts N \
  --mini-bs N --world-size N --tp-size N
```

## Workflow

1. Record GPU count/memory, model config, dtype, TP constraints, optimizer, and semantic token lengths.
2. Measure rollout with `--smoke-infer` when useful; it must allocate cache and capture decode graphs.
3. Measure train with `--smoke-train`; it skips rollout/prefill and uses the candidate microbatch.
4. Use `--tune-params` when requested. Keep peak memory at or below the requested fraction, never above `0.9` when selecting a default safety target.
5. Avoid excessive probes. Probe to make a decision, then confirm the chosen setting.
6. Run a bounded real workload if the overall user goal is a working task.

Preserve `max_new_tokens` and `max_context_len`. Read [references/parameter-relations.md](references/parameter-relations.md) before adjusting multiple dimensions.
