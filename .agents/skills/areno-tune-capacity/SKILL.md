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

## Capacity Recommendations

Generate conservative, balanced, and throughput-oriented override sets from
measured or estimated memory data **without starting a training run**:

```bash
# With measured profile data (from --smoke-infer or --tune-params)
python .agents/skills/areno-tune-capacity/scripts/recommend_capacity.py \
  --tp-size 4 --world-size 8 --batch-size 32 --n-samples 8 --mini-bs 16 \
  --peak-mem-frac 0.82 --json

# Without profile data (fallback estimation from GPU memory and model size)
python .agents/skills/areno-tune-capacity/scripts/recommend_capacity.py \
  --tp-size 4 --world-size 8 --batch-size 32 --n-samples 8 --mini-bs 16 \
  --gpu-memory-gb 80 --model-params-billions 7.0 \
  --output-dir /tmp/areno-overrides
```

Each recommendation is validated against AReno config constraints and exported
as an override file. The recommender never submits a training run. Read
[references/recommendation-strategy.md](references/recommendation-strategy.md)
for the full adjustment rules table and memory estimation formulas.
