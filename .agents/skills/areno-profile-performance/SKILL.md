---
name: areno-profile-performance
description: Measure and diagnose AReno rollout, prefill, decode, training, checkpoint, role-switch, communication, or Python scheduling performance. Use when throughput or step time is slow and evidence from metrics, py-spy, or Nsight is required. Do not optimize before correctness is established.
---

# Profile AReno Performance

Start with TensorBoard timing and throughput, then choose Python or GPU profiling based on evidence.

```bash
python .agents/skills/areno-profile-performance/scripts/summarize_time_metrics.py <metrics-dir>
python .agents/skills/areno-profile-performance/scripts/build_nsys_command.py \
  --output /tmp/areno-profile -- <bounded-command> [args...]
```

## Workflow

1. Record workload, commit, model, topology, token lengths, concurrency, GPU, and dependency versions.
2. Exclude first-step compilation from steady-state conclusions.
3. Use TensorBoard `time/*` and throughput metrics to identify the dominant stage.
4. Use `py-spy` when the process is spending wall time in Python, I/O, serialization, or compilation orchestration.
5. Use a bounded Nsight Systems capture for GPU compute, communication, synchronization, or launch gaps. Read [references/profile-policy.md](references/profile-policy.md).
6. Compare baseline and candidate with identical workloads and multiple steady-state observations.
7. Re-run correctness validation after optimization.

Report raw workload metadata, selected window, stage breakdown, bottleneck evidence, and before/after metrics. A single warmup step is not a benchmark.
