---
name: areno-profile-performance
description: Measure and diagnose AReno rollout, prefill, decode, training, checkpoint, role-switch, communication, or Python scheduling performance. Use when throughput or step time is slow and evidence from metrics, py-spy, or Nsight is required. Do not optimize before correctness is established.
---

# Profile AReno Performance

Profile a bounded, representative train or serve workload. Collect low-overhead
time, GPU, process, and TensorBoard evidence first; use sampling or tracing only
after those signals identify the subsystem to inspect.

```bash
python .agents/skills/areno-profile-performance/scripts/monitor_gpu.py \
  --pid <areno-pid> --duration 60 --output /tmp/areno-gpu.jsonl
python .agents/skills/areno-profile-performance/scripts/monitor_process.py \
  --pid <areno-pid> --duration 60 --output /tmp/areno-process.jsonl
python .agents/skills/areno-profile-performance/scripts/summarize_monitor.py \
  /tmp/areno-gpu.jsonl
python .agents/skills/areno-profile-performance/scripts/summarize_monitor.py \
  /tmp/areno-process.jsonl
python .agents/skills/areno-profile-performance/scripts/summarize_events.py \
  <metrics-dir> --list
python .agents/skills/areno-profile-performance/scripts/summarize_events.py \
  <metrics-dir> --pattern 'time/*' --pattern '*throughput*' --drop-first 1
python .agents/skills/areno-profile-performance/scripts/probe_openai_latency.py \
  --base-url http://127.0.0.1:8000 --model <model> --requests 16 --concurrency 4
python .agents/skills/areno-profile-performance/scripts/build_nsys_command.py \
  --output /tmp/areno-profile -- <bounded-command> [args...]
```

## Workflow

1. Record workload, commit, model, topology, token lengths, concurrency, GPU, and dependency versions.
2. Find the parent AReno PID and monitor its process tree. For train jobs, attach during at least two post-warmup steps. For serve jobs, use a fixed request set and concurrency.
3. Capture per-GPU utilization, memory, power, and target-process memory. Memory capacity, compute utilization, and throughput are separate signals.
4. For train, list TensorBoard scalar names before selecting series. Summarize stage time, tokens/throughput, communication, optimizer, loss, and memory metrics that actually exist; do not assume fixed tags.
5. For serve, measure TTFT and total request latency with streaming enabled while the GPU and process monitors run. Record prompt/output lengths and active concurrency.
6. Exclude initialization, checkpoint load, compilation, and CUDA graph capture from steady-state conclusions, but report them separately when startup is the problem.
7. Use `py-spy record -p <pid> -o /tmp/areno.svg --duration 30` for Python scheduling, data processing, serialization, blocking I/O, or compilation orchestration.
8. Use a bounded Nsight Systems capture for GPU compute, communication, synchronization, allocator activity, or launch gaps. Read [references/profile-policy.md](references/profile-policy.md).
9. Compare baseline and candidate with identical workloads and multiple steady-state observations, then re-run correctness validation after optimization.

Report raw workload metadata, selected window, GPU peak/average memory and utilization, process CPU/RSS, stage breakdown or TTFT/latency, bottleneck evidence, profiler overhead, and before/after metrics. A single warmup step is not a benchmark.
