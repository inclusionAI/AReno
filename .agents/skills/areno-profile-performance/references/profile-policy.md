# Profile Policy

- Keep traces bounded; capture a known steady-state interval rather than process lifetime.
- Start with GPU/process sampling and TensorBoard events. Escalate to `py-spy`, PyTorch Profiler, or Nsight only when a lower-overhead signal justifies it.
- Include CUDA, NVTX, OS runtime, and process tree only when required for the question.
- Attribute gaps before naming them idle time: they may be synchronization, CPU launch delay, compilation, allocator work, or data preparation.
- Decode throughput depends on active sequence count and context distribution. Report both.
- Train comparisons match tokens, microbatching, optimizer, recompute, and parallel topology.
- GPU memory reports include peak and steady-state values per device. Distinguish allocated model/cache memory from transient peaks when evidence permits.
- TensorBoard event directories are job-specific. List available tags and select the target job's files; never combine unrelated runs into one conclusion.
- Serve comparisons report warmup policy, request concurrency, prompt/output lengths, TTFT, total latency, and throughput.
- Profile overhead is not production throughput; validate the change without the profiler.
