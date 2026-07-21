# Profile Policy

- Keep traces bounded; capture a known steady-state interval rather than process lifetime.
- Include CUDA, NVTX, OS runtime, and process tree only when required for the question.
- Attribute gaps before naming them idle time: they may be synchronization, CPU launch delay, compilation, allocator work, or data preparation.
- Decode throughput depends on active sequence count and context distribution. Report both.
- Train comparisons match tokens, microbatching, optimizer, recompute, and parallel topology.
- Profile overhead is not production throughput; validate the change without the profiler.
