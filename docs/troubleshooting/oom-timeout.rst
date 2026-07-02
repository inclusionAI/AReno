:orphan:

Training OOM and timeout
========================

Out-of-memory and timeout failures usually come from rollout volume, sequence
length, tensor parallelism, model size, or slow external agent work.

First reductions:

* Lower ``--batch-size``.
* Lower rollout or sequence length settings.
* Use a smaller checkpoint for the first reproduction.
* Reduce agent concurrency for tool or environment tasks.
* Confirm no unrelated GPU process is consuming memory.

For agentic tasks, distinguish model time from environment time. Tool calls,
tests, sleeps, browser work, and sandbox actions can dominate rollout wall
time before the model is called again.

See :doc:`/cli/observability` for timing and metric interpretation.
