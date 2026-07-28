# Validation Matrix

| Change | Required comparisons |
| --- | --- |
| Algorithm/loss | masks, rewards/advantages, token logprobs, loss, gradients |
| Model | reference logits, rollout text, rollout/train logprobs, gradients |
| Checkpoint | loaded key coverage, shapes, load-save-load logits |
| Scheduler/runtime | ordering, cancellation, cache isolation, batching |
| Kernel | forward, backward, dtype, layout and supported non-contiguous inputs |
| Performance | steady throughput, step time, memory; never correctness proof |
