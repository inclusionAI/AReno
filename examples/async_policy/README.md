# Experimental async policy training smoke recipe

Minimal single-node, multi-GPU recipe for the experimental asynchronous
rollout/train pipeline (`async-grpo` / `async-gspo`):

- a background rollout producer generates `n_samples` completions per prompt
  on the **rollout engine**, scores rewards, and pushes prompt groups into a
  bounded queue;
- the train loop pops groups from the queue and runs optimizer steps on the
  **train engine**, so generation and training overlap on disjoint GPUs;
- after each optimizer step the new policy is streamed to the rollout engine
  with the NCCL policy-sync path, and every queued group is stamped with the
  policy version that produced it.

## Prerequisites

- A Linux node with several NVIDIA GPUs (suggested layout: 2 rollout + 2 train).
- AReno installed with its CUDA extension (`pip install -e . --no-build-isolation`).
- The model is resolved through ModelScope when a repository id is passed.
- The recipe runs on AReno's `native` attention backend, so the optional
  `flash-attn` package is not required.

## Run

```bash
python run_smoke.py --model Qwen/Qwen3-0.6B \
    --rollout-devices 0,1 --train-devices 3,4 \
    --n-samples 4 --max-steps 10
```

Both engine groups use TP 2 by default (`--tp-size`, `--rollout-tp-size`).
Device indices are logical positions inside the parent process'
`CUDA_VISIBLE_DEVICES`; the two groups must not overlap unless the GPUs can
hold both worker sets.

`--dataset-path` accepts the CLI-style reference `repo/name:config[:split]`
(default `gsm8k:main`) or a local path; remote references resolve through
ModelScope, or through Hugging Face with `--model-hub hf`.

## Async knobs

| Flag | Meaning | Default |
| --- | --- | --- |
| `--queue-max-groups` | Bounded queue capacity in prompt groups. When full, the rollout producer blocks (backpressure) instead of running ahead of training. | 8 |
| `--max-staleness` | Maximum policy versions a queued group may lag the current train policy. | 2 |
| `--stale-policy` | `drop` discards over-stale groups; `error` aborts the run instead. | drop |
| `--policy-sync-bucket-mb` | Bucket size for the GPU-to-GPU weight transfer. | 64 |

## What to look for

The pipeline is overlapping when:

1. Per-step logs interleave `stage=rollout_end` (producer thread) with
   `stage=train_start` / `stage=train_end` (consumer thread) instead of
   alternating strictly.
2. `train_stats` include `policy_sync_time_s` / `policy_sync_throughput_gbps`
   next to `step_train_time_s`, and the rollout engine is generating while the
   train engine steps (visible in `nvidia-smi` utilization on both groups).
3. `stale_dropped=N` in the train logs stays small; a large N means training
   outruns rollout — lower `--queue-max-groups` or raise `--max-staleness`.
4. `ratio_mean` / `logp_abs_diff_mean` from the async losses stay near 1/0;
   large drift shows staleness is high and the clipped importance ratio is
   doing real work.

Step wall-time metrics are approximate under overlap (the producer and
consumer share the Trainer step accounting); gradient and weight semantics
are unaffected.

## Related code

- Trainer: `areno/experimental/async_policy/trainer.py`
- Bounded queue: `areno/experimental/async_policy/queueing.py`
- Off-policy-correct losses: `areno/experimental/async_policy/losses.py`
- Weight sync: `areno/engine/policy_sync.py`
