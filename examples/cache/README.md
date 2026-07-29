# Dataset tokenization cache demo (Issue #206)

A tiny, runnable example for the opt-in dataset tokenization cache exposed by
`areno train --dataset-cache-path`. The cache is off by default; setting the
path caches tokenized prompt samples keyed on dataset content, tokenizer, chat
template, and relevant options, so re-runs with the same fingerprint skip
re-tokenization. See `docs/cli/dataset_cache.rst` for the full contract.

## Run

```bash
python examples/cache/cache_demo.py                      # single-GPU smoke (world_size=1)
ARENO_WORLD_SIZE=2 python examples/cache/cache_demo.py   # reproduce the dp=2 case
ARENO_CKPT=Qwen/Qwen3-0.8B python examples/cache/cache_demo.py
```

The script generates a minimal math JSONL under a temp dir, launches
`areno train` against `examples/math/math_verify_reward.py`, and prints the
cache events.

## Where the cache events go

`stage=dataset_cache_hit` / `stage=dataset_cache_miss` (plus `rejected` / `skip`)
are emitted by the `areno` logger at the `INFO` level to **stderr**, not stdout.
So a subprocess that only captures `stdout` will see the config panel but no
cache events. The demo reads `result.stderr` for exactly this reason. From a
shell you can filter the same way:

```bash
areno train ... 2>&1 | grep stage=dataset_cache_
```

The first run with a given fingerprint always logs `miss` and writes the
artifact; a subsequent run with the same fingerprint logs `hit`. Verbosity is
controlled by the `ARENO_LOG_LEVEL` environment variable (default `INFO`).

## Notes

* The cache-load path runs only in the driver CLI process; spawned rank workers
  never tokenize prompts, so exactly one cache event is produced per epoch
  regardless of `world_size` / `dp_size`.
* The dataset is intentionally four rows -- enough to exercise the cache; the
  point is the event log, not the trained weights.