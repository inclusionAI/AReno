:orphan:

Dataset tokenization cache
===========================

AReno Issue #206 adds an opt-in on-disk cache of the prompt samples that
``Trainer.load_prompt_batches`` produces after tokenization. With the cache
enabled, the first epoch tokenizes the dataset as usual; every later epoch — and
any later run whose dataset, tokenizer, chat template, and preprocessing options
are unchanged — replays the cached samples without re-tokenizing.

The cache is **off by default**. Leaving ``--dataset-cache-path`` unset keeps the
historical behavior (every epoch re-tokenizes), so existing runs are unchanged.

Enabling the cache during training
----------------------------------

``areno train`` exposes two options:

``--dataset-cache-path TEXT``
   Directory that holds the cache artifacts. Unset (the default) disables
   caching entirely; setting it opts in. The directory is created on demand.

``--dataset-cache-mode [auto|refresh|readonly]``
   Controls how the cache is used:

   * ``auto`` (default) — read on a hit; tokenize and persist on a miss.
   * ``refresh`` — ignore any existing artifact, re-tokenize, and overwrite.
     Use this to regenerate the cache after a code or data fix.
   * ``readonly`` — read on a hit; tokenize in memory on a miss but never
     persist. Use this for a read-only filesystem or a shared immutable cache.

The cache key fingerprints the dataset content, the tokenizer asset files, the
active chat template (including the ``--disable-thinking`` switch), and the
preprocessing options (``--max-prompt-tokens`` plus the prompt/solutions keys).
Changing any of those inputs resolves to a new key, so a stale or incompatible
entry is never reused.

Cache inputs are validated **before** the model and workers are initialized: an
unwritable path or an invalid mode fails fast with a
``stage=dataset_cache_config`` error rather than mid-training.

Observable output
-----------------

Each epoch Arreno logs a single ``stage=dataset_cache_hit`` or
``stage=dataset_cache_miss`` line, for example::

   ... stage=dataset_cache_hit key=a582e4e30a9b items=250 size_bytes=40211 skipped_long=3 load_time_s=0.012 tokenization_time_s=0.000 mode=auto

These events are emitted at the ``INFO`` level through the ``areno`` logger,
which writes to **stderr** (not stdout) -- so they will not appear in stdout
captures. Filter them with ``areno train ... 2>&1 | grep stage=dataset_cache_``
(or read ``stderr`` directly when driving ``areno`` as a subprocess). Adjust the
level with the ``ARENO_LOG_LEVEL`` environment variable (default ``INFO``).
The first run with a given fingerprint always logs ``miss`` and writes the
artifact; a subsequent run with the same fingerprint logs ``hit``.

When metrics recording is enabled, the same fields are written to the run's
dashboard-state JSON under ``stage=dataset_cache_load`` with the structured
fields ``hit``, ``items``, ``size_bytes``, ``skipped_long``, ``load_time_s``,
``tokenization_time_s``, ``mode``, and ``fingerprint_hash``.

Inspecting and removing the cache
---------------------------------

``areno dataset-cache`` reports cache artifacts and removes them explicitly.
Its ``--cache-path`` defaults to the ``ARENO_DATASET_CACHE_DIR`` environment
variable.

.. code-block:: bash

   # Human-readable summary (fingerprint prefix, row count, size, validity)
   areno dataset-cache inspect --cache-path /tmp/areno-cache

   # Machine-readable JSON report
   areno dataset-cache inspect --cache-path /tmp/areno-cache --json

   # Remove one artifact by fingerprint hash
   areno dataset-cache clean --cache-path /tmp/areno-cache --fingerprint a582e4e30a9b

   # Remove every artifact under the cache directory
   areno dataset-cache clean --cache-path /tmp/areno-cache --all --json

``inspect --json`` returns ``{"cache_path", "count", "entries": [...]}`` where
each entry has ``fingerprint_hash``, ``count``, ``skipped_long``, ``size_bytes``,
``mtime``, and ``valid``. ``clean --json`` returns
``{"cache_path", "removed", "bytes_freed"}``.

Limitations
-----------

* The cache covers the rollout tokenization path
  (``Trainer.load_prompt_batches``; SFT, DPO, GSPO, GRPO, PPO, and agentic
  rollouts that route through it). SFT and DPO offline tokenization are not
  cached yet and will reuse this mechanism in a follow-on change.
* Cached records must be JSON-serializable. Datasets that carry non-serializable
  fields (for example multimodal rows with binary payloads) gracefully fall back
  to re-tokenizing every epoch and write no partial artifact; a
  ``stage=dataset_cache_skip reason=non_serializable_record`` line is logged.
* On a cache hit the per-batch ``scanned``/``skipped_long`` counters are
  best-effort (the over-long rows were already filtered). The cumulative
  ``rollout/total_skipped_long`` metric is preserved exactly.
* The cache materializes one epoch's samples into a single JSON artifact, so
  enabling it raises peak memory modestly for very large datasets.
* Multiple training processes sharing one cache directory never observe a
  partial write (artifacts are written atomically via a temp file + rename), but
  may each tokenize once before converging on the same artifact.

Minimal reproducible example
----------------------------

No network or sandbox is needed. Build a tiny local dataset, run two epochs with
the cache enabled, and observe the second epoch skip tokenization:

.. code-block:: bash

   mkdir -p /tmp/areno-cache-demo
   printf '%s\n' '{"prompt": "What is 2+2?", "solutions": ["4"]}' \
                 '{"prompt": "What is 3+3?", "solutions": ["6"]}' > /tmp/areno-cache-demo/data.jsonl

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path /tmp/areno-cache-demo/data.jsonl \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo --tp-size 1 --world-size 1 --batch-size 1 --epochs 2 \
     --dataset-cache-path /tmp/areno-cache-demo/cache

The first epoch logs ``stage=dataset_cache_miss`` and writes the artifact; the
second logs ``stage=dataset_cache_hit`` with no tokenization time. Inspect it
afterward:

.. code-block:: bash

   areno dataset-cache inspect --cache-path /tmp/areno-cache-demo/cache --json

The same flow is packaged as a runnable script that generates the dataset,
captures the events from **stderr** (not stdout), and exposes the heavy knobs
(``ARENO_CKPT``, ``ARENO_WORLD_SIZE``, ...) as environment variables::

   python examples/cache/cache_demo.py                       # single-GPU smoke
   ARENO_WORLD_SIZE=2 python examples/cache/cache_demo.py    # reproduce the dp=2 case

See ``examples/cache/README.md`` for details.
