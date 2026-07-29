FAQ
===

Can I run AReno training on CPU?
   No. CPU-only machines can build docs, run packaging checks, and run
   lightweight CPU tests, but AReno training and serving require CUDA hardware.

Is FlashAttention required?
   It is optional unless the selected attention backend requires it. Use
   ``--attn-backend native`` when debugging unsupported FlashAttention setups.

Should examples be copied from Cookbook or Reference?
   Start from Cookbook for runnable recipes. Use Reference when you already
   know which command, SDK type, or API contract you need.

Where should I start after installation?
   Run :doc:`/getting-started/quickstart`, then choose the RLVR or agentic
   rollout path that matches your task.

Why is my dataset tokenization cache not being used?
   The cache (``--dataset-cache-path``) is content-addressed, so a
   ``stage=dataset_cache_miss`` instead of ``_hit`` means one fingerprint input
   changed: the dataset content/schema, tokenizer asset files, chat template
   (including ``--disable-thinking``), or ``--max-prompt-tokens``. Run
   ``areno dataset-cache inspect --cache-path <dir>`` to list artifacts and
   their fingerprints. A ``stage=dataset_cache_rejected`` line means an
   artifact failed validation (corrupt file or version/fingerprint mismatch) and
   was recomputed; ``stage=dataset_cache_skip`` means a record was not
   JSON-serializable so the run fell back to re-tokenizing without caching. See
   :doc:`/cli/dataset_cache` for limitations.
