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
