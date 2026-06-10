Supported models
================

areno currently supports the following checkpoint families:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Family
     - Notes
   * - Llama-style dense decoder models
     - Dense causal decoder checkpoints with Llama-compatible layouts.
   * - Qwen3 dense
     - Qwen3 text checkpoints.
   * - Qwen3-MoE
     - Routed expert checkpoints with Areno MoE kernels.
   * - Qwen3.5
     - Dense Qwen3.5 text checkpoints.
   * - Qwen3.5-MoE
     - Qwen3.5 routed expert checkpoints, including ``qwen3_5_moe`` text/MoE layouts, with Areno MoE kernels.
   * - Bailing MoE Linear v2
     - Local model adapter for Bailing MoE Linear v2 checkpoints.
   * - Gemma4 text models
     - Gemma4 text-only checkpoints used by the local training stack.
   * - MiniCPM-family adapters
     - MiniCPM-family adapters used by the local training stack.

.. important::

   Model support means the checkpoint can be loaded through an Areno model
   adapter. Some model families may support inference before every training or
   save path is fully optimized.
