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
     - Dense text and Qwen3.5-VL image checkpoints.
   * - Qwen3.5-MoE
     - Qwen3.5 routed expert text and VL checkpoints, including
       ``qwen3_5_moe`` layouts, with Areno MoE kernels.
   * - Bailing MoE Linear v2
     - Local model adapter for Bailing MoE Linear v2 checkpoints.
   * - Gemma4
     - Gemma4 text and conditional-generation checkpoints. Native Gemma4 and
       Gemma4 Unified processors support image, audio, and video inputs for
       serving and training.
   * - MiniCPM-family adapters
     - MiniCPM-family text and vision adapters used by the local training
       stack.

.. important::

   Model support means the checkpoint can be loaded through an Areno model
   adapter. Some model families may support inference before every training or
   save path is fully optimized.

For the common media message format and runtime behavior, see
:doc:`../concepts/multimodal-inputs`.
