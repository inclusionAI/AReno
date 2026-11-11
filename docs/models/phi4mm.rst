Phi-4 Multimodal
================

AReno provides a native CUDA adapter for ``phi4mm`` checkpoints. It covers the
language decoder, SigLIP vision tower, Conformer audio encoder, modality
projectors, and the checkpoint's vision and speech LoRA branches. Text-only,
image-only, audio-only, and combined image/audio requests share the same model
adapter.

Checkpoint compatibility
------------------------

The adapter expects the released Phi-4 Multimodal configuration semantics,
including LongRoPE, the 448-pixel SigLIP vision path, and the 80-bin Cascades
audio processor with time reduction 8. Unsupported processor or embedding
layouts fail during configuration instead of silently loading incompatible
weights.

Load and save preserve language, vision, audio, projector, and modality-LoRA
weights. Tensor-parallel checkpoint conversion shards packed Q/K/V and gated
MLP sections independently. A saved training checkpoint can therefore be
reloaded for serving without exporting through a separate model format.

Serving
-------

Start the normal OpenAI-compatible endpoint:

.. code-block:: bash

   areno serve \
     --model-path /models/Phi-4-multimodal-instruct \
     --tp-size 1 \
     --world-size 1 \
     --max-running-prompts 1

Send audio with an OpenAI ``input_audio`` part or an ``audio_url`` part. Image
and audio parts can appear in the same user message; the processor selects the
combined vision/speech mode and keeps the expanded media-token slots aligned
during chunked prefill, decode, cache re-prefill, rollout, and log-probability
scoring. See :doc:`../concepts/multimodal-inputs` for the supported message
schema and media-input requirements.

Training
--------

The media tower and projector are frozen by default. The shared multimodal
training flags also apply to Phi-4:

.. code-block:: bash

   --unfreeze-mm-projector --mm-projector-lr 1e-5
   --unfreeze-mm-tower --mm-tower-lr 1e-6

The audio encoder belongs to the tower group and both speech/vision audio
projection heads belong to the projector group. Frozen groups remain in
evaluation mode. Unfreezing adds activation, gradient, and optimizer-state
memory, so validate the projector alone before training the full Conformer.

Dataset loaders should return structured messages rather than inserting media
tokens manually. Relative paths must be visible to every worker, and training
records should use the same message shape as serving requests.

Validation and limits
---------------------

AReno's CPU tests cover processor conversion, audio token alignment, mixed
batches, chunked prefill, cache re-prefill, rollout/log-probability packing,
tensor-parallel LoRA mapping, training gradients, and save/reload inference
parity. Native fused kernels and real checkpoint throughput still require a
CUDA environment with the AReno extension installed.

Audio features expand the prompt beyond its text length. Begin with one
running prompt and size ``--max-prompt-tokens``/``--max-context-len`` from the
processed token count. See :doc:`../concepts/multimodal-inputs` for the common
message schema and troubleshooting guidance.
