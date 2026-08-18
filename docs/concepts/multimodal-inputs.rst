Multimodal inputs
=================

AReno can pass image, audio, and video inputs through a checkpoint's native
processor during serving and training. The processor expands media placeholders
into model-specific token slots and produces the tensors consumed by the model
adapter. Using this shared path keeps media token expansion and feature
alignment consistent between ``areno serve`` and agentic rollout.

Available modalities depend on the checkpoint family. For example, Qwen3.5-VL
and MiniCPM-V accept visual input, while Gemma4 conditional-generation
checkpoints can expose image, audio, and video inputs. Consult
:doc:`../models/supported` and the checkpoint's processor configuration before
building a dataset.

They also depend on the selected backend. CUDA support is defined by AReno's
model adapters. On Apple Silicon, the checkpoint must be implemented by
``mlx-vlm`` and its processor must support the requested modality. A family
working on CUDA does not automatically imply an MLX implementation, or vice
versa.

Message format
--------------

A multimodal user message has a list of typed content parts. AReno accepts the
OpenAI-style ``image_url``, ``audio_url``, and ``video_url`` forms:

.. code-block:: json

   {
     "role": "user",
     "content": [
       {"type": "video_url", "video_url": {"url": "/data/clip.mp4"}},
       {"type": "audio_url", "audio_url": {"url": "/data/clip.wav"}},
       {"type": "text", "text": "Describe the synchronized event."}
     ]
   }

The equivalent direct processor forms use ``image``, ``audio``, or ``video``
as the type and provide the reference in ``url``. OpenAI-style base64 audio is
also accepted:

.. code-block:: json

   {
     "type": "input_audio",
     "input_audio": {"data": "<base64-data>", "format": "wav"}
   }

Local paths are resolved by the server or training worker, not by the client.
The files must therefore exist in the worker environment. Use data URLs when
the client and worker do not share a filesystem. Support for remote URLs and
individual media codecs is determined by the active processor and decoding
backend.

Serving
-------

Start an OpenAI-compatible server with any supported multimodal checkpoint:

.. code-block:: bash

   areno serve \
     --model-path /path/to/multimodal-checkpoint \
     --tp-size 1 \
     --world-size 1 \
     --max-running-prompts 1 \
     --port 8000

The standard OpenAI client can send structured content without an AReno-specific
request type:

.. code-block:: python

   from openai import OpenAI

   client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
   response = client.chat.completions.create(
       model="multimodal-model",
       messages=[
           {
               "role": "user",
               "content": [
                   {
                       "type": "image_url",
                       "image_url": {"url": "/data/example.png"},
                   },
                   {"type": "text", "text": "Describe this image."},
               ],
           }
       ],
       max_tokens=128,
   )
   print(response.choices[0].message.content)

Multimodal messages may include the normal Chat Completions ``tools`` and
``tool_choice`` fields. AReno selects the tool-call parser for the loaded model
family and returns parsed calls in ``message.tool_calls``. Keep related media
and instructions in the same user message so the processor's chat template
places modality tokens correctly.

Agentic training
----------------

A multimodal dataset loader returns structured messages in the same format as
serving. Reference answers belong in private source metadata consumed by the
reward function; placing them in the prompt leaks the target to the policy.

The AVE event-recognition example contains a complete audiovisual dataset
generator, loader, tool-calling agent, semantic reward, and GRPO command:

.. code-block:: text

   examples/multimodal/ave_event_recognition/

Visual agent examples are also available under ``examples/vl/``. These
examples demonstrate how to keep the message schema shared between dataset
generation, rollout, and serving.

Training behavior
-----------------

Model adapters may freeze pretrained media towers while training the language
model. This avoids updating large encoders and keeps frozen modules in
evaluation mode, while their output features still condition the trainable
language model. The exact trainable parameter set is model-specific.

Towers and projectors/mergers are frozen by default. Use
``--unfreeze-mm-tower`` and ``--unfreeze-mm-projector`` to train them, with
optional independent schedules through ``--mm-tower-lr`` and
``--mm-projector-lr``. Unfreezing either group adds activation, gradient, and
optimizer-state memory; start with the projector before unfreezing a large
encoder tower.

On MLX, processor output uses NumPy as the interchange format and is converted
to MLX arrays by the provider. Supported PIL image processors therefore do not
require Torch. Multimodal training uses the native lazy MLX graph rather than
compiling the full media forward, while text training may compile its train
step.

Media features can make the processor-expanded prompt much longer than its
text. Choose ``--max-prompt-tokens`` and ``--max-context-len`` from the encoded
sequence length rather than from text token counts alone. Start with
``--max-running-prompts 1`` to validate a new media pipeline, then increase
concurrency after measuring memory use and prefill latency.

Troubleshooting
---------------

* A missing-file error means the media path is not visible in the worker
  environment.
* A decoder or codec error means the active media backend cannot decode that
  container. Validate the file in the same Python environment used by AReno.
* Video inputs need valid frame-rate metadata. Browser-recorded files may need
  remuxing when their container omits it.
* Set ``ARENO_LOG_COMPLETIONS=1`` during agentic smoke tests to verify output
  and tool-call formatting before launching a long run.
* On Apple Silicon, reduce ``--mini-bs`` and ``--max-running-prompts`` first;
  use ``--drop-rollout-state`` when retained KV/cache state competes with
  training activations. See :doc:`../getting-started/mlx`.
