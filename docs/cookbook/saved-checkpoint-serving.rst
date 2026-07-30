Saved-checkpoint serving
========================

This tutorial walks through serving a saved AReno training checkpoint with
``areno serve``. It covers the environment requirements, tensor-parallel
sizing, port selection, and a minimal OpenAI-compatible request.

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

Serving requires the same CUDA environment as training:

* Linux x86_64 or aarch64 with an NVIDIA GPU
* CUDA-enabled PyTorch >= 2.6
* AReno installed with the CUDA extension built (``areno_accel``)

If you have not installed AReno yet, follow :doc:`/getting-started/installation`.
To verify the environment before serving, run:

.. code-block:: bash

   areno check

``areno check`` reports common setup problems — missing GPU, CPU-only PyTorch,
missing ``areno_accel`` — with concrete next steps. See
:doc:`/cli/diagnostics` for the full diagnostics reference.

.. note::

   AReno serving uses its own runtime path — the AReno engine with
   tensor-parallel workers, CUDA graph decode, and the OpenAI-compatible
   FastAPI server in ``areno/cli/serve.py``. It does **not** use vLLM, SGLang,
   or any external inference framework. The same engine that runs rollout
   during training also handles inference at serve time.

Step 1 — Train and save a checkpoint
------------------------------------

During training, pass ``--save-path`` to specify where AReno writes
checkpoints and ``--save-interval`` to control how often (in train steps)
they are written:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 1 \
     --save-path /tmp/areno-checkpoints \
     --save-interval 10

If ``--save-path`` is omitted, AReno prints a warning and no checkpoints are
written:

.. code-block:: text

   WARNING: no checkpoint output path configured (--save-path); checkpoints will not be saved.

AReno saves checkpoints in `Hugging Face-compatible safetensors format
<https://huggingface.co/docs/safetensors>`_. The output directory contains
weight shards (``model-*.safetensors``), a weight index
(``model.safetensors.json``), the tokenizer files, and the model config
(``config.json``). Because the layout is HF-compatible, the same directory
can be loaded by any HF-compatible tool — but AReno serving uses its own
engine, not an external framework.

You can also serve an existing checkpoint that was not produced by AReno
training, as long as it follows the HF safetensors layout and the model
family has a registered AReno adapter (see :doc:`/models/supported`).

Step 2 — Serve the checkpoint
------------------------------

Start the OpenAI-compatible server with ``areno serve``, pointing
``--model-path`` at the saved checkpoint directory:

.. code-block:: bash

   areno serve \
     --model-path /tmp/areno-checkpoints \
     --tp-size 1 \
     --world-size 1 \
     --host 0.0.0.0 \
     --port 8000

Expected startup
~~~~~~~~~~~~~~~~

The server loads the tokenizer and model weights, initializes the
tensor-parallel worker cluster, opens a rollout session, and binds the HTTP
listener. When the server is ready, it prints a Uvicorn startup line:

.. code-block:: text

   INFO:     Uvicorn running on http://0.0.0.0:8000

From this point the server accepts ``POST /v1/chat/completions`` requests
until the process is stopped.

Tensor-parallel sizing
~~~~~~~~~~~~~~~~~~~~~~

``--tp-size`` and ``--world-size`` control how the model is sharded across
GPUs:

* ``--tp-size`` — tensor parallel degree. The model's weight matrices are
  split across this many GPU ranks. Use a value that divides the model
  evenly; larger values reduce per-GPU memory but add communication overhead.
* ``--world-size`` — total local GPU ranks. Must be divisible by
  ``--tp-size``. Any remainder is used for data-parallel serving.

For a single-GPU machine, both default to ``1``:

.. code-block:: bash

   areno serve --model-path /tmp/areno-checkpoints

For a 4-GPU machine with TP=4:

.. code-block:: bash

   areno serve \
     --model-path /tmp/areno-checkpoints \
     --tp-size 4 \
     --world-size 4 \
     --port 8000

Port selection
~~~~~~~~~~~~~~

``--port`` selects the HTTP listen port (default ``8000``). ``--host``
selects the bind address (default ``0.0.0.0``, all interfaces). For local
testing, bind to localhost only:

.. code-block:: bash

   areno serve \
     --model-path /tmp/areno-checkpoints \
     --host 127.0.0.1 \
     --port 8000

Step 3 — Send a request
-----------------------

Once the server is running, send an OpenAI-compatible chat-completion
request with ``curl`` or any HTTP client:

.. code-block:: bash

   curl http://127.0.0.1:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{
       "model": "areno",
       "messages": [
         {"role": "user", "content": "Solve 12 * 13."}
       ],
       "max_tokens": 128,
       "temperature": 0.0
     }'

Expected response
~~~~~~~~~~~~~~~~~

The server returns a standard OpenAI chat-completion JSON envelope:

.. code-block:: json

   {
     "id": "chatcmpl-...",
     "object": "chat.completion",
     "created": 1234567890,
     "model": "areno",
     "choices": [
       {
         "index": 0,
         "message": {
           "role": "assistant",
           "content": "156"
         },
         "finish_reason": "stop"
       }
     ],
     "usage": {
       "prompt_tokens": 14,
       "completion_tokens": 3,
       "total_tokens": 17
     }
   }

The ``model`` field in the request is optional and echoed back in the
response. ``finish_reason`` is ``"stop"`` when generation hits the EOS token
or ``"length"`` when the token budget is exhausted.

Using the OpenAI Python client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any OpenAI-compatible client works. With the official ``openai`` package:

.. code-block:: python

   from openai import OpenAI

   client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
   response = client.chat.completions.create(
       model="areno",
       messages=[{"role": "user", "content": "Solve 12 * 13."}],
       max_tokens=128,
       temperature=0.0,
   )
   print(response.choices[0].message.content)

Step 4 — Serving options
------------------------

Beyond the basics, ``areno serve`` exposes several options that affect
serving behavior. See :doc:`/cli/inference` for the complete reference.

Decode progress logging
~~~~~~~~~~~~~~~~~~~~~~~

Set ``--decode-progress-interval-s`` to a positive value to print periodic
worker throughput:

.. code-block:: bash

   areno serve \
     --model-path /tmp/areno-checkpoints \
     --decode-progress-interval-s 5.0

Output looks like:

.. code-block:: text

   rollout decode progress: dp=0/1 active=1 cuda_graph=True tokens_per_second=58.3

``tokens_per_second`` is the scheduled decode throughput for that DP worker.
``cuda_graph=True`` means CUDA graph replay is active.

Eager decode
~~~~~~~~~~~~

Pass ``--eager-decode`` to disable CUDA graph decode and run eagerly. This is
useful for debugging or when CUDA graphs are not compatible with the current
GPU:

.. code-block:: bash

   areno serve \
     --model-path /tmp/areno-checkpoints \
     --eager-decode

Attention backend
~~~~~~~~~~~~~~~~~

``--attn-backend`` selects between ``flash`` (FlashAttention, default) and
``native`` (AReno's compatibility path). AReno automatically falls back to
``native`` on GPUs that do not support FlashAttention (such as Tesla T4) and
prints a warning:

.. code-block:: bash

   areno serve \
     --model-path /tmp/areno-checkpoints \
     --attn-backend native

Tool calls
~~~~~~~~~~

The server supports OpenAI-compatible function tool calls using the same
tool-call parser as agentic rollout:

.. code-block:: python

   from openai import OpenAI

   client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
   response = client.chat.completions.create(
       model="areno",
       messages=[{"role": "user", "content": "Choose a move: left or right."}],
       tools=[
           {
               "type": "function",
               "function": {
                   "name": "choose_move",
                   "parameters": {
                       "type": "object",
                       "properties": {
                           "direction": {"type": "string", "enum": ["left", "right"]},
                       },
                       "required": ["direction"],
                   },
               },
           }
       ],
       tool_choice={"type": "function", "function": {"name": "choose_move"}},
   )
   print(response.choices[0].message.tool_calls)

See :doc:`/cli/inference` for the full tool-call reference and supported
model families.

Troubleshooting
---------------

If the server does not start or requests fail, collect diagnostics first:

.. code-block:: bash

   areno check
   areno env --json

Common issues:

* **GPU not visible** — ensure ``nvidia-smi`` reports the expected GPUs and
  ``CUDA_VISIBLE_DEVICES`` is not restricting visibility.
* **Out of memory** — reduce ``--tp-size`` to shard across more GPUs, or use
  a smaller model. The ``--eager-decode`` flag can also change memory
  characteristics.
* **FlashAttention not supported** — AReno falls back to ``native``
  automatically; pass ``--attn-backend native`` to suppress the warning.
* **Port already in use** — choose a different ``--port``.

See :doc:`/troubleshooting/index` for more guides.

Summary
-------

1. Train with ``--save-path`` to produce an HF-compatible checkpoint.
2. Serve with ``areno serve --model-path <checkpoint-dir>``.
3. Send OpenAI-compatible requests to ``/v1/chat/completions``.
4. AReno serving uses its own engine — not vLLM or SGLang — with
   tensor-parallel workers and CUDA graph decode.

Key references:

* :doc:`/cli/inference` — full ``areno serve`` CLI reference
* :doc:`/cli/training` — full ``areno train`` CLI reference (including
  ``--save-path`` and ``--save-interval``)
* :doc:`/getting-started/installation` — environment setup
* :doc:`/cli/diagnostics` — ``areno check`` and ``areno env``