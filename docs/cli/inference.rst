Inference CLI reference
=======================

``areno serve``

Start an OpenAI-compatible HTTP server backed by the Areno inference engine.
The server exposes ``/v1/chat/completions``, accepts standard chat-completion
``tools`` fields, and keeps one rollout session open for the process lifetime
so rollout state and CUDA graph state can be reused across requests.

.. code-block:: bash

   areno serve \
     --model-path /path/to/hf/checkpoint \
     --tp-size 1 \
     --world-size 1 \
     --host 0.0.0.0 \
     --port 8000

areno serve
-----------

Serve chat completions.

Options:

``--model-path TEXT``
   Local checkpoint/tokenizer path or Hugging Face repo ID. Required.

``--tp-size INTEGER``
   Tensor parallel size. Default: ``1``.

``--world-size INTEGER``
   Total number of local worker ranks. Default: ``1``.

``--host TEXT``
   HTTP bind host. Default: ``0.0.0.0``.

``--port INTEGER``
   HTTP bind port. Default: ``8000``.

``--max-running-prompts INTEGER``
   Maximum concurrent rollout prompts per request chunk. Default: ``128``.

``--default-max-tokens INTEGER``
   Default max generated tokens when requests omit a token budget. Default:
   ``1024``.

``--decode-progress-interval-s FLOAT``
   Worker decode progress log interval. Default: ``0.0``.

``--eager-decode``
   Disable decode CUDA graph and run rollout decode eagerly.

``--attn-backend [flash|native]``
   Attention backend. Default: ``flash``. Use ``native`` to run without
   ``flash-attn`` on the areno_accel native compatibility path. AReno
   automatically falls back to ``native`` on flash-attn-unsupported GPUs such
   as Tesla T4 and prints a warning. ``native`` is slower than ``flash`` on
   supported GPUs.

``--disable-thinking``
   Pass ``enable_thinking=False`` to tokenizer chat templates when supported.
   Use this when serving a model whose chat template supports a thinking-mode
   switch and you want normal responses without reasoning spans. Tokenizers
   that do not accept ``enable_thinking`` automatically fall back to their
   normal chat-template call.

``world-size`` must be divisible by ``tp-size``.

Examples
--------

Single-rank server
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   areno serve \
     --model-path /path/to/model \
     --tp-size 1 \
     --world-size 1 \
     --port 8000

TP4 server
~~~~~~~~~~

.. code-block:: bash

   areno serve \
     --model-path /path/to/model \
     --tp-size 4 \
     --world-size 4 \
     --port 8000

Chat completion request
~~~~~~~~~~~~~~~~~~~~~~~

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

Request fields
--------------

``POST /v1/chat/completions``

.. list-table::
   :header-rows: 1
   :widths: 24 26 50

   * - Field
     - Type
     - Description
   * - ``model``
     - ``str | None``
     - Optional model name echoed by the client.
   * - ``messages``
     - ``list[ChatMessage]``
     - Required chat messages.
   * - ``max_tokens``
     - ``int | None``
     - Generated token budget.
   * - ``max_completion_tokens``
     - ``int | None``
     - Alternative generated token budget.
   * - ``temperature``
     - ``float``
     - Sampling temperature. Defaults to ``1.0``; use ``0.0`` for greedy
       decoding.
   * - ``top_p``
     - ``float``
     - Nucleus sampling threshold.
   * - ``top_k``
     - ``int``
     - Top-k sampling threshold. Defaults to ``-1``; non-positive values
       disable top-k filtering.
   * - ``n``
     - ``int``
     - Number of completions per prompt.
   * - ``stream``
     - ``bool``
     - Streaming flag. ``true`` is not supported.
   * - ``stop``
     - ``str | list[str] | None``
     - Stop string or list of stop strings.
   * - ``seed``
     - ``int | None``
     - Deterministic sampling seed when sampling is enabled.
   * - ``tools``
     - ``list[Tool] | None``
     - OpenAI-compatible function tools. The same model-native tool-call
       parser used by agentic rollout converts generated tool-call text into
       ``message.tool_calls`` for supported model families.
   * - ``tool_choice``
     - ``str | dict | None``
     - Optional tool-choice directive, including a forced function name.

``ChatMessage`` fields:

``role``
   Usually ``system``, ``user``, ``assistant``, or ``tool``.

``content``
   Message content as ``str | list | None``.

Continuous batching behavior
----------------------------

The server runs inside a long-lived rollout session. Compatible requests can be
admitted into an active worker decode loop through continuous batching; requests
with different generation settings are scheduled separately. Requests are
compatible when these fields match:

* generated token budget
* temperature
* top-p
* top-k
* seed
* stop token ids
* EOS token id

Requests with different generation settings are scheduled separately.

Decode progress logs
--------------------

Set ``--decode-progress-interval-s`` to a positive value to print worker decode
progress:

.. code-block:: text

   rollout decode progress: dp=0/4 active=32 cuda_graph=True tokens_per_second=2810.7

``tokens_per_second`` is the scheduled decode throughput for that DP worker in
the reporting window. It excludes prefill and is not the same as end-to-end
request throughput. ``cuda_graph=True`` means the worker used CUDA graph replay
for at least one decode step in that window; ``False`` means the window ran
eagerly.

Tool calls
----------

``areno serve`` supports the Chat Completions tool-call shape and reuses the
same tool-call parser as agentic rollout:

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

Tool-call parsing is selected from the model/tokenizer family. Current parsers
cover Qwen/Qwen3.5/MiniCPM-style ``<tool_call>`` blocks, Gemma4 tool-call
blocks, and generic JSON tool-call output.

Help
----

.. code-block:: bash

   areno serve --help
