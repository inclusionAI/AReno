Inference CLI reference
=======================

``areno serve``

Start an OpenAI-compatible HTTP server backed by the Areno inference engine.
The server exposes ``/v1/chat/completions`` and batches compatible requests in a
continuous decode loop.

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

``--max-batch-prompts INTEGER``
   Maximum prompts to merge into one generate call. Default: ``128``.

``--batch-wait-ms FLOAT``
   Milliseconds to wait for compatible requests before starting a new decode
   session. Default: ``10.0``.

``--default-max-tokens INTEGER``
   Default max generated tokens when requests omit a token budget. Default:
   ``1024``.

``--decode-progress-interval-s FLOAT``
   Worker decode progress log interval. Default: ``0.0``.

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
     - Sampling temperature. Use ``0.0`` for greedy decoding.
   * - ``top_p``
     - ``float``
     - Nucleus sampling threshold.
   * - ``top_k``
     - ``int``
     - Top-k sampling threshold. ``0`` disables top-k filtering.
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

``ChatMessage`` fields:

``role``
   Usually ``system``, ``user``, ``assistant``, or ``tool``.

``content``
   Message content as ``str | list | None``.

Batching behavior
-----------------

The server batches compatible chat-completion requests. Requests are compatible
when these fields match:

* generated token budget
* temperature
* top-p
* top-k
* seed
* stop token ids
* EOS token id

Requests with different generation settings are scheduled separately.

Help
----

.. code-block:: bash

   areno serve --help
