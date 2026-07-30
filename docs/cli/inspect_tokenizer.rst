Tokenizer inspector CLI reference
=================================

``areno inspect-tokenizer`` is a focused, read-only diagnostic that renders how a
tokenizer encodes plain prompts, chat messages, and tool calls, and checks that
the tokenizer vocabulary is aligned with the model ``config.json``. It does not
modify the tokenizer, load model weights, or initialize the AReno engine.

Use it to debug chat-template rendering, EOS placement, loss-mask spans, unknown
tokens, truncation, and vocab-size mismatches before an expensive training run.

.. code-block:: bash

   areno inspect-tokenizer --model /path/to/model --prompt "Hello 你好"

For a machine-readable report:

.. code-block:: bash

   areno inspect-tokenizer --model /path/to/model --prompt "Hi" --json

Inputs
------

Provide exactly one of:

* ``--prompt TEXT`` — a plain prompt string.
* ``--messages JSON`` — a JSON array of chat messages, e.g.
  ``[{"role":"user","content":"hello"}]``.
* ``--tool-call JSON`` — a JSON array of messages containing an assistant turn
  with ``tool_calls`` and a ``tool``-role reply, e.g.::

      [{"role":"assistant","content":null,
        "tool_calls":[{"id":"c1","type":"function",
                       "function":{"name":"get_weather","arguments":"{\"city\":\"x\"}"}}]},
       {"role":"tool","name":"get_weather","content":"sunny"}]

Options
-------

* ``--model PATH`` (required) — local tokenizer/model directory or hub id.
* ``--max-length INT`` — truncate the inspected token sequence to this length
  and report truncation.
* ``--enable-thinking / --disable-thinking`` — force the chat template
  ``enable_thinking`` switch (default: tokenizer default).
* ``--no-vocab-align`` — skip the tokenizer vs model config vocab-size check.
* ``--add-generation-prompt / --no-add-generation-prompt`` — append the
  assistant generation prompt for ``--messages`` (default on). Ignored for
  ``--prompt``.
* ``--json`` — emit a machine-readable JSON report.
* ``-h / --help`` — show usage.

Output fields
-------------

Human-readable mode prints a per-token table:

* ``idx`` — token index in the encoded sequence.
* ``id`` — token id.
* ``piece`` — ``convert_ids_to_tokens`` result (truncated to 16 chars).
* ``S`` — ``S`` marks a special token (BOS/EOS/PAD/template control), ``.`` otherwise.
* ``E`` — ``E`` marks an EOS token, ``.`` otherwise.
* ``role`` — chat turn role (``system``/``user``/``assistant``/``tool``), or empty
  for plain prompts and the generation-prompt header.
* ``loss`` — ``Y`` if the token is inside an ``assistant`` turn (counts toward
  training loss), ``N`` otherwise.

The summary line reports:

* ``round-trip`` — ``OK`` if ``decode(encode(input))`` reproduces the input text,
  else ``DIFF`` with a difference category (whitespace / length / content).
* ``vocab`` — ``OK`` if ``len(tokenizer)`` is within the model ``config.json``
  ``vocab_size`` (equal, or smaller when the model reserves padded vocab rows —
  common in Qwen-style configs aligned to TP sharding); ``FAIL`` if
  ``len(tokenizer)`` exceeds it (token ids would fall outside the embedding);
  ``SKIP`` if no model path or no ``vocab_size`` is available.
* ``truncated`` / ``has_unknown`` — whether truncation or an unknown token occurred.
* ``eos_positions`` — indices of EOS tokens, when present.
* ``WARNING`` lines — vocab mismatch and truncation details.

JSON mode emits the same data as a dict with keys ``kind``, ``raw``,
``segments``, ``eos_positions``, ``round_trip``, ``vocab_alignment``,
``truncated``, ``has_unknown``, and ``warnings``.

Defaults and backward compatibility
-----------------------------------

This command is additive: it does not change any existing tokenizer path,
training, serving, or dashboard behavior. When the feature is not invoked,
behavior is unchanged. ``enable_thinking`` defaults to the tokenizer default
(no mutation unless explicitly set).

Limitations
-----------

* Read-only: the tokenizer object and its on-disk config are never modified.
* Vocab alignment requires a ``config.json`` with ``vocab_size`` (or a nested
  ``text_config.vocab_size`` for multimodal models); otherwise the check is
  ``SKIP`` (not a failure).
* Role and loss-mask spans use a prefix-differencing approach over the chat
  template, so cost is quadratic in the number of turns — intended for
  debugging small prompts, not whole datasets.
* Hub ids are resolved via modelscope (default) or huggingface_hub and require
  network access; pass a local directory to avoid downloads.

Example
-------

Inspect a chat exchange with JSON output::

   areno inspect-tokenizer --model /path/to/model \
     --messages '[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]' \
     --json

Observable result: the ``segments`` array marks the ``assistant`` turn tokens
with ``loss_mask: true`` and the ``user`` turn with ``loss_mask: false``, and
``vocab_alignment.status`` is ``OK`` when the tokenizer and model vocab sizes
match.