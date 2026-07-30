.. _troubleshooting-chat-template-inspector:

Chat Template Compatibility Inspector
======================================

Before starting a training run with a new model, you can verify that the
model's chat template correctly renders all message types used by AReno's
training pipeline — **without loading model weights**.

The inspector loads only the tokenizer, renders five canonical message
scenarios through the chat template, and runs five diagnostic checks.  It
produces both a human-readable terminal table and a structured JSON report.


Quick start
-----------

.. code-block:: bash

    areno inspect chat-template --model Qwen/Qwen3-0.6B

Output (text mode):

.. code-block:: text

    Chat Template Inspection: Qwen/Qwen3-0.6B
    Overall: PASS
    21 checks: 21 passed, 0 failed, 0 warnings.
    ------------------------------------------------------------
      [OK  ] missing_template (_global)
      [OK  ] role_support (single_turn_basic)
      [OK  ] generation_boundary (single_turn_basic)
      [OK  ] tool_schema (single_turn_basic)
      [OK  ] duplicate_special_tokens (single_turn_basic)
      [OK  ] role_support (multi_turn)
      [OK  ] generation_boundary (multi_turn)
      [OK  ] tool_schema (multi_turn)
      [OK  ] duplicate_special_tokens (multi_turn)
      [OK  ] role_support (tool_call_request)
      [OK  ] generation_boundary (tool_call_request)
      [OK  ] tool_schema (tool_call_request)
      [OK  ] duplicate_special_tokens (tool_call_request)
      [OK  ] role_support (no_system_role)
      [OK  ] generation_boundary (no_system_role)
      [OK  ] tool_schema (no_system_role)
      [OK  ] duplicate_special_tokens (no_system_role)
      [OK  ] role_support (empty_assistant)
      [OK  ] generation_boundary (empty_assistant)
      [OK  ] tool_schema (empty_assistant)
      [OK  ] duplicate_special_tokens (empty_assistant)

For machine-readable output (e.g. scripting or CI pipelines):

.. code-block:: bash

    areno inspect chat-template --model Qwen/Qwen3-0.6B --output-format json

Output (JSON mode, abbreviated):

.. code-block:: json

    {
      "model_name": "Qwen/Qwen3-0.6B",
      "overall_status": "pass",
      "summary": "21 checks: 21 passed, 0 failed, 0 warnings.",
      "results": [
        {
          "check_name": "missing_template",
          "scenario_name": "_global",
          "status": "pass",
          "message": "",
          "detail": {"has_chat_template": true}
        },
        {
          "check_name": "role_support",
          "scenario_name": "single_turn_basic",
          "status": "pass",
          "message": "",
          "detail": {}
        },
        ...
      ]
    }


Command options
---------------

``--model`` (required)
    Model name or local checkpoint path.  Only the tokenizer is loaded;
    model weights are never read.  Remote references are resolved via
    ModelScope (default) or Hugging Face when ``--model-hub`` is set on
    the parent command.

``--output-format`` (optional, default: ``text``)
    - ``text`` — human-readable terminal table (default).
    - ``json`` — machine-readable JSON object for scripting and CI.


Output fields
-------------

Text mode
~~~~~~~~~

Each line in the report has the format::

    [STATUS] check_name (scenario_name)

Where:

- **STATUS** is ``OK`` (pass), ``FAIL`` (fail), or ``WARN`` (warning).
- **check_name** is one of: ``missing_template``, ``role_support``,
  ``generation_boundary``, ``tool_schema``, ``duplicate_special_tokens``.
- **scenario_name** is one of: ``_global`` (for the template-existence
  check), ``single_turn_basic``, ``multi_turn``, ``tool_call_request``,
  ``no_system_role``, ``empty_assistant``.

When a check fails or warns, an indented message line follows explaining
the issue, e.g.::

    [FAIL] generation_boundary (single_turn_basic)
           add_generation_prompt output is NOT a prefix of the full
           conversation render for scenario 'single_turn_basic'.
           Loss mask boundary will be incorrect.

JSON mode
~~~~~~~~~

The JSON object contains the following top-level fields:

``model_name`` (string)
    The model identifier passed to ``--model``.

``overall_status`` (string)
    One of ``"pass"``, ``"fail"``, ``"warning"``.  ``"fail"`` if any
    check failed; ``"warning"`` if no failures but warnings exist;
    ``"pass"`` if all checks passed.

``summary`` (string)
    A one-line summary, e.g. ``"21 checks: 21 passed, 0 failed, 0 warnings."``

``results`` (array of objects)
    Each element has:

    - ``check_name`` (string) — the diagnostic check name.
    - ``scenario_name`` (string) — the scenario that was tested.
    - ``status`` (string) — ``"pass"``, ``"fail"``, or ``"warning"``.
    - ``message`` (string) — human-readable detail; empty on pass.
    - ``detail`` (object) — structured diagnostic fields, e.g.
      ``{"has_chat_template": true}`` or
      ``{"missing_parts": ["function_name:get_weather"]}``.


Exit codes
----------

``0``
    All checks passed or only warnings were emitted.

``1``
    At least one check failed.  The failing check name, scenario, and
    error message are shown in the output.


What it checks
---------------

The inspector renders five canonical message scenarios through the
tokenizer's chat template and runs five diagnostic checks:

1. **Missing template** — verifies ``chat_template`` is defined on the
   tokenizer.  If absent, all further checks are skipped (fast-fail).

2. **Role support** — renders scenarios containing ``system``, ``user``,
   ``assistant``, and ``tool`` roles.  Catches templates that raise
   errors on unsupported roles or silently drop message content.

3. **Generation boundary** — verifies that the output of
   ``apply_chat_template(..., add_generation_prompt=True)`` is a prefix
   of the full conversation render.  A mismatch means the loss mask
   boundary will be incorrect during training, causing gradients to be
   computed on the wrong tokens.

4. **Tool schema** — for scenarios containing ``tool_calls`` and ``tool``
   role messages, checks that function names, arguments, and tool
   response content appear in the rendered text.  Catches templates that
   silently ignore tool-call messages.

5. **Duplicate special tokens** — tokenizes the rendered text and
   checks for consecutive duplicate special token IDs, which may
   indicate a template bug that inflates sequence length or corrupts
   token boundaries.


Canonical scenarios
-------------------

The five fixed scenarios cover all message types in AReno's training
pipeline:

``single_turn_basic``
    system + user + assistant — basic single-turn with a system prompt.

``multi_turn``
    user + assistant (2 turns) — multi-turn conversation.

``tool_call_request``
    user + assistant (with ``tool_calls``) + tool + assistant — full
    tool-call cycle including function invocation and response.

``no_system_role``
    user + assistant — conversation without a system message (some
    templates do not support the ``system`` role).

``empty_assistant``
    user + assistant (empty content) — boundary case for empty
    assistant responses.


Limitations
-----------

- The inspector loads **only the tokenizer** — it does not load model
  weights or verify training quality.
- It checks template rendering correctness, not inference accuracy.
- The canonical scenarios are fixed; if your training data uses unusual
  message formats not covered by the five scenarios, consider adding
  custom checks.
- The ``duplicate_special_tokens`` check uses ``tokenizer.encode()``
  without ``add_special_tokens=False``; some tokenizers may add extra
  special tokens that could affect the result.
- GPU is not required; the inspector runs entirely on CPU.  No minimal
  GPU validation is needed because no model weights are loaded.