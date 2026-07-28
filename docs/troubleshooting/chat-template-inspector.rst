.. _troubleshooting-chat-template-inspector:

Chat Template Compatibility Inspector
======================================

Before starting a training run with a new model, you can verify that the
model's chat template correctly renders all message types used by AReno's
training pipeline — **without loading model weights**.

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
      ...

For machine-readable output (e.g. scripting or CI):

.. code-block:: bash

    areno inspect chat-template --model Qwen/Qwen3-0.6B --output-format json

What it checks
---------------

The inspector renders five canonical message scenarios through the tokenizer's
chat template and runs five diagnostic checks:

1. **Missing template** — verifies ``chat_template`` is defined on the tokenizer.
2. **Role support** — renders scenarios with ``system``, ``user``,
   ``assistant``, and ``tool`` roles; catches templates that raise errors or
   silently drop content.
3. **Generation boundary** — verifies that ``add_generation_prompt=True``
   output is a prefix of the full conversation render.  A mismatch means the
   loss mask boundary will be incorrect during training.
4. **Tool schema** — for tool-call scenarios, checks that function names,
   arguments, and tool responses appear in the rendered text.
5. **Duplicate special tokens** — tokenizes the rendered text and checks for
   consecutive duplicate special token IDs, which may indicate a template bug.

Exit codes
----------

``0``
    All checks passed or only warnings were emitted.

``1``
    At least one check failed.  The failing check name and scenario are shown
    in the output.

Limitations
-----------

- The inspector loads **only the tokenizer** — it does not load model weights
  or verify training quality.
- It checks template rendering correctness, not inference accuracy.
- The canonical scenarios are fixed; if your training data uses unusual message
  formats not covered by the five scenarios, consider adding custom checks.