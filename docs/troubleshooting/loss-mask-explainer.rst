.. _troubleshooting-loss-mask-explainer:

Loss Mask Explainer
====================

The loss mask determines which token positions contribute to the gradient
during SFT and agentic training.  The ``areno inspect loss-mask`` command
maps the packer-produced mask back to conversational structure — roles,
turns, and text previews — so you can verify which parts of a conversation
are being trained on.

Quick start
-----------

Create a JSON file with an OpenAI-style message list:

.. code-block:: json

    [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "What is 1+1?"},
      {"role": "assistant", "content": "2"}
    ]

Run the explainer:

.. code-block:: bash

    areno inspect loss-mask --model Qwen/Qwen3-0.6B --messages messages.json

Output (text mode):

.. code-block:: text

    Loss Mask Report: Qwen/Qwen3-0.6B
    Total tokens: 18, trainable: 4 (22.2%)
    ----------------------------------------------------------------------
    Role         Turn  Tokens   Train  Ratio  Text
    ----------------------------------------------------------------------
    system          0        7       0    0%  You are helpful.
    user            1        5       0    0%  What is 1+1?
    assistant       2        6       4  67%  2

For machine-readable output:

.. code-block:: bash

    areno inspect loss-mask --model Qwen/Qwen3-0.6B --messages messages.json --output-format json

To see full decoded text instead of truncated previews:

.. code-block:: bash

    areno inspect loss-mask --model Qwen/Qwen3-0.6B --messages messages.json --show-full-text


Command options
---------------

``--model`` (required)
    Model name or local checkpoint path.  Only the tokenizer is loaded;
    model weights are never read.

``--messages`` (required)
    Path to a JSON file containing the OpenAI-style message list.

``--show-full-text`` (optional, default: off)
    Show full decoded text per span.  By default, text previews are
    truncated to 50 characters to avoid exposing full training samples.

``--output-format`` (optional, default: ``text``)
    - ``text`` — human-readable terminal table.
    - ``json`` — machine-readable JSON object.


Output fields
-------------

Text mode
~~~~~~~~~

Each row in the table represents one conversational span:

- **Role** — ``system``, ``user``, ``assistant``, ``tool``, or ``trailing``.
- **Turn** — the message index in the original conversation.
- **Tokens** — total token count in this span.
- **Train** — number of trainable tokens (loss mask = True).
- **Ratio** — trainable / total, as a percentage.
- **Text** — decoded text preview (truncated unless ``--show-full-text``).

JSON mode
~~~~~~~~~

Top-level fields:

``model_name`` (string)
    The model identifier passed to ``--model``.

``total_tokens`` (int)
    Total token count in the sequence.

``trainable_tokens`` (int)
    Number of tokens with ``loss_mask = True``.

``overall_mask_ratio`` (float)
    ``trainable_tokens / total_tokens``.

``show_full_text`` (bool)
    Whether full text was requested.

``spans`` (array of objects)
    Each element has: ``role``, ``turn_index``, ``text_preview``,
    ``token_count``, ``trainable_count``, ``mask_ratio``, ``is_trainable``.


How it works
-------------

The explainer consumes the packer's actual output — token ids and the
loss mask — rather than reimplementing mask rules.  It renders each
message incrementally through ``apply_chat_template`` to find token
boundaries, then maps each contiguous region back to its role, turn
index, and text.  This ensures the explanation always matches the
trainer's real behaviour.

For SFT, only assistant tokens are trainable (system/user tokens are
masked out).  For agentic trajectories, the mask may additionally
suppress tool-result tokens depending on ``LossMaskPolicy``.


Limitations
-----------

- The inspector loads **only the tokenizer** — it does not load model
  weights or run the actual packer.  The loss mask is derived from role
  boundaries using the SFT convention.
- To inspect the packer's actual loss mask for agentic training, pass
  the packer output directly to ``LossMaskExplainer.explain()`` in
  Python rather than using the CLI.
- GPU is not required; the explainer runs entirely on CPU.  No minimal
  GPU validation is needed because no model weights are loaded.
- Text previews may contain special-token characters when
  ``skip_special_tokens=False`` is used during decoding.