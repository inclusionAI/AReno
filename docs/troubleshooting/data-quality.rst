:orphan:

Degenerate and empty training samples
======================================

AReno automatically detects and filters **degenerate training samples** — rows
that would produce no useful learning signal or cause silent training failures.

Detected degeneracy types
-------------------------

The following conditions are checked at two stages (pre-tokenization on raw
text, post-tokenization on token IDs):

============================ =========================================== ==================
Type                        Description                                 Stage
============================ =========================================== ==================
``empty``                    Prompt or response is an empty string       pre-tokenization
``whitespace_only``          Prompt or response contains only whitespace pre-tokenization
``special_tokens_only``      All prompt tokens are special tokens        post-tokenization
``no_trainable_tokens``      No trainable (non-prompt) tokens remain     post-tokenization
``identical_preference_branches`` DPO chosen and rejected are identical pre-tokenization
============================ =========================================== ==================

Policy configuration
--------------------

Use the ``--degenerate-policy`` CLI flag to control behaviour:

* ``--degenerate-policy skip`` (default): silently skip degenerate rows and
  log the reason counts. This preserves backward compatibility.
* ``--degenerate-policy error``: raise a ``ValueError`` on the first
  degenerate row, stopping training immediately. Use this when data quality
  must be enforced before training starts.

Where detection runs
--------------------

All three trainer paths share the same detection utilities from
``areno.api.data``:

* **Rollout path** (GSPO/GRPO/PPO): checked in ``Trainer.load_prompt_batches``
  on each prompt before tokenization and after tokenization.
* **SFT path**: checked in ``SFTTrainer._record_to_train_sequence`` on both
  prompt and response text (pre-tokenization) and on the final prompt mask
  (post-tokenization).
* **DPO path**: checked in ``DPOTrainer._record_to_train_pair`` on the prompt
  text and chosen-vs-rejected equality (pre-tokenization), and in
  ``_make_sequence`` on the prompt mask (post-tokenization).

All-degenerate datasets
-----------------------

If every row in the dataset is degenerate, training will **not** silently
succeed. The trainers raise a ``ValueError`` listing the reason counts so you
can fix the data before retrying.

Interpreting logs
-----------------

When rows are skipped, the log includes reason counts:

.. code-block:: text

   stage=sft_dataset_filter skipped_long_or_empty=3 degenerate_reasons=empty=1 whitespace_only=2

This means 3 rows were skipped total: 1 for empty content and 2 for
whitespace-only content.

Programmatic usage
------------------

The detection utilities are also available as a public API:

.. code-block:: python

   from areno.api.data import (
       DegenerateFilterConfig,
       DegeneratePolicy,
       check_prompt_text,
       check_response_text,
       check_tokenized_prompt,
       check_trainable_tokens,
       check_preference_pair,
       apply_degenerate_policy,
   )

   config = DegenerateFilterConfig(policy=DegeneratePolicy.ERROR)
   report = check_prompt_text(my_prompt)
   apply_degenerate_policy(report, config)  # raises if degenerate
