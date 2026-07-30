:orphan:

Empty completion handling
=========================

During RL rollout the model may produce completions that carry no real
content: empty strings, whitespace-only text, special-token-only output, or
immediate-EOS generations.  These degenerate completions pollute rewards and
gradients if they silently reach ``reward_fn`` or training code.

AReno provides a configurable completion validation step that classifies and
filters such completions **before** they enter reward computation.

Quick start
~~~~~~~~~~~

Add ``--empty-completion-policy filter`` to your training command:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --empty-completion-policy filter

The feature is **off** by default.  Existing runs are unaffected unless you
explicitly enable it.

Architecture
~~~~~~~~~~~~

The validation logic is designed for minimal intrusion into the training
code path:

* **``areno/engine/runtime/completion_validator.py``** — Standalone module
  that classifies individual completions and validates batches.  Uses only
  the Python standard library; zero additional dependencies.

* **``_filter_empty_completions``** — Wrapper method on ``PolicyOnlyTrainer``
  that is called **before** ``_materialize_train_batch``.  It inspects
  rollout results, applies the configured policy, and returns a clean list
  of ``RolloutResult`` objects.  The ``_materialize_train_batch`` method
  itself is **unchanged** from the main branch — no signature modifications,
  no inline validation logic.

* **``_filter_agentic_completions``** — Analogous wrapper for the agentic
  training path, operating on ``samples`` and ``reward_records``.

When ``--empty-completion-policy`` is ``"off"`` (the default), neither
wrapper is invoked.  The training code path is identical to the main branch.

Classification
~~~~~~~~~~~~~~

The validator classifies a completion as invalid when it matches any of the
following:

============================  ==================================================
Type                          Detection rule
============================  ==================================================
``empty``                     No response tokens, or decoded text is ``""``
``whitespace``                Decoded text contains only whitespace
``immediate_eos``             Response is a single token and it is an EOS token
``special_token``             Every response token is a special token
============================  ==================================================

Policies
~~~~~~~~

``--empty-completion-policy off``
   Default.  No validation; all completions reach ``reward_fn`` unchanged.

``--empty-completion-policy filter``
   Invalid completions are dropped before ``reward_fn``.  Valid completions
   in the same prompt group are preserved and used for reward and advantage
   computation.  If all completions for a prompt are invalid, the prompt is
   skipped with a warning.

``--empty-completion-policy resample``
   Invalid completions trigger re-generation of the entire prompt group
   (all ``n_samples`` completions) up to ``--empty-completion-resample-budget``
   times (default 3).  After the budget is exhausted, any remaining invalid
   completions are filtered.  Only supported for standard GSPO/GRPO and PPO
   training; agentic mode uses ``filter`` semantics.

Resample behavior
~~~~~~~~~~~~~~~~~

When ``--empty-completion-policy resample`` is active, each prompt group is
validated before ``reward_fn``:

1. Check all completions in the group.
2. If any are invalid, re-run rollout for that prompt (replacing all
   ``n_samples`` completions).
3. Repeat up to ``--empty-completion-resample-budget`` times.
4. After the budget is exhausted, apply ``filter`` as a fallback.

The resample budget is configured via:

.. code-block:: bash

   areno train ... --empty-completion-policy resample --empty-completion-resample-budget 5

Or via the SDK:

.. code-block:: python

   from areno.api.trainer_config import PolicyTrainerConfig

   config = PolicyTrainerConfig(
       algo="gspo",
       ckpt="Qwen/Qwen3-0.6B",
       dataset_path="gsm8k:main",
       empty_completion_policy="resample",
       empty_completion_resample_budget=5,  # default 3
   )

Parameter validation runs before model initialization:

* ``--empty-completion-policy`` must be one of ``off``, ``filter``, ``resample``.
* ``--empty-completion-resample-budget`` must be positive when policy is
  ``resample``.

Log output reference
~~~~~~~~~~~~~~~~~~~~

The following log messages may appear depending on the active policy and
outcome.  All messages are emitted at INFO or WARNING level.

**Normal operation (filter)**

.. code-block:: text

   completion_validation metrics={'completion_total': 8.0, 'completion_valid': 5.0,
     'completion_invalid': 3.0, 'completion_invalid_empty': 2.0,
     'completion_invalid_whitespace': 1.0, 'completion_filtered': 3.0}

**All completions invalid (filter)**

.. code-block:: text

   WARNING  all completions for prompt were empty or invalid; skipping this prompt
            (dropped=4 policy=filter prompt_preview='What is 1+1?').
            Consider disabling --empty-completion-policy if this happens frequently.

**Resample attempt in progress**

.. code-block:: text

   resample attempt 1/3: re-generating prompt with 2 invalid completions

**Resample succeeded**

.. code-block:: text

   resample succeeded after 2 attempt(s)

**Resample exhausted (falling back to filter)**

.. code-block:: text

   WARNING  resample exhausted after 3 attempt(s): 2 completions still invalid,
            falling back to filter

**PPO-specific variants** use the prefix ``ppo resample`` instead of
``resample``.

**Agentic path** uses ``agentic completion_validation metrics=...``.

Quarantine file
~~~~~~~~~~~~~~~

When ``--save-path`` is set, dropped completions are written to:

.. code-block:: text

   {save_path}/empty_completions.jsonl

Each line is a JSON record:

==================  =========================================
Field               Description
==================  =========================================
``index``           Position in the original batch
``invalid_type``    One of ``empty`` / ``whitespace`` / ``immediate_eos`` / ``special_token``
``completion``      Decoded text (truncated to 500 chars)
``resp_token_count`` Number of response tokens
``prompt``          Original prompt text (truncated to 500 chars)
``policy``          The active policy
==================  =========================================

Covered algorithms
~~~~~~~~~~~~~~~~~~

The feature applies to all algorithms that perform rollout and reward
scoring:

* **GSPO** / **GRPO** — ``filter`` and ``resample``
* **PPO** — ``filter`` and ``resample``
* **Agentic** (GSPO/GRPO + ``--agent-fn``) — ``filter`` only

SFT and DPO do not use rollout and are unaffected.

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   from areno.api.trainer_config import PolicyTrainerConfig

   config = PolicyTrainerConfig(
       algo="gspo",
       ckpt="Qwen/Qwen3-0.6B",
       dataset_path="gsm8k:main",
       empty_completion_policy="filter",            # "off" (default), "filter", "resample"
       empty_completion_resample_budget=3,          # max re-generation attempts
   )

Limitations
~~~~~~~~~~~

* **Rollout only.** The validator inspects completions produced by the model
  during RL rollout.  It does **not** filter empty or invalid records in
  supervised datasets (SFT/DPO).  Pre-filter your training data for those
  algorithms.

* **Quarantine file growth.** The ``empty_completions.jsonl`` file is appended
  to on every step.  It is not rotated or truncated automatically.  Monitor
  disk usage for long-running jobs.

* **Special-token classification depends on the tokenizer.** The
  ``special_token`` check uses ``tokenizer.all_special_ids`` and
  ``tokenizer.added_tokens_encoder``.  Different tokenizer versions or
  configurations may expose different special token sets, which can affect
  whether a completion is classified as ``special_token``.

* **GRPO/GSPO group size.** Filtering reduces the number of completions per
  prompt group.  If a group shrinks to a single completion,
  ``compute_group_advantages`` returns zero advantage (std=0), which is
  numerically safe but eliminates the group-normalization benefit for that
  prompt.

* **Resample replaces the entire group.** When resample re-generates, all
  ``n_samples`` completions for the prompt are replaced, not just the invalid
  ones.  Previously valid completions are discarded.

* **Agentic mode does not support resample.** Agentic trajectories involve
  multi-turn tool interactions; re-running the entire agent loop is not
  supported.  Agentic mode uses ``filter`` semantics even when
  ``--empty-completion-policy resample`` is set.

Testing
~~~~~~~

CPU-only tests live in ``tests/test_completion_validator_cpu.py``:

.. code-block:: bash

   pytest tests/test_completion_validator_cpu.py -v

The test suite covers 48 test cases across the following areas:

* **Classification** (8 tests) — all four invalid types, valid completions,
  edge cases.
* **Batch validation** (11 tests) — filter, resample metrics, quarantine
  file I/O, empty inputs, boundary conditions.
* **Special token extraction** (2 tests) — normal extraction and graceful
  handling of missing tokenizer attributes.
* **Configuration** (3 tests) — default values, PPO inheritance, SFT
  exclusion.
* **Quarantine records** (2 tests) — field completeness, ordering.
* **Reward function isolation** (3 tests) — verifying that invalid
  completions never reach ``reward_fn``.
* **Integration** (9 tests) — full ``_filter_empty_completions`` →
  ``_materialize_train_batch`` pipeline including resample retry success,
  budget exhaustion, and partial-success fallback.
* **Agentic path** (3 tests) — filter, all-invalid RuntimeError, off-policy
  no-op.
* **PPO path** (3 tests) — filter, all-invalid RuntimeError, off-policy
  no-op.
* **CLI validation** (4 tests + 17 subtests) — invalid policy rejection,
  valid policy acceptance, resample budget constraints.