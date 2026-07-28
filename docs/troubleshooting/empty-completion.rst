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
   computation.

Observable output
~~~~~~~~~~~~~~~~~

When filtering is active, each training step emits log lines like:

.. code-block:: text

   completion_validation metrics={'completion_total': 8.0, 'completion_valid': 5.0, 'completion_invalid': 3.0, 'completion_invalid_empty': 2.0, 'completion_invalid_whitespace': 1.0, 'completion_filtered': 3.0}

When ``--save-path`` is set, dropped completions are also written to:

.. code-block:: text

   {save_path}/empty_completions.jsonl

Each line is a JSON record with the following fields:

==================  =========================================
Field               Description
==================  =========================================
``index``           Position in the original batch
``invalid_type``    One of the four classification types
``completion``      Decoded text (truncated to 500 chars)
``resp_token_count`` Number of response tokens
``prompt``          Original prompt text (truncated to 500 chars)
``policy``          The active policy (``filter``)
==================  =========================================

Covered algorithms
~~~~~~~~~~~~~~~~~~

The feature applies to all algorithms that perform rollout and reward
scoring:

* **GSPO** / **GRPO** (standard and agentic modes)
* **PPO**

SFT and DPO do not use rollout and are unaffected.

Configuration
~~~~~~~~~~~~~

The policy is stored in ``RolloutTrainerConfig`` and inherited by
``PolicyTrainerConfig`` (GSPO/GRPO) and ``PPOTrainerConfig`` (PPO):

.. code-block:: python

   from areno.api.trainer_config import PolicyTrainerConfig

   config = PolicyTrainerConfig(
       algo="gspo",
       ckpt="Qwen/Qwen3-0.6B",
       dataset_path="gsm8k:main",
       empty_completion_policy="filter",  # "off" (default) or "filter"
   )

Testing
~~~~~~~

CPU-only tests live in ``tests/test_completion_validator_cpu.py``:

.. code-block:: bash

   pytest tests/test_completion_validator_cpu.py -v
