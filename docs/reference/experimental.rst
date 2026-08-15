:orphan:

Experimental and auxiliary APIs
===============================

Experimental or auxiliary APIs should stay out of the new-user path until
their contracts are stable.

When documenting an experimental surface:

* Say which AReno version or branch it applies to.
* Link to the owning module or example.
* Mark the expected stability level.
* Keep migration notes close to the page that users will find by search.

Stable public surfaces should graduate into the relevant Reference page once
the contract is ready.

DAPO
----

``areno.experimental.dapo`` provides an experimental implementation of
`DAPO: An Open-Source LLM Reinforcement Learning System at Scale
<https://arxiv.org/abs/2503.14476>`_. It follows the four techniques described
by the paper and the maintained `verl DAPO recipe
<https://github.com/verl-project/verl-recipe/tree/main/dapo>`_:

* **Clip-Higher** uses independent lower and upper token-ratio clip values.
  The default interval is ``[1 - 0.2, 1 + 0.28]``.
* **Dynamic sampling** removes prompt groups whose raw scalar rewards are all
  equal, then samples candidate batches until ``batch-size`` qualified groups
  are available.
* **Token-level policy-gradient loss** divides by all response tokens in one
  optimizer step. Backend metadata preserves this global mean across unequal
  microbatches, gradient accumulation, and data-parallel gradient averaging.
* **Soft overlong reward shaping** linearly subtracts up to the configured
  penalty factor over the final response-length buffer.

The dynamic-sampling test always uses the raw reward returned by
``reward_fn``. Overlong shaping happens afterwards, so length penalties cannot
turn an otherwise constant group into an eligible group. A partial qualified
batch at dataset exhaustion is dropped, and reaching
``dapo-max-num-gen-batches`` raises with sampling counters rather than silently
changing the requested training batch size.

Example
~~~~~~~

.. code-block:: bash

   areno train \
     --algo dapo \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 8 \
     --dapo-gen-batch-size 16 \
     --n-samples 8 \
     --mini-bs 8 \
     --dapo-overlong-buffer-len 512 \
     --max-new-tokens 2048

DAPO defaults ``gradient-accumulation-steps`` to ``1``. That gives later
training microbatches a meaningful importance ratio against rollout
logprobs. Increase it explicitly when a larger optimizer batch is required;
the token-level denominator then covers the complete accumulation group.

Current limitations
~~~~~~~~~~~~~~~~~~~

* The interface is experimental and may change before it becomes a stable
  built-in algorithm.
* Only direct prompt rollouts are supported; ``--agent-fn`` is rejected.
* Dynamic sampling uses the scalar ``reward_fn`` output as its filter metric.
* The CPU suite verifies formulas, gradients, filtering, and batching. It does
  not claim reproduction of the paper's large-scale AIME result.
