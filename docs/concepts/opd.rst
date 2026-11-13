On-Policy Distillation
======================

On-Policy Distillation (OPD) trains a student model on the answers it has just
generated, with a frozen teacher scoring every answer token. It is available as
``--algo opd``.

Where it fits
-------------

SFT and DPO learn from a fixed dataset. PPO, GRPO and GSPO sample online but
compress an entire completion into a scalar reward, so every token of that
completion receives the same credit. OPD keeps online sampling but replaces the
single reward with a dense signal: the teacher's log-probability of each token
the student actually produced.

This makes OPD a natural fit for a framework that already colocates training
and inference. The teacher can score the sequences that were just generated
without a separate serving stack or an offline data pipeline.

Training loop
-------------

1. The student samples ``n_samples`` completions for each prompt.
2. The frozen teacher runs one teacher-forcing forward over those exact
   ``prompt + response`` token rows and returns a logprob for every position.
3. The trainer assembles training rows. The teacher scores are carried in the
   framework's frozen-reference slot: the CUDA pack builder converts it to
   ``packed_ref_logprobs`` on the response-token action view, while MLX reads
   ``ref_logprobs`` directly from its padded batch.
4. A token-level loss is computed only where ``packed_response_mask`` is set.
   Prompt and padding positions never contribute.
5. The optimizer step runs, and the next step samples again with the new
   weights. Teacher scores are always computed from the current step's rollout
   and are never reused across steps, which is what keeps the objective
   on-policy.

Objectives
----------

Two forms are available through ``--opd-loss-type``.

``pg`` (default) treats the difference between teacher and rollout logprobs as
a dense advantage, clips it to ``--opd-clip``, detaches it, and applies a
policy-gradient style loss:

.. code-block:: text

   A_t = clamp( log pi_teacher - log pi_old, -clip, clip )
   L   = -opd_coef * mean_mask( A_t * log pi_student )

``k3`` uses the low-variance reverse-KL estimator:

.. code-block:: text

   d_t     = clamp( log pi_teacher - log pi_student, -20, 5 )
   L       = opd_coef * mean_mask( exp(d_t) - d_t - 1 )

Only the exponent input is bounded, to prevent overflow. The loss output is
never clamped: a flat loss would zero the gradient precisely where the student
disagrees most with the teacher. Step size is left to the framework's existing
gradient clipping.

Tokenizer compatibility
------------------------

The teacher scores token ids produced by the student tokenizer. A mismatched
vocabulary, padding id or encoding would shift every per-token score while the
loss still looked finite. The trainer therefore compares vocabulary size,
``pad_token_id`` and a probe encoding at start-up and aborts on a mismatch.
``eos`` / ``bos`` are intentionally excluded: the teacher never generates text,
it only teacher-forces the token ids sampled by the student. This is what makes
a ``Qwen3-0.6B-Base`` student with a post-trained ``Qwen3-1.7B`` teacher valid,
even though their eos ids differ.

No reward function
------------------

``opd`` is registered with ``requires_rollout=True`` and
``requires_reward=False``. This distinction is explicit in ``AlgorithmSpec``:
rollout and reward requirements are different things. The CLI therefore does
not demand ``--reward-fn-path`` or ``--reward-ckpt`` for OPD, while existing
online algorithms keep their previous behavior unchanged.

A ``--reward-fn-path`` may still be supplied for the held-out generation
benchmark described below. It is never used by the training loss.

Scope of the first version
--------------------------

The teacher can come from an independent checkpoint (``--teacher-ckpt``) or
from the framework's existing reference role. Both share the student's prompt
prefix, so teacher and response positions align directly.

A self-teacher with privileged context needs an extra prefix, which changes the
sequence length and requires slicing and remapping the teacher scores back onto
the student response window. That is left as a later extension. Mixing an
external reward into the distillation loss and using the teacher's top-k
distribution are likewise out of scope for the basic version.

Held-out evaluation
-------------------

Two kinds of metrics are reported on the existing ``--eval-*`` schedule.

Task accuracy comes from a generation pass. When ``--reward-fn-path`` is
supplied, the trainer samples held-out prompts with the current student policy
and scores the completions through the standard ``reward_fn(record)`` contract.
This reuses the parent trainer's rollout and reward plumbing rather than a
separate evaluation stack, and greedy sampling keeps repeated evaluations
comparable. With a binary verifier, ``generation_accuracy`` is the task-level
result to compare before and after training.

Distillation progress comes from teacher-forcing fixed held-out sequences with
both the student and the frozen teacher, producing ``teacher_student_gap`` and
per-token NLL. It is a cheap monitor of whether the student is approaching the
teacher, not a task-level result. It remains available even without a verifier.

See :doc:`../cookbook/opd-training` for runnable commands.
