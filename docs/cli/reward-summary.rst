Reward Summary
==============

Summarise reward distributions from local training artifacts in the terminal.

Synopsis
--------

.. code-block:: bash

   areno reward-summary --metrics-log-dir <dir> [options]

Description
-----------

The ``reward-summary`` command reads ``reward_metrics.*.jsonl`` files
produced during training and prints summary statistics for reward
distributions: mean, standard deviation, min/max, zero fraction, missing
fraction, and a configurable outlier fraction.

Statistics are computed for both the **total** reward and any **named
reward components** that the reward function produced.  A reward function
that returns a plain ``float`` yields only the ``total`` row; one that
returns a ``dict[str, float]`` additionally produces one row per component
key.

Options
-------

``--metrics-log-dir <dir>``
    Directory containing ``reward_metrics.*.jsonl`` files.
    Default: ``/tmp/areno/tfevent``.

``--outlier-threshold <float>``
    A value is counted as an outlier when its absolute deviation from the
    mean exceeds ``threshold`` standard deviations.
    Default: ``3.0``.

``--step <int>``
    Only summarise records from this training step.

``--json``
    Emit machine-readable JSON instead of a table.

Output Fields
-------------

Each row in the table (or each entry in the JSON output) contains:

============ =================================================================
Field        Description
============ =================================================================
Count        Total number of samples (including missing/non-finite).
Mean         Arithmetic mean of finite reward values.
Std          Standard deviation of finite reward values.
Min          Minimum finite reward value.
Max          Maximum finite reward value.
Zero%        Fraction of samples whose reward is exactly 0.0.
Missing%     Fraction of samples that are missing, NaN, or infinite.
Outlier%     Fraction of samples deviating more than ``--outlier-threshold``
             std devs from the mean.
============ =================================================================

Named Reward Components
-----------------------

A reward function may return a ``dict[str, float]`` instead of a plain
``float`` to expose individual reward components::

   def reward_fn(record) -> dict[str, float]:
       return {
           "correctness": 1.0 if is_correct(record) else 0.0,
           "format": 0.5 if is_well_formatted(record) else 0.0,
       }

AReno sums all component values to obtain the scalar total used by the
training loop.  The individual component values are persisted to
``reward_metrics.*.jsonl`` and appear as separate rows in the summary.

Components that appear in some samples but not others are treated as
*missing* for the samples where they are absent — this distinguishes
"component not produced" (missing) from "component value is 0" (zero).

Example
-------

After a short training run::

   areno train --algo gspo --ckpt model --dataset-path data.jsonl \\
       --reward-fn-path reward.py --metrics-log-dir /tmp/areno/run1

Summarise the reward distribution::

   areno reward-summary --metrics-log-dir /tmp/areno/run1

Output::

    Reward Distribution Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     Component   Count      Mean       Std        Min        Max   Zero% Missing% Outlier%
     total          50    0.6200    0.4521     0.0000     1.5000 20.00%    0.00%    10.00%
     correctness    50    0.5000    0.5051     0.0000     1.0000 50.00%    0.00%     0.00%
     format         50    0.1200    0.3279     0.0000     0.5000 80.00%    0.00%     0.00%
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Samples: 50

JSON output::

   areno reward-summary --metrics-log-dir /tmp/areno/run1 --json