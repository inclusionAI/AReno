:orphan:

Metrics CLI reference
=====================

``areno metrics`` reads scalar tags and values from local TensorBoard event
files without starting a training run or loading a model.  When no metric name
is given, it lists all available scalar tags.  When a name is given, it prints
recent values, min/max, last value, mean, and a compact sparkline trend.

.. code-block:: bash

   areno metrics --log-dir /tmp/areno/tfevent

Example output:

.. code-block:: text

   Available metric tags:
     rollout/rewards_mean
     train/policy_loss

To query a specific metric:

.. code-block:: bash

   areno metrics --log-dir /tmp/areno/tfevent --name train/policy_loss

Example output:

.. code-block:: text

   Metric: train/policy_loss
     Points:    3
     Steps:     0 -> 2
     Min:       0.25
     Max:       1
     Last:      0.25
     Mean:      0.583333
     Trend:     ▆▃▁
     Recent:
       step      0: 1
       step      1: 0.5
       step      2: 0.25

For machine-readable output:

.. code-block:: bash

   areno metrics --log-dir /tmp/areno/tfevent --name train/policy_loss --json

The JSON output includes:

* ``name`` — metric tag
* ``count`` — number of valid data points
* ``first_step`` / ``last_step`` — step range
* ``min_value`` / ``max_value`` / ``last_value`` / ``mean_value`` — summary statistics
* ``sparkline`` — compact ASCII trend
* ``recent`` — list of recent ``{step, value, wall_time}`` points

Options
-------

``--log-dir`` (required)
    Directory containing TensorBoard event files.  The command searches
    recursively for ``events.out.tfevents.*`` files.

``--name``
    Metric tag to query.  Omit to list all available tags.  When the tag is not
    found, the command exits with an error and lists available tag names.

``--limit`` (default: 50)
    Maximum number of recent points to display.  Memory is bounded by this
    limit and the TensorBoard ``size_guidance`` setting.

``--json``
    Emit machine-readable JSON output instead of a human-readable table.

Limitations
-----------

* Reads only TensorBoard scalar tags; text, images, and histograms are not
  supported.
* NaN and Inf values are silently skipped.
* When the same step appears in multiple event files (e.g. after a training
  restart), the value from the last file is kept.
* The command does not modify stored data; it is read-only.