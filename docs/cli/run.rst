:orphan:

Run inspection CLI
==================

``areno run``

List and inspect AReno training and serving runs from the terminal. The command
reads the local job registry at ``~/.areno/dashboard-jobs.json``, which is
written automatically every time ``areno train`` or ``areno serve`` is launched
via the CLI or SDK. No database or external service is required.

areno run list
--------------

List active and recent runs in a compact table.

.. code-block:: bash

   areno run list

Example output:

.. code-block:: text

   ID           Kind   Status       Step Name                                     Age        Created
   ------------ ------ ------------ ------ ---------------------------------------- ---------- --------------------
   abc123def456 train  rollout         5 gspo Qwen/Qwen3-0.6B                   3m ago     2026-07-30T01:38:00
   def789abc012 train  succeeded      12 sft my-checkpoint                      1h ago     2026-07-30T00:41:00
   ghi345def678 serve  exited           0 serve my-model                         2d ago     2026-07-28T01:41:00

Columns:

* **ID** — short registry identifier (12 hex chars).
* **Kind** — ``train`` or ``serve``.
* **Status** — derived from PID liveness and ``dashboard_state`` files (see
  below).
* **Step** — last known training step (0 for serve runs).
* **Name** — run name from the registry.
* **Age** — human-readable elapsed time since the run started.
* **Created** — UTC timestamp of when the run was registered.

For machine-readable output:

.. code-block:: bash

   areno run list --json

The output is a JSON array with one object per run, including ``id``, ``kind``,
``name``, ``pid``, ``status``, ``stage``, ``step``, ``created_at``,
``updated_at``, ``age_s``, ``metrics_dir``, and ``returncode``.

To limit the number of entries:

.. code-block:: bash

   areno run list --limit 5

The default limit is 20. Use ``--limit 0`` to show all entries.

Status derivation
^^^^^^^^^^^^^^^^^

For each registered job, the status is determined as follows:

* **PID alive** and a ``dashboard_state.{pid}.json`` file with a ``stage`` field
  is present — the stage name is shown (e.g. ``rollout``, ``train``).
* **PID alive** but no state file — ``running``.
* **PID dead** with ``returncode == 0`` — ``succeeded``.
* **PID dead** with ``returncode != 0`` — ``failed``.
* **PID dead** with no returncode — ``exited``.

areno run info
--------------

Show detailed information for a single run, including configuration, metric
summaries, per-step time breakdown, and recent rollout samples.

Accepts either a registry ID or a directory path containing run artifacts.

.. code-block:: bash

   areno run info abc123def456

By directory path:

.. code-block:: bash

   areno run info /tmp/areno-metrics

For machine-readable output:

.. code-block:: bash

   areno run info abc123def456 --json

Example output (table mode):

.. code-block:: text

   Run: abc123def456
     Kind:        train
     Name:        gspo Qwen/Qwen3-0.6B
     Status:      running
     Stage:       rollout
     Step:        5
     PID:         12345
     Created:     2026-07-30T01:38:00+00:00
     Updated:     2026-07-30T01:41:00+00:00
     Metrics Dir: /tmp/areno-metrics

   Configuration:
     algo: gspo
     ckpt: Qwen/Qwen3-0.6B

   Metrics Summary:
     Name                                       Latest        Step  Count
     ---------------------------------------- -------- -------- ------
     loss                                       0.234567        5      50

   Time Breakdown (avg per step):
     rollout                                    1.50s
     train                                      2.25s

   Recent Rollout Samples (last 3):
     step=5 reward=0.8 resp_len=120
     step=4 reward=0.6 resp_len=85
     step=3 reward=1.0 resp_len=200

If the run ID is not found, the command exits with a non-zero status and prints
an error message.

.. code-block:: text

   Run not found: nonexistent
   Aborted!

Sensitive fields in the configuration (e.g. ``api_key``, ``token``,
``password``) are masked as ``***`` in both table and JSON output. Rollout
samples only show ``step``, ``reward``, and ``response_len`` — full prompt and
response text are never printed.

Limitations
-----------

* The registry is capped at 200 entries; older entries are dropped
  automatically.
* Status is inferred from PID liveness and local state files. If a process was
  killed externally without writing a returncode, its status will show as
  ``exited``.
* ``areno run info`` loads TensorBoard event files, rollout samples, and run
  config from the ``metrics_dir`` recorded at registration time. If the
  directory has been moved or deleted, those sections will be empty.
