Dashboard
=========

The AReno dashboard is a web interface for launching training and serving tasks,
preparing plans with an agent, and monitoring jobs. Local jobs and Modal jobs
share the same task forms and job views. Modal-specific setup is covered in the
:doc:`dashboard-modal` subtopic.

Start and stop
--------------

With AReno installed, start the background dashboard server:

.. code-block:: bash

   areno dashboard --start

Open ``http://127.0.0.1:8765``. To use a different port, pass ``--port``. To stop
the background server:

.. code-block:: bash

   areno dashboard --stop

For foreground operation, run:

.. code-block:: bash

   python -m areno.dashboard.server --host 127.0.0.1 --port 8765

Local training and serving require a supported local backend; starting the
interface does not provide one. The bundled frontend is ready to use. From a
repository checkout, rebuild changed frontend source with
``pnpm --dir dashboard build``. After changing dashboard Python code, restart
the server; a page refresh alone does not update the backend.

Find your way around
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - View
     - Purpose
   * - Overview
     - Review runtime health, active work, selected job information, and quick actions.
   * - Jobs
     - Browse tasks and open job details, metrics, samples, configuration, and logs.
   * - Runtime
     - Inspect the local environment and runtime checks before starting local work.
   * - Launcher
     - Configure a Train or Serve task using the shared forms and presets.
   * - Agent
     - Discuss tasks, inspect jobs through tools, and review proposed execution plans.

Launch training or serving
--------------------------

Open **Launcher** and choose **Train** or **Serve**. Select a preset or fill in
the model, dataset, algorithm, and execution parameters. Advanced fields expose
additional options without requiring a command-line invocation.

For training, use a dataset loader compatible with the selected algorithm and
actual dataset fields. A successful dataset download alone does not guarantee
that its rows satisfy the training schema. See
:doc:`../cookbook/writing-loaders-and-rewards` and
:doc:`../concepts/dataset-formats` for examples and field requirements.

Review runtime checks before launching a local task. GPU count, tensor
parallelism, model architecture, and available memory must agree. The checks
provide feedback; running a training or serving task still uses the configured
backend. After launch, open the task in **Jobs** to follow execution.

The **Run on Modal** switch uses the same Train/Serve form with additional cloud
resource fields. See :doc:`dashboard-modal` for its setup and confirmation flow.

Work with the agent
-------------------

Open **Agent**, configure its provider in **Settings**, and describe the task or
ask about a selected job. The agent can inspect live job information, metrics,
logs, and repository examples through dashboard tools. Supply the intended
model and dataset rather than relying on invented repository paths.

Review proposed task plans before confirming actions that start work. Modal
execution plans support inline parameter editing, addition, and deletion; saving
revalidates the plan. Chat preserves your position when you scroll up or focus
a plan field. Conversation history is available from the Agent history tab.

The **+** button next to the chat input opens managed dataset attachments for
Modal workflows. File and repository-URL import details are documented under
:doc:`dashboard-modal`.

Inspect jobs
------------

The **Jobs** list shows status, stage, step, metrics, and elapsed time. Open a job
for its detailed views:

* **Metrics** show recorded scalar series. Charts reduce rendered points for
  long runs while retaining the recorded metric history.
* **Runtime timeperf** shows per-step timing segments when the trainer reports
  timing metrics. It remains empty before those measurements are available.
* **Rollout Sample** shows captured prompts, completions, and associated record
  fields. One sample per rollout step is captured by default;
  ``ARENO_LOG_COMPLETIONS=0`` disables capture in the training environment.
* **Config** shows the task settings; multi-stage workflows display individual
  stage sections and their parameters.
* **Logs** show recent task output. The view follows new lines when you are at
  the bottom and preserves your position when you scroll up.

The stage indicator reflects reported trainer state. Elapsed time freezes when
a job stops, exits, fails, or succeeds. Use the stop action to terminate a task;
closing its detail view does not stop it. For metric tags, timing semantics, and
rollout sample contents, see :doc:`observability`.

Settings and troubleshooting
----------------------------

The dashboard gear and the Agent Settings button open the same settings panel.
Configure the **Agent provider** there. Modal credentials are a separate
connection in that panel; their persistence and reconnect behavior are covered
in :doc:`dashboard-modal`.

If a view remains empty, check that the intended job is selected, inspect its
status and logs, and verify that it has reached the stage that emits the data.
Local scalar recording uses ``--metrics-log-dir``. A stopped job will not produce
new training output. Runtime environment checks help diagnose local setup
issues; see :doc:`../troubleshooting/index` for backend-specific failures.

After an update, rebuild the frontend if its source changed, restart the Python
server if backend code changed, and refresh the page. Dashboard records and
Modal workflow data persist separately from the browser UI.

Modal jobs
----------

.. toctree::
   :maxdepth: 1

   dashboard-modal
