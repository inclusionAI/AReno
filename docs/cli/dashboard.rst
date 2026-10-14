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
* **Step timeline** shows per-step timing segments when the trainer reports
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

Training detail workspace
-------------------------

Training jobs share an execution workspace in **Jobs > Open**. Ordinary jobs
show captured prompt/completion samples, media, loss, gradients, memory, and
training progress. Jobs without rollout samples still show their recorded
training metrics. Agentic jobs additionally show harness turns, tool calls,
file changes, and grading results. The sample matrix groups captures by
prompt or task and step. Select an attempt to replay its model turns,
thinking, tool arguments and outputs, code edits, and grader results. Use **+**
on another sample to compare attempts side by side. **Training signal** shows
the exact loss mask for each recorded model turn.

The compact **Step timeline** card stays fixed on the right while scrolling.
The card evenly samples at most ten records across the step history. Use
**Search step** to find an exact step in the complete history, including
records outside the sampled list. Select a step to filter its samples, or
**Clear filter** to return to all steps.
**AVG** shows mean phase durations over all recorded steps and clears the
step filter when selected. Hover or focus a row for its total and phase
durations. Collapse the card when needed; on small screens it starts collapsed at the
bottom right. Phase timings appear only in this card.

Below the conversation, inspect reward distributions, per-sample
scores, failure reasons, context growth, tool timing and execution order,
training loss, gradient norm, native logprob differences, memory residency,
MST progress, and task reward history. Select timeline or metric points to
filter the step; tool timing and execution-map items jump to tool output.
**Learning signal** separates prompt groups with varying rewards from all-zero,
all-perfect, other equal-reward, and insufficiently scored groups. **Sampling
funnel** tracks requested, returned, scored, accepted, and completed-training
episodes. One episode can produce several model-turn training rows; the funnel
keeps those counts separate. **Dataset mix & outcomes** reports source shares,
mean rewards, row counts, and context-token volume, including repeated history.
Select a source to filter the sample matrix.

**Run notices** brings reported incidents, skipped updates, and warning/error logs into the workspace. These views require
the corresponding histories from a current trainer; missing data remains empty.
No independent evaluation curves are added.

Reward summaries cover all matching samples. The sample matrix uniformly selects
at most ten steps, including the first and latest, and pages their samples.
Selecting a specific step exposes all its samples. Completed harness turns
appear live as they are constructed; tool cards replace repeated raw tool-call
markup. At conversation scroll boundaries, scrolling continues on the page.

Run with ``--metrics-log-dir`` to retain structured histories. Agentic histories
are independent of ``ARENO_LOG_COMPLETIONS``. Harness integrations can use ``Trainer.record_agent_event`` to report live
tool and grader details. Generic agent trajectories still provide model turns, rewards, masks, and
recorded tool calls/results after rollout. Tool execution durations and grader
diagnostics require explicit harness events. Unreported values remain empty. Modal jobs use the
same views when their AReno image includes structured event reporting. Ordinary
rollout captures also populate this workspace, including existing sample files;
``ARENO_LOG_COMPLETIONS`` controls how many ordinary completions are captured.

New events follow the conversation only while you are at the bottom. Scrolling
up preserves your position, and **New events** returns to the latest output.
Large outputs can be downloaded from their tool cards. CUDA allocations are
part of host memory on unified-memory systems; do not add the two series.

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

MST charts show recorded chunk sizes. Microbatch counters are not recorded in
the workspace. Progress reporting survives autograd thread changes and does
not introduce device synchronization.

The Pi secondary proxy reports each completed model turn before returning it to
the agent, with explicit prompt/sample identity. Deferred trajectory reconstruction
preserves training data and does not duplicate those live turn events.

Memory residency defaults to the last five minutes of recorded data, and MST
to the last minute. Both offer one-, five-, and fifteen-minute windows. The
windows follow the latest recorded timestamp, including for completed jobs.
