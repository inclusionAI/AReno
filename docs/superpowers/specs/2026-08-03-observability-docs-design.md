# Document the current observability surface — Design Spec

**Issue**: [#45](https://github.com/inclusionAI/AReno/issues/45) / [#122](https://github.com/inclusionAI/AReno/issues/122)
**Date**: 2026-08-03
**Status**: approved

## Goal

Enhance AReno's existing observability documentation so it covers:

- All metrics namespaces (`rollout/*`, `train/*`, `time/*`)
- Console logs and their verbosity controls
- TensorBoard event files and how to launch TensorBoard for a run
- Dashboard CLI commands and metrics-directory diagnostic artifacts
- Debugging workflows: how a user can read a metric programmatically
- Per-step `train_stats` fields and their meanings
- Agentic trajectory diagnostics and the `ARENO_LOG_COMPLETIONS` env var

The document must cite `areno.api.metrics`, `--metrics-log-dir`, and trainer logging paths. It must explicitly state that wandb is **not** currently supported.

## Audience

Users who are running training and need to understand what data is emitted, where it goes, and how to inspect it — both during training (console logs, dashboard) and after the fact (TensorBoard, JSONL artifacts, programmatic metric reading for debugging).

## Files to modify

### 1. `docs/cli/observability.rst` (primary deliverable)

Three additions to the existing document:

#### 1a. Console logs — log level control

Append a paragraph at the end of the Console logs section:

```rst
Adjust the verbosity with the ``ARENO_LOG_LEVEL`` environment variable
(default ``INFO``). Set it to ``DEBUG`` to see per-request decode progress
and fine-grained engine-level diagnostics:

.. code-block:: bash

   ARENO_LOG_LEVEL=DEBUG areno train ...

The log level is resolved once at startup by ``areno.engine.log`` and
applies to all submodules under the ``areno`` logger.
```

#### 1b. Reading metrics during debugging

New section inserted between "TensorBoard metrics" and "Agentic diagnostics":

```rst
Reading metrics during debugging
--------------------------------

You can read TensorBoard scalar series directly from Python without
launching the TensorBoard UI. This is useful when you want to inspect
metrics programmatically during debugging:

.. code-block:: python

   from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

   ea = EventAccumulator("/tmp/areno/tfevent")
   ea.Reload()
   for scalar_event in ea.Scalars("rollout/rewards_mean"):
       print(f"step={scalar_event.step}  reward_mean={scalar_event.value:.4f}")

   # Filter a specific tag to a plain list for plotting or analysis:
   values = [
       event.value for event in ea.Scalars("train/loss")
   ]
   print(f"loss over {len(values)} steps: min={min(values):.4f}  max={max(values):.4f}")
```

#### 1c. Dashboard section

New section appended after "Agentic diagnostics":

```rst
Dashboard
---------

AReno ships a lightweight local dashboard that surfaces training progress,
metrics, and rollout samples without requiring TensorBoard.

Launch it
~~~~~~~~~

Start the dashboard server on the default port (8765) and open the web UI:

.. code-block:: bash

   areno dashboard start

Pass ``--port`` to use a different port, or ``--no-browser`` to skip
opening the browser:

.. code-block:: bash

   areno dashboard start --port 9090 --no-browser

Stop a running dashboard server:

.. code-block:: bash

   areno dashboard stop

The dashboard reads from the same ``--metrics-log-dir`` directory that
feeds TensorBoard (default ``/tmp/areno/tfevent``). Every training run
that records metrics is automatically discoverable by the dashboard.

Metrics-directory artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``--metrics-log-dir`` is set, every training step writes a small set
of JSON and JSONL files alongside the TensorBoard event files:

``dashboard_state.{pid}.json``
   Per-step lightweight state snapshot with ``stage``, ``step``, ``epoch``,
   ``role``, and ``status`` fields. Used by the dashboard for low-latency
   progress tracking without parsing TensorBoard events. At most one file
   per process.

``rollout_samples.{pid}.jsonl``
   Decoded prompt/completion pairs written by ``MetricsRecorder``. Each
   line is a JSON object with ``kind`` (``"rollout"`` or ``"agentic"``),
   ``epoch``, ``step``, ``prompt_idx``, ``sample_idx``, prompt text, and
   sampled completion text. The number of samples logged per step is
   controlled by ``ARENO_LOG_COMPLETIONS`` (default 1).

``areno_run_config.{pid}.json``
   Run configuration snapshot in machine-readable form. Useful for
   reproducing a run or correlating metrics with hyperparameters.

``areno_run_config.{pid}.txt``
   Human-readable summary of the run configuration.

All files are scoped by ``pid`` so concurrent runs writing to the same
directory do not interfere with each other.
```

### 2. `docs/reference/environment-variables.rst`

Add two new entries:

```rst
``ARENO_LOG_LEVEL``
   Control the log level of the ``areno`` logger. Default is ``INFO``.
   Set to ``DEBUG`` for per-request decode progress and fine-grained
   engine diagnostics. Accepts standard Python log level names
   (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``). Resolved once at
   startup by ``areno.engine.log``.

``ARENO_LOG_COMPLETIONS``
   Number of decoded rollout samples to persist per training step (default
   ``1``). Samples are written to ``rollout_samples.{pid}.jsonl`` in the
   metrics log directory. Set to ``0`` to disable. For agentic rollouts
   the samples include rendered prompts, message lists, tool calls, and
   loss-mask summaries.
```

## What stays unchanged

- `docs/index.rst` toctree — observability.rst remains `:orphan:`. Existing
  cross-references from `docs/reference/cli.rst` and `docs/cli/training.rst`
  already guide users to it.
- All existing content in `docs/cli/observability.rst` — only additive changes.

## Acceptance criteria trace

| Criterion | How it's met |
|-----------|-------------|
| Cites `areno.api.metrics` | Already present in existing doc; preserved |
| Cites `--metrics-log-dir` | Already present; preserved |
| Cites trainer logging paths | Already present in Console logs section; preserved |
| Does not claim wandb support | Already stated; preserved |
| `rollout/*`, `train/*`, `time/*` namespaces | Already documented in TensorBoard section; preserved |
| Example of reading a metric during debugging | New "Reading metrics during debugging" section |
| Explain how to launch TensorBoard | Already present; preserved |
| Agentic trajectory diagnostics | Already present; env var now documented in environment-variables.rst |
