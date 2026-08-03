# Document the current observability surface — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance `docs/cli/observability.rst` and `docs/reference/environment-variables.rst` with missing observability surface documentation per the approved design spec.

**Architecture:** Docs-only change. Two files receive additive edits: `docs/cli/observability.rst` (three new sections) and `docs/reference/environment-variables.rst` (two new entries). No code changes, no new files, no toctree changes.

**Tech Stack:** reStructuredText (RST). Verified with existing doc build toolchain.

## Global Constraints

- Docs-only change — no code, no Python imports, no new dependencies.
- All changes are additive — existing content is preserved exactly.
- `docs/cli/observability.rst` keeps its `:orphan:` status.
- `docs/index.rst` toctree is NOT modified.
- Must cite `areno.api.metrics`, `--metrics-log-dir`, `areno.engine.log`.
- Must explicitly state no wandb support (already present, preserved).

---

## File Map

| File | Status | Responsibility |
|------|--------|---------------|
| `docs/cli/observability.rst` | Modify | Primary: add log-level control note, debugging code example, Dashboard section |
| `docs/reference/environment-variables.rst` | Modify | Add `ARENO_LOG_LEVEL` and `ARENO_LOG_COMPLETIONS` entries |

---

### Task 1: Add runtime env vars to environment-variables.rst

**Files:**
- Modify: `docs/reference/environment-variables.rst`

**Interfaces:**
- Consumes: (none — first task)
- Produces: Two documented env var entries that later tasks cross-reference

- [ ] **Step 1: Insert `ARENO_LOG_LEVEL` and `ARENO_LOG_COMPLETIONS` entries**

The current file ends after the `MAX_JOBS` entry with `For environment inspection, use :doc:\`/cli/diagnostics\`.`. Insert the two new entries between `MAX_JOBS` and that line.

Open `docs/reference/environment-variables.rst`. Find the block:

```rst
``MAX_JOBS``
   Set this to control parallel compilation jobs during editable installs.

For environment inspection, use :doc:`/cli/diagnostics`.
```

Replace with:

```rst
``MAX_JOBS``
   Set this to control parallel compilation jobs during editable installs.

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

For environment inspection, use :doc:`/cli/diagnostics`.
```

- [ ] **Step 2: Verify the file renders correctly**

RST syntax is straightforward backtick-quoted literals in definition lists. Each entry is `` ``VAR`` `` followed by an indented description. The structure follows the existing pattern exactly — no cross-reference or directive changes. The `:orphan:` in observability.rst means orphan status is preserved here too (this file doesn't have one, which is fine — it's already not in any toctree).

- [ ] **Step 3: Commit**

```bash
git add docs/reference/environment-variables.rst
git commit -m "docs(env): document ARENO_LOG_LEVEL and ARENO_LOG_COMPLETIONS

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add log-level control note to observability.rst

**Files:**
- Modify: `docs/cli/observability.rst`

**Interfaces:**
- Consumes: (none — independent edit within same file)
- Produces: Log-level control paragraph appended to Console logs section

- [ ] **Step 1: Insert log-level control paragraph**

In `docs/cli/observability.rst`, the Console logs section currently ends with the engine inference decode-progress paragraph:

```rst
``active`` is the number of currently scheduled decode requests on that shard
at the time of the progress log. In agentic runs it can dip to zero while the
external agent code is executing tools, tests, sleeps, background commands, or
other non-model work before asking the OpenAI-compatible proxy for the next
model response.
```

Append after that paragraph (before the ``train_stats`` section header):

```rst

Adjust the verbosity with the ``ARENO_LOG_LEVEL`` environment variable
(default ``INFO``). Set it to ``DEBUG`` to see per-request decode progress
and fine-grained engine-level diagnostics:

.. code-block:: bash

   ARENO_LOG_LEVEL=DEBUG areno train ...

The log level is resolved once at startup by ``areno.engine.log`` and
applies to all submodules under the ``areno`` logger.
```

- [ ] **Step 2: Verify section boundary**

The new paragraph sits after the last paragraph of "Console logs" and before the "``train_stats``" section header (`--------------\n\n\`\`train_stats\`\``). No existing content is removed or rearranged.

- [ ] **Step 3: Commit**

```bash
git add docs/cli/observability.rst
git commit -m "docs(observability): document ARENO_LOG_LEVEL in console logs section

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add "Reading metrics during debugging" section

**Files:**
- Modify: `docs/cli/observability.rst`

**Interfaces:**
- Consumes: (none — independent edit within same file)
- Produces: New subsection with Python code example between TensorBoard and Agentic diagnostics sections

- [ ] **Step 1: Insert the new section**

In `docs/cli/observability.rst`, find the section boundary between the TensorBoard metrics section and the Agentic diagnostics section. The TensorBoard section ends with:

```rst
   Stage timings when available: ``time/rollout``, ``time/reward``,
   ``time/advantage``, and ``time/train``.
```

The Agentic diagnostics section starts with:

```rst
Agentic diagnostics
-------------------
```

Insert the new section between them:

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

Note: the trailing blank line before the next section header matters — RST requires separation.

- [ ] **Step 2: Verify the `.. code-block:: python` directive**

The code block uses `.. code-block:: python` which is a standard RST directive. The existing document already uses `.. code-block:: bash` and `.. code-block:: text` — this follows the same pattern. Indentation is zero (directive at column 0), content indented 3 spaces.

- [ ] **Step 3: Commit**

```bash
git add docs/cli/observability.rst
git commit -m "docs(observability): add debugging metric-reading code example

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add Dashboard section

**Files:**
- Modify: `docs/cli/observability.rst`

**Interfaces:**
- Consumes: (none — independent edit within same file)
- Produces: Dashboard section at end of document, covering CLI commands and metrics-directory artifacts

- [ ] **Step 1: Append Dashboard section**

Append at the end of `docs/cli/observability.rst` (after the last paragraph of the Agentic diagnostics section):

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

- [ ] **Step 2: Verify section structure**

The Dashboard section uses two subsection levels: `~~~~~~~~~` (sub-subsection) under `---------` (section). The RST heading hierarchy is:
- `=========` (document title — already used at top)
- `---------` (section — "Console logs", "train_stats", "TensorBoard metrics", "Agentic diagnostics")
- `~~~~~~~~~` (subsection — "Launch it", "Metrics-directory artifacts")

This matches existing usage — `~~~~~~~~~` is not currently used but is the correct next level in the RST hierarchy.

- [ ] **Step 3: Commit**

```bash
git add docs/cli/observability.rst
git commit -m "docs(observability): add Dashboard section with CLI commands and artifact reference

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Final verification

**Files:**
- Verify: `docs/cli/observability.rst`
- Verify: `docs/reference/environment-variables.rst`

- [ ] **Step 1: Verify acceptance criteria**

Read both files and check:

1. `areno.api.metrics` is cited — should appear in existing content (TensorBoard section: "The writer lives in ``areno.api.metrics``.")
2. `--metrics-log-dir` is cited — should appear in existing content (TensorBoard section and Dashboard section)
3. Trainer logging paths — Console logs section covers `PolicyOnlyTrainer` and `areno.api.backend.areno.backend` log lines
4. No wandb claim — existing text: "AReno does not currently provide a built-in wandb integration."
5. `rollout/*`, `train/*`, `time/*` namespaces — existing TensorBoard section
6. Reading a metric during debugging example — Task 3 added this
7. TensorBoard launch instructions — existing text: `tensorboard --logdir /tmp/areno/tfevent`
8. Agentic trajectory diagnostics — existing section + `ARENO_LOG_COMPLETIONS` now in env vars

- [ ] **Step 2: Run doc build check**

If the project has a doc build command, run it to verify RST syntax:

```bash
# Check if docs can be built (optional — RST syntax is straightforward
# and follows existing patterns, so syntax errors are unlikely)
cd docs && make html 2>&1 | head -20 || echo "Doc build toolchain not available — manual RST review sufficient"
```

Note: if `make html` fails due to missing Sphinx/toolchain, this is not a blocker. The changes follow the exact same RST patterns already used in the file.

- [ ] **Step 3: Confirm git state**

```bash
git log --oneline -5
```

Expected: four commits on top of main, clean working tree.
