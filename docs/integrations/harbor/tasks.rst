Adapt a task
============

A task adapter defines what the agent receives and how its work is scored.
It is independent of the harness: switching Pi to another agent does not
require changing your dataset loader or grader. This guide completes the
``my-shell`` integration from :doc:`/integrations/harbor`, then shows how to
substitute a custom data source and reward.

1. Define the task record
-------------------------

Every task needs a nonempty ``prompt``. Other fields belong to your adapter;
the core does not prescribe a dataset format or register a default dataset.
For a file-editing task, a useful schema is:

.. code-block:: python

   import json
   from pathlib import Path

   task = {
       "prompt": "Write the number 42 to result.txt in the task workspace.",
       "files": {"result.txt": ""},
       "grader": {
           "entrypoint": "grade.py",
           "files": {
               "grade.py": (
                   "import json, sys\n"
                   "from pathlib import Path\n"
                   "answer = Path(sys.argv[1]) / 'result.txt'\n"
                   "correct = answer.exists() and answer.read_text().strip() == '42'\n"
                   "print(json.dumps({'reward': float(correct)}))\n"
               ),
           },
       },
       "max_turns": 8,
       "timeout": 900,
       "verify_timeout": 120,
   }
   Path("tasks.jsonl").write_text(json.dumps(task) + "\n")

``files`` is the agent-visible starting workspace. ``grader`` is trusted
verification input: the adapter uploads it after the agent finishes. Keep
answers, reference images and judge credentials out of ``files`` and the
prompt when the agent should not see them. With a local environment these
paths still share the host filesystem; use isolation when that matters.

The runner recognizes three optional budgets:

.. list-table::
   :header-rows: 1

   * - Field
     - Generic default
     - Scope
   * - ``max_turns``
     - 32
     - Model requests for a sample. Also passed to native agents whose option
       schema supports a turn limit.
   * - ``timeout``
     - 1800 seconds
     - Agent execution and sample model requests.
   * - ``verify_timeout``
     - 120 seconds
     - Task grading, including any wait for a grading slot.

An individual remote judge request can have a separate timeout. Dataset
generators and adapters may choose different defaults.

2. Reuse a task base class
--------------------------

Choose the smallest reusable contract that fits the task:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Contract
     - Implement
   * - ``HarnessDataset``
     - ``normalize(record)``, async ``prepare(attempt)`` and async
       ``verify(attempt)``. Use this protocol for tasks without file workspaces.
   * - ``FileWorkspaceDataset``
     - ``workspace_files(record)`` and ``verify(attempt)``. The base reuses
       relative-path validation and workspace transfer.
   * - ``WorkspaceDataset``
     - ``workspace_files(record)`` and ``verification(attempt)``. The base also
       uploads trusted grader files and executes a ``TaskVerification`` command.
       Override ``score(result)`` for a continuous reward.

The workspace bases and ``TaskVerification`` live in
``areno.agent.harness_base``. They do not choose a dataset, agent or verifier.
``WorkspaceDataset.score`` defaults to one for command exit code zero and zero
otherwise. For a grader returning JSON, save this adapter as ``task_adapter.py``:

.. code-block:: python

   import json
   from pathlib import PurePosixPath

   from areno.agent.harness_base import (
       TaskVerification,
       WorkspaceDataset,
       workspace_path,
   )
   from areno.api import register_dataset


   class FileTasks(WorkspaceDataset):
       def normalize(self, record):
           record = dict(record)
           if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
               raise ValueError("task requires a nonempty prompt")
           if not isinstance(record.get("files"), dict):
               raise ValueError("task requires a files mapping")
           grader = record["grader"]
           workspace_path(grader["entrypoint"])
           if grader["entrypoint"] not in grader["files"]:
               raise ValueError("grader entrypoint must be supplied in grader files")
           return record

       def workspace_files(self, record):
           return record["files"]

       def verification(self, attempt):
           handle = attempt.environment
           grader = attempt.item.record["grader"]
           entrypoint = PurePosixPath(handle.control_dir) / grader["entrypoint"]
           return TaskVerification(
               command=[handle.python, str(entrypoint), handle.workspace],
               files=grader["files"],
           )

       def score(self, result):
           if result.return_code != 0:
               raise RuntimeError("trusted task grader failed to execute")
           return float(json.loads(result.stdout)["reward"])


   register_dataset("file-tasks", FileTasks, replace=True)

This adapter's grader prints ``{"reward": score}`` and exits successfully for
both correct and incorrect submissions. A grader execution failure raises an
error instead of being confused with an incorrect answer. It can return
fractional credit, such as the proportion of checks passed, without changing
the harness or trainer.

``normalize`` runs during loading and again on a copied record for each
sample, so keep it deterministic and safe to repeat. ``prepare`` and
``verify`` receive ``attempt.item.record``, a temporary ``attempt.root`` and
``attempt.environment``. The environment handle supplies workspace/control
paths and Harbor's ``exec``, upload and download operations. A fresh adapter
is created for every sample; put shared judge concurrency limits outside
per-instance state.

3. Load your own dataset
------------------------

When your rows already use the adapter schema, use
``workflow.dataset_loader(path, default_loader=...)`` as in ``hooks.py`` from
:doc:`/integrations/harbor`. The CLI supplies the default source loader; the
runner applies your adapter's normalization to each row.

For a different source schema, add a reader to ``my_loader.py``:

.. code-block:: python

   import json


   def read_tasks(dataset_path, *, limit=None):
       with open(dataset_path) as source:
           for index, line in enumerate(source):
               if limit is not None and index >= limit:
                   break
               row = json.loads(line)
               yield {
                   "prompt": row["instruction"],
                   "files": row["starter_files"],
                   "grader": row["verification"],
                   "max_turns": row.get("turn_budget", 8),
               }

Append the reader configuration to ``my_integration.py`` so that all three
CLI hooks use it:

.. code-block:: python

   from my_loader import read_tasks

   workflow.loader_fn = read_tasks
   workflow.loader_kwargs = {"limit": 200}

Alternatively set ``workflow.loader_fn = "my_loader.py:read_tasks"``. The
default function name when the suffix is omitted is
``load_training_dataset``. A reader can return an iterable of mappings,
including a generator. ``loader_kwargs`` supplies defaults and invocation
options override them. Framework helpers such as ``default_loader`` are
passed only if the reader accepts them. With no custom reader, a
``default_loader`` is required; there is no hidden source fallback.

Keep downloading, source columns and split selection in this reader. Keep
task preparation and scoring in the dataset adapter. A custom CSV, database
or ModelScope loader can then reuse the same adapter. See
:doc:`/reference/dataset-loader-api` for the outer CLI hook and :doc:`pi` for
a custom CSV source using the SVG adapter.

4. Connect the reward and verify failures
-----------------------------------------

``verify(attempt)`` must return a dictionary containing a finite numeric
``reward``. Other entries are retained as grading details. The runner stores
this result in the sampled row's ``harness_result``; its
``reward_fn(record)`` reads that scalar through ``record.source_record``.
Use the wrapper from :doc:`/integrations/harbor` with the one-record, one-float
:doc:`/reference/reward-function-api` contract.

Decide how to handle an agent that reaches its turn/time budget or exits
unsuccessfully. The generic runner defaults to
``require_agent_success=True``, which forces those outcomes to zero. With
``False``, a trusted grader can score the submitted state. The Pi SVG example
uses this setting to give partial credit to incomplete but useful work.

An incorrect submission with a complete model trace is valid training data.
An environment, proxy or judge exception invalidates the sample through
``AgentTrajectory.invalid_items`` and removes its reward. The current
verification-budget timeout path instead returns reward zero and
``verify_timed_out=True``; inspect that flag separately from task quality.

Before training, exercise correct, partially correct and incorrect results,
then a broken grader. Confirm that the first three produce the intended
scores and the last produces an infrastructure error. For GSPO/GRPO, check
reward variation among samples of the **same prompt**: identical rewards
produce zero group-relative advantages, even if the batch mean is nonzero.

Persist artifacts and clean up
------------------------------

Set ``HarnessRunner.result_dir`` to keep a summary JSON and a trial directory
per attempt. The summary contains ``prompt_index``, ``sample_index``, ``result``
and the exact request/response ``trace``. Native logs and, when supported,
``agent/trajectory.json`` are copied from Harbor; your adapter can save
additional evidence in ``attempt.root / "artifacts"``.

``attempt_dir`` selects the temporary staging parent. For a Docker daemon
outside the controller, this parent must be on a path shared with the daemon
host. Staging directories are removed after completion; use ``result_dir``
for persistent evidence. The runner waits for trial/proxy cleanup before
returning to training. Harbor's Docker provider defaults to removing its
containers; configure its native retention options if needed.

Run a small integration check
-----------------------------

From the repository root, keep ``my_harness.py``, ``my_integration.py``,
``task_adapter.py``, ``hooks.py`` and ``tasks.jsonl`` together. The initial
task file uses the adapter schema, so use the default loader rather than the
optional source-remapping reader in step 3. Check loading without a GPU:

.. code-block:: bash

   python .agents/skills/areno-run-training/scripts/inspect_dataset.py \
     --dataset-path tasks.jsonl --loader hooks.py --algo gspo

Then use a local policy checkpoint that fits your GPU. This two-step check
exercises the new harness and task through the public training hooks:

.. code-block:: bash

   : "${MODEL_PATH:?Set the local policy checkpoint path}"
   python -m areno.cli.main train \
     --ckpt "$MODEL_PATH" --model-hub modelscope --algo gspo \
     --dataset-path tasks.jsonl --dataset-loader-fn hooks.py \
     --agent-fn hooks.py --reward-fn-path hooks.py \
     --world-size 1 --tp-size 1 --batch-size 1 --n-samples 4 \
     --max-running-prompts 2 --mini-bs 1 --score-micro-bs 1 \
     --max-context-len 8192 --max-prompt-tokens 6144 --max-new-tokens 1024 \
     --max-steps 2 --metrics-log-dir runs/my-harness/metrics

This deliberately simple task checks wiring, not task difficulty. Replace it
with tasks that produce different outcomes within prompt groups before
evaluating learning. Inspect the saved traces and graders, then use the
metric checks in :doc:`pi`.
