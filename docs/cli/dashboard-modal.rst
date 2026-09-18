Modal jobs
==========

Modal training and serving use the same Launcher, Agent, and Jobs views as local
tasks. Start the dashboard as described in :doc:`dashboard`, then install the
optional Modal SDK and dataset-preview dependencies in that Python environment
from the repository checkout:

.. code-block:: bash

   python -m pip install -r areno/dashboard/flow/requirements.txt

The control plane does not need a local CUDA GPU. There is no separate Areno
Flow application to start.

Configure connections
---------------------

Open **Settings** using the dashboard gear or the Agent tab's Settings button.
Enter **Modal Token ID** and **Modal Token Secret**, then connect. The server can
also read ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` from its environment.

**Remember credentials on this server and reconnect automatically** saves
validated credentials in a server-side file with owner-only permissions. The
dashboard reconnects on startup and retries transient connection failures.
Uncheck it for a session-only connection, or use **Forget saved credentials**
to delete the saved copy. Modal tokens are not stored in browser storage or
included in chat and job records.

Configure **Agent provider** in the same settings panel to use chat-based task
planning. Modal credentials and the agent provider connection are separate.

Prepare datasets
----------------

Use the **+** attachment button next to the Agent chat input to open dataset
management. Choose **Upload file** for JSON, JSONL, CSV, TSV, Parquet, or Arrow
files up to 16 MiB. For repository datasets, paste a dataset page URL such as:

.. code-block:: text

   https://huggingface.co/datasets/owner/name
   https://modelscope.cn/datasets/owner/name

These are dataset repository imports, not arbitrary file URL downloads.
Repository downloads and previews run in the background. Wait for the dataset
to be ready, then choose **Use in chat** to attach its managed ID. **Remove**
removes the library entry without removing assets already snapshotted for jobs.

Every agent training plan should include a compatible ``dataset_loader_fn``.
For RL, the loader must produce ``prompt`` and retain the reference answers or
metadata used by the reward function and agent. A raw dataset with ``question``
instead of ``prompt`` needs normalization even if its download succeeded.
See :doc:`../cookbook/writing-loaders-and-rewards` for loader examples.

Explicit loader paths survive managed dataset selection, plan editing, and
execution. A selected per-stage loader overrides the explicit path; an empty
loader selector does not erase an explicit path. Loader files must be available
inside the runtime, either from the image or staged as managed assets.

Launch from chat or the launcher
--------------------------------

In **Agent**, describe the Modal training or serving task and supply the model
and dataset. The agent reads the catalog and prepares a plan. The existing
Modal quick action also starts this workflow.

Alternatively, open **Launcher**, select **Train** or **Serve**, and enable
**Run on Modal**. The normal task form remains visible; the extra Modal fields
specify the model adapter, GPU type/count, CPU cores, host memory, and maximum
runtime. Model repository loading currently uses Hugging Face, even though
managed datasets support both Hugging Face and ModelScope.

Preparing a plan does not start a GPU sandbox. The plan appears inline in chat:

1. Review the commands, GPU reservation, memory assumptions, and estimated fee.
2. Choose **Edit parameters** to change, add, or delete fields. Use AReno parameter
   names; required fields must remain valid. Choose **Save changes & estimate**
   to validate the edited plan and refresh its estimates.
3. Choose **Confirm execution** to start the billable sandbox.

Plans can execute once and persist across dashboard restarts. A plan expires
after 30 minutes; edit and save it to revalidate before execution. Scrolling up
or focusing a plan field pauses chat auto-scroll.

GPU recommendations and costs
-----------------------------

The launcher can automatically select a GPU from the model and task settings.
Agent plans use the same recommendation service. Training estimates default to
Adam 4-bit, consistent with the default training optimizer. This reduces
optimizer-state memory; it does not quantize the model weights to four bits.

The estimate includes BF16 weights and gradients, compact master-weight
metadata, packed first moments and factored variance, activation and KV-cache
allowances, and 20% headroom. Sequence length, microbatch size, rollout
concurrency, algorithm, and tensor parallelism affect the estimate. Increasing
GPU count without increasing ``tp_size`` does not imply model sharding.

The suggested GPU is the lowest GPU list cost among estimated fits at the
configured world/TP size. You can select resources manually. Parameter counts
inferred from model names are approximate; use **Total model parameters** to
supply an override for custom models, including all experts for MoE models.
Unknown sizes are reported rather than guessed.

These are planning heuristics, not measurements from a CUDA probe. They do not
deduct LoRA or data-parallel optimizer-sharding savings. Review the assumptions
and actual GPU memory usage for the selected architecture.

The fee estimate includes reserved GPUs, CPU, and host memory for the selected
duration. If live public rates cannot be refreshed, the UI identifies cached
rates and their verification date. **Usage so far** appears in the job list and
job details. It is a compute estimate, not an invoice, and excludes storage,
networking, and billing adjustments. Elapsed time stops advancing when a job
stops, exits, fails, or succeeds.

Monitor execution
-----------------

Open a Modal job in **Jobs** to view its stage, configuration, metrics, runtime
timeperf, rollout samples, logs, and usage. Multi-stage workflows keep metric
series distinct and display each stage's parameters separately.

* Metrics retain the full recorded scalar history while charts reduce the
  number of rendered points for long runs.
* Runtime timeperf derives per-step segments from reported timing metrics.
  No timing row appears before timing metrics are emitted.
* Trainer state updates drive the rollout, score, train, save, and done
  indicators. Older sandbox logs can supply compatible trainer state messages.
* One rollout sample per step is captured by default. Set
  ``ARENO_LOG_COMPLETIONS=N`` in the training environment to change the limit,
  or ``0`` to disable capture. See :doc:`observability` for sample contents.
* Logs follow new output when you are at the bottom and preserve your position
  when you scroll up. New Modal sandboxes relay carriage-return progress output
  as flushed lines. Sandbox startup may precede any runtime output; the UI shows
  that waiting phase separately.

For serving, the dashboard generates an endpoint API key and displays it once
after execution. Save that key before dismissing the dialog; it is not persisted
locally.

Runtime images, persistence, and troubleshooting
------------------------------------------------

New plans resolve ``ghcr.io/inclusionai/areno:latest`` to a digest. If the
``latest`` tag is absent, resolution falls back to the highest stable release.
The reviewed digest remains pinned through execution. Modal image setup installs
``tilelang`` for the reported Hopper gated-backward issue with Triton versions
from 3.4.0 up to, but excluding, 3.7.1. Existing sandboxes retain their original
image; relaunch a failed job to use an updated setup.

Job records, datasets, uploads, plans, and scalar history live under
``.areno-modal/`` by default. ``ARENOFLOW_DATA_DIR`` overrides that directory.
Preserve it when restarting the server so saved credentials and job history
remain available. Reconnection reattaches active sandboxes; it does not launch
replacement jobs.

If a fee request reports ``Unsupported Modal route``, restart the updated Python
backend. Refreshing only the frontend can leave an older server running. If
training reports ``dataset row must contain prompt``, check the selected loader
and its output schema, then prepare a corrected plan. If logs or metrics remain
empty, first inspect the job's startup phase and status; failed or stopped jobs
will not produce new training output.
