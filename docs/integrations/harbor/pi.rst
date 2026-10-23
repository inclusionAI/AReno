Pi SVG walkthrough
==================

This walkthrough composes Harbor's native Pi agent, the SVG task adapter and
AReno's GSPO trainer. Use it as a working example of
:doc:`/integrations/harbor` and :doc:`tasks`: change the loader or grading
rubric for your data, or replace the harness while retaining the task adapter.

Prepare the runtime
-------------------

Start from the Python 3.12+/Harbor 0.24.0 setup in :doc:`/integrations/harbor`.
Pi, Node, Chromium and Python must be available inside the selected task environment.
The integration has been exercised with Node 22.19 and Pi 1.2.0. For that Pi
version, the npm package is ``@earendil-works/pi-coding-agent``. The controller
also needs the example's data/image dependencies: ModelScope, PyArrow and
Pillow. The example reuses preinstalled tools during rollouts.

For **local execution**, check the host tools and choose the provider:

.. code-block:: bash

   node --version
   pi --version
   export ARENO_HARBOR_ENVIRONMENT=local
   export ARENO_PI_PYTHON="$(command -v python)"
   export ARENO_PI_CHROMIUM="$(command -v chromium)"

Set ``ARENO_PI_CHROMIUM`` to the actual executable if your system calls it
``chromium-browser`` or installs it elsewhere. The local Pi wrapper gives
each sample separate home, config and temporary directories, disables GPU
visibility for its tools and keeps judge credentials out of its inherited
environment. It shares the host filesystem; it is not an OS sandbox.

For **Docker execution**, use an image containing those tools:

.. code-block:: bash

   export ARENO_HARBOR_ENVIRONMENT=docker
   export ARENO_HARBOR_IMAGE=my-pi-runtime:1.2.0
   export ARENO_PI_PYTHON=python
   export ARENO_PI_CHROMIUM=/usr/bin/chromium

The image name above is a user-built image, not a published AReno artifact.
Native Harbor Pi sources ``~/.nvm/nvm.sh`` when launching; ensure that setup
works in the image as well as ``pi --version``. Paths in these variables
refer to executables **inside** the task environment. A local run ignores a
leftover ``ARENO_HARBOR_IMAGE`` value. Pass extra native provider settings
through ``ARENO_HARBOR_ENVIRONMENT_OPTIONS`` as a JSON object; for example,
``{"proxy_connect_host": "controller.example.com"}`` when containers reach
the controller through that hostname. See :doc:`/integrations/harbor` for
routing details.

Configure the controller-side judge
-----------------------------------

Use an OpenAI-compatible VLM that accepts images and JSON-schema responses.
The VLM grades task outputs; AReno's policy model runs the agent. Configure
the judge independently in the controller shell:

.. code-block:: bash

   export ARENO_VLM_BASE_URL=https://vlm.example.com/v1
   export ARENO_VLM_MODEL=your-vision-model
   : "${ARENO_VLM_API_KEY:?Set the judge key in your shell or secret manager}"
   export ARENO_VLM_API_KEY
   export ARENO_VLM_TIMEOUT=300
   export ARENO_PI_GRADING_CONCURRENCY=8

Replace the example endpoint and model with your provider. Optional
``ARENO_VLM_REQUEST_OPTIONS`` is a JSON object of additional request fields.
The default request timeout is 60 seconds; the setting above uses 300 seconds.
Judge concurrency is shared across samples, separately from
``--max-running-prompts``. Keep keys out of data records, task files and agent
configuration.

Prepare ModelScope data
-----------------------

From the repository root, generate 10,000 short icon tasks:

.. code-block:: bash

   export DATASET_PATH="$PWD/data/svg.jsonl"
   python examples/agentic/pi/generate_dataset.py \
     --output "$DATASET_PATH" --limit 10000 --seed 0 --max-turns 8

The generator selects ``OmniSVG/MMSVG-Icon`` from ModelScope at revision
``0224c19f60f16ac3f3e7828eb418060fccfd634a``. It filters malformed, blank,
duplicate and overly long examples, writes the task JSONL and a provenance
report, and saves reference PNGs beside it in ``svg-references/``. For an
offline machine, supply local shards with repeated ``--parquet shard.parquet``
arguments. Move the reference files with the JSONL and update their absolute
paths if the destination changes.

Each task asks Pi to write ``icon.svg``, run the provided preview helper and
refine its output. The agent sees an empty starting SVG and preview tools.
The judge receives the **original source prompt**, reference image, rendered
candidate and rendering evidence. Reference images stay controller-side.
Generation defaults to eight turns and 900-second execution/verification
budgets. For a large concurrent run, adjust budgets in the task records; the
eight-GPU validation used ``timeout=1800`` and ``verify_timeout=3600``.

Check normalized records before loading the model:

.. code-block:: bash

   python .agents/skills/areno-run-training/scripts/inspect_dataset.py \
     --dataset-path "$DATASET_PATH" \
     --loader examples/agentic/pi/dataset_loader.py --algo gspo

Require ``ok: true`` and inspect the prompt, ``entrypoint``, source metadata,
turn budget and reference paths in a record.

Customize the data and reward
-----------------------------

For your own CSV with a ``brief`` column and optional ``reference_png``
column, save this as ``custom_pi.py``. It reuses SVG rendering and scoring;
only source mapping and the rubric change:

.. code-block:: python

   import csv
   from pathlib import Path

   from areno.api import register_dataset
   from examples.agentic.pi.run_agent import create_runner
   from examples.agentic.pi.svg_dataset import SVGIconDataset
   from examples.agentic.pi.visual_reward import RubricDimension


   def read_icons(dataset_path):
       path = Path(dataset_path).resolve()
       with path.open(newline="") as source:
           for row in csv.DictReader(source):
               brief = row["brief"].strip()
               reference = row.get("reference_png", "").strip()
               yield {
                   "prompt": f"Create an SVG icon: {brief}\nWrite icon.svg in the workspace.",
                   "source_prompt": brief,
                   "files": {"icon.svg": ""},
                   "reference_images": (
                       {"icon": str((path.parent / reference).resolve())} if reference else {}
                   ),
                   "max_turns": 8,
                   "timeout": 900,
                   "verify_timeout": 900,
               }


   def icon_adapter(**options):
       return SVGIconDataset(dimensions={
           "accuracy": RubricDimension("Match the original prompt's subject and style.", 0.7),
           "aesthetics": RubricDimension("Balanced, clean and legible as a small icon.", 0.3),
       }, **options)


   register_dataset("custom-icons", icon_adapter, replace=True)
   workflow = create_runner(dataset="custom-icons", loader_fn=read_icons)


   def load_training_dataset(dataset_path, **kwargs):
       return workflow.dataset_loader(dataset_path, **kwargs)


   async def run_agent(ctx, batch):
       return await workflow(ctx, batch)


   def reward_fn(record):
       return workflow.reward_fn(record)

Pass ``custom_pi.py`` for all three CLI hooks and your CSV as
``--dataset-path``. Missing, malformed, unrenderable or blank SVG submissions
receive zero before a VLM request. Visible submissions receive a weighted
continuous score; the built-in rubric covers semantic match, composition,
geometry, style and aesthetics. The custom example instead uses accuracy
and aesthetics. Rendering and VLM logic live under ``examples/agentic/pi/``;
they are not requirements of the reusable harness runtime.

Launch training
---------------

Set artifact locations in the controller. With an external Docker daemon,
the staging path must also exist on its host at the same path:

.. code-block:: bash

   export RUN_DIR="$PWD/runs/pi-svg"
   export ARENO_PI_RESULT_DIR="$RUN_DIR/results"
   export ARENO_HARBOR_ATTEMPT_DIR="$RUN_DIR/staging"

The following full-parameter configuration was exercised on eight H20 GPUs
with a local Qwen3.5-4B checkpoint. Set ``MODEL_PATH`` to your checkpoint.
Adapt GPU topology and batch/concurrency limits to available memory:

.. code-block:: bash

   : "${MODEL_PATH:?Set the local policy checkpoint path}"
   bash examples/agentic/pi/train.sh \
     --ckpt "$MODEL_PATH" --dataset-path "$DATASET_PATH" \
     --world-size 8 --tp-size 4 \
     --batch-size 32 --n-samples 8 --max-running-prompts 256 \
     --mini-bs 8 --score-micro-bs 8 \
     --max-context-len 16384 --max-prompt-tokens 12288 --max-new-tokens 4096 \
     --disable-thinking --drop-rollout-state --mem-frac 0.7 \
     --lr 1e-6 --epochs 4 --max-steps 1000 --save-interval 50 \
     --save-path "$RUN_DIR/checkpoints" --metrics-log-dir "$RUN_DIR/metrics"

The script selects GSPO, ModelScope and the example's dataset/agent/reward
hooks. For a first integration check, use a small dataset and override
``--max-steps 2``; retain multiple samples per prompt to check reward variation.
AReno's generation limit applies to each model request, while the task's
``max_turns`` controls the number of requests.

Inspect trajectories and the training signal
--------------------------------------------

For each attempt, ``$ARENO_PI_RESULT_DIR/<attempt>.json`` contains its
``prompt_index``, ``sample_index``, ``result`` and exact model ``trace``. The
matching directory contains Harbor logs and grading artifacts, including:

* ``agent/trajectory.json``: native Pi's ATIF export, when available.
* ``agent/pi.txt`` and Pi session logs: raw tool arguments and observations.
* ``artifacts/grading.json``: rendering evidence and rubric details.
* ``artifacts/submission/icon.svg``: the delivered submission, when captured.

Check ``valid`` and ``error`` before interpreting a reward. For zero scores,
inspect ``page_valid``, ``zero_reasons`` and ``verify_timed_out``. A missing
SVG caused by malformed tool arguments is a task failure. Browser startup,
judge transport and authentication errors are infrastructure failures; a
verification-budget timeout follows the zero-reward path described in
:doc:`tasks`.

Read optimizer and reward metrics with the repository helper:

.. code-block:: bash

   python .agents/skills/areno-run-training/scripts/read_metrics.py "$RUN_DIR/metrics" \
     --metric rollout/rewards_mean --metric rollout/rewards_std \
     --metric train/grad_norm --metric train/lr --metric train/logp_abs_diff_mean \
     --last 2

Confirm two optimizer updates, finite gradients and variation within at least
some prompt groups. The full-parameter integration validation completed two
updates with nonzero gradients and reward standard deviations of about
0.226 and 0.221. This establishes a working training signal; it does not
establish sustained reward improvement. A batch average can hide prompt
groups whose rewards are all identical, so group saved summaries by
``prompt_index`` when diagnosing zero advantages.

.. list-table:: Troubleshooting
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Inspect or change
   * - Every reward is zero
     - Inspect delivered SVGs, tool ``path``/``content`` arguments and
       ``zero_reasons`` first. Compare the rendering report with the original
       prompt before changing the rubric.
   * - Judge requests fail or time out
     - Check controller endpoint/model/key and service limits. The judge has
       bounded retries for HTTP 429/502/503/504. Lower grading concurrency;
       budget verification time for queueing as well as requests.
   * - Docker agent cannot reach the policy
     - Check the container-to-controller route and
       ``proxy_connect_host``. A loopback address inside an isolated container
       does not reach a separate controller.
   * - Workspace or container remains after a trial
     - Inspect cleanup logs and native retention settings. Staging is
       temporary; inspect saved results after cleanup rather than relying on
       a live task environment.
