Kaggle dataset-mixing validation
================================

This guide records the reproducible Kaggle GPU validation for deterministic
weighted SFT dataset mixing introduced by Issue #198. It covers environment
setup, model preparation, a three-source training run, deterministic-plan
verification, filtering analysis, and the evidence that should be captured
from the Kaggle notebook.

The validation targets the data-mixing and SFT execution contracts. It is not
an evaluation of downstream model quality. Prompts, responses, credentials,
and Hugging Face tokens must not be included in screenshots or attached logs.

Validation summary
------------------

The tested implementation completed all of the following:

* loaded and normalized three public Alpaca-contract datasets;
* produced the requested 60/30/10 deterministic sample schedule;
* reproduced the same schedule hash across three independent runs;
* completed 500 single-GPU SFT optimizer steps;
* trained 1,000 accepted rows and 77,533 target tokens;
* exposed source-specific scheduled, filtered, trained, and token counts;
* completed without NaN, Inf, CUDA, or worker errors.

The implementation under test was commit ``e33e40d``. Later commits on the PR
branch only add or refine documentation.

Validated environment
---------------------

The run was captured from a fresh Kaggle notebook with Internet access and the
GPU accelerator enabled.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Component
     - Observed value
   * - Operating system
     - Linux 6.12, x86_64
   * - Python
     - 3.12.13
   * - PyTorch
     - 2.10.0+cu128
   * - PyTorch CUDA build
     - 12.8
   * - NVCC
     - 12.8 (V12.8.93)
   * - Accelerator
     - 2 x Tesla T4 available; the SFT run used one GPU
   * - Per-GPU memory
     - 15,360 MiB reported by ``nvidia-smi``
   * - GPU compute capability
     - 7.5
   * - Attention backend
     - ``native``
   * - Model
     - Qwen3-0.6B converted to FP16

.. admonition:: Screenshot K1 — Kaggle runtime and GPU environment

   Capture one output region containing the Python, PyTorch, CUDA, GPU model,
   GPU count, memory, and NVCC version. Include enough of the Kaggle page to
   show that the output came from a notebook cell, but crop unrelated browser
   controls. Suggested filename: ``kaggle-01-environment.png``.

1. Check out the PR branch
--------------------------

Clone the fork branch into the ephemeral Kaggle working directory:

.. code-block:: bash

   cd /kaggle/working
   git clone \
     --branch codex/issue-198-dataset-mixing \
     --single-branch \
     https://github.com/lkxdsb/AReno.git \
     AReno-issue198
   cd /kaggle/working/AReno-issue198

Record the exact source revision before installing:

.. code-block:: bash

   git branch --show-current
   git rev-parse HEAD
   git status --short --branch

The branch should be ``codex/issue-198-dataset-mixing`` and the working tree
should be clean. The exact commit can be newer than the validated
implementation commit when it contains documentation-only updates.

.. admonition:: Screenshot K2 — branch and source revision

   Capture the branch name, full commit SHA, and clean ``git status`` output in
   one image. This binds every later result to an exact source state.
   Suggested filename: ``kaggle-02-source-revision.png``.

2. Install AReno and compile the CUDA extension
-----------------------------------------------

Kaggle's T4 uses compute capability 7.5. Restricting the extension build to
``sm_75`` reduces build time and temporary disk use:

.. code-block:: bash

   export CUDA_HOME=/usr/local/cuda
   export TORCH_CUDA_ARCH_LIST=7.5
   export MAX_JOBS=2

   python -m pip install "setuptools>=69" wheel psutil ninja
   python -m pip install "flash-linear-attention>=0.2"
   python -m pip install -e . --no-build-isolation

Do not set ``ARENO_BUILD_EXT=0`` for this validation. Native attention still
uses the compiled ``areno_accel`` extension for training operations.

Run the built-in diagnostic after installation:

.. code-block:: bash

   areno check

The check must see CUDA, the model/runtime dependencies, and the compiled
extension. A successful package installation without ``areno_accel`` is not a
valid training environment. A missing ``flash_attn`` warning is acceptable for
this test because the command explicitly selects ``--attn-backend native``;
the diagnostic must still exit with code zero.

.. admonition:: Screenshot K3 — installation and AReno diagnostic

   Do not capture the full dependency installation log. Capture the final
   successful editable-build lines followed by the complete ``areno check``
   result. The extension status and CUDA status must be readable. Suggested
   filename: ``kaggle-03-areno-check.png``.

3. Prepare the FP16 checkpoint
------------------------------

The T4 run used a local FP16 copy of Qwen3-0.6B:

.. code-block:: python

   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_id = "Qwen/Qwen3-0.6B"
   output_dir = "/kaggle/working/qwen3-0.6b-fp16"

   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = AutoModelForCausalLM.from_pretrained(
       model_id,
       dtype=torch.float16,
       low_cpu_mem_usage=True,
   )
   model.save_pretrained(output_dir, safe_serialization=True)
   tokenizer.save_pretrained(output_dir)

   print("checkpoint:", output_dir)
   print("dtype:", next(model.parameters()).dtype)

Warnings about anonymous Hugging Face rate limits or tied-weight metadata do
not invalidate the conversion. The decisive evidence is a completed
``model.safetensors`` write and ``torch.float16`` parameter dtype.

.. admonition:: Screenshot K4 — model preparation

   Capture the completed model download/write progress together with the local
   checkpoint path and ``torch.float16`` output. Do not expose an ``HF_TOKEN``.
   Suggested filename: ``kaggle-04-fp16-checkpoint.png``.

4. Define the three-source mix
------------------------------

The GPU validation used three datasets accepted by the shared Alpaca loader:

.. list-table::
   :header-rows: 1
   :widths: 24 48 14 14

   * - Source name
     - Dataset reference
     - Weight
     - Rows available
   * - ``alpaca``
     - ``yahma/alpaca-cleaned``
     - 0.6
     - 51,760
   * - ``stanford``
     - ``tatsu-lab/alpaca``
     - 0.3
     - 52,002
   * - ``gpt4``
     - ``vicgalle/alpaca-gpt4``
     - 0.1
     - 52,002

All three sources are normalized by
``examples/sft/alpaca/dataset_loader.py`` into the SFT ``prompt`` /
``response`` contract. The source names are diagnostic identifiers and do not
need to match repository names.

Inline source weights are normalized automatically. The run used the inline
defaults:

* seed: ``42``;
* exhaustion: ``cycle``;
* shuffle within sources: enabled;
* weight unit: ``sample``;
* samples per epoch: sum of loaded source row counts when omitted.

5. Run the 500-step validation
------------------------------

The following is the command corresponding to the recorded 500-step result:

.. code-block:: bash

   areno train \
     --algo sft \
     --ckpt /kaggle/working/qwen3-0.6b-fp16 \
     --model-hub hf \
     --dataset-source alpaca=yahma/alpaca-cleaned:0.6 \
     --dataset-source stanford=tatsu-lab/alpaca:0.3 \
     --dataset-source gpt4=vicgalle/alpaca-gpt4:0.1 \
     --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
     --world-size 1 --tp-size 1 \
     --batch-size 2 --mini-bs 1 \
     --max-steps 500 \
     --max-prompt-tokens 256 --max-new-tokens 256 \
     --attn-backend native

This run intentionally omitted ``--save-path`` because its purpose was feature
validation rather than preserving a fine-tuned model. Add a save path for a
training run whose weights must survive process exit:

.. code-block:: bash

   --save-path /kaggle/working/qwen3-mixed-sft \
   --save-interval 500

The recorded plan used the automatic 155,764-row epoch budget. For a shorter
bounded plan, add:

.. code-block:: bash

   --dataset-mix-samples-per-epoch 5000

This option bounds scheduled rows. It does not limit source download/loading,
and it does not guarantee 5,000 accepted training rows. In this configuration,
``500 steps x batch size 2`` requires 1,000 accepted rows; filtering causes the
trainer to scan more than 1,000 scheduled rows to fill those batches.

.. admonition:: Screenshot K5 — resolved training config and mix plan

   Capture the resolved ``Algorithm``, ``Inputs``, ``Runtime``, ``Training``,
   and ``Outputs`` summaries plus the complete ``AReno dataset mix`` block.
   The image must show the seed, policy, planned rows, requested weights,
   selected rows, and termination reason. Suggested filename:
   ``kaggle-05-config-and-plan.png``.

6. Inspect the structured plan
------------------------------

When no explicit metrics directory is supplied, the default directory is
``/tmp/areno/tfevent``. Read the newest sample-free plan:

.. code-block:: python

   import glob
   import json
   import os

   plan_path = max(
       glob.glob("/tmp/areno/tfevent/dataset_mix_plan.*.json"),
       key=os.path.getmtime,
   )
   with open(plan_path, encoding="utf-8") as handle:
       plan = json.load(handle)

   print("mix_spec_hash:", plan["mix_spec_hash"])
   print("schedule_hash:", plan["schedule_hash"])
   print("planned_rows:", plan["planned_rows"])
   for source in plan["sources"]:
       print(
           source["name"],
           "requested=", source["weight_requested"],
           "selected=", source["rows_selected"],
           "observed=", source["observed_proportion"],
           "duplicates=", source["duplicates"],
       )

The recorded epoch-zero plan was:

.. list-table::
   :header-rows: 1
   :widths: 20 18 18 18 18

   * - Source
     - Requested
     - Selected
     - Observed
     - Duplicates
   * - ``alpaca``
     - 60%
     - 93,486
     - 60.0177%
     - 41,726
   * - ``stanford``
     - 30%
     - 46,775
     - 30.0294%
     - 0
   * - ``gpt4``
     - 10%
     - 15,503
     - 9.9529%
     - 0

The full-epoch ``cycle`` plan repeats the smaller high-weight Alpaca source
after its 51,760 unique rows are exhausted. The 500-step run consumed only the
first 1,171 scheduled rows, so this full-plan duplicate count does not mean
that the short validation consumed 41,726 duplicates.

.. admonition:: Screenshot K6 — structured plan

   Capture the compact plan-inspection output above. It must include both
   hashes and all three requested/selected/observed source rows. Suggested
   filename: ``kaggle-06-structured-plan.png``.

7. Verify deterministic replay
------------------------------

The 128-token run, 256-token run, and 500-step run all used the same source
contract and produced:

.. code-block:: text

   mix_spec_hash:
   sha256:fda6d7cbdab898b0ddafb6e6f37fa6c81b581da73b4bde7ebdfda8a0438057b4

   schedule_hash:
   sha256:26e61327f09b9c27a0e7219a40e2d17bebc5604571b47a1b8a3dc4119aad420b

Token limits do not participate in source scheduling, so changing 128 to 256
tokens must not change either hash. Compare all saved plans:

.. code-block:: python

   import glob
   import json
   import os

   for path in sorted(
       glob.glob("/tmp/areno/tfevent/dataset_mix_plan.*.json"),
       key=os.path.getmtime,
   ):
       with open(path, encoding="utf-8") as handle:
           plan = json.load(handle)
       print(os.path.basename(path), plan["schedule_hash"])

.. admonition:: Screenshot K7 — deterministic hashes

   Capture at least two independently generated plan filenames with identical
   schedule hashes. Three matching runs are preferred. Suggested filename:
   ``kaggle-07-deterministic-hashes.png``.

8. Compare token-limit filtering
--------------------------------

Before the final run, two 50-step runs compared 128-token and 256-token
prompt/response budgets.

.. list-table::
   :header-rows: 1
   :widths: 32 22 22

   * - Metric
     - 128 / 128
     - 256 / 256
   * - Scheduled rows
     - 165
     - 117
   * - Filtered rows
     - 65
     - 17
   * - Filter rate
     - 39.39%
     - 14.53%
   * - Trained rows
     - 100
     - 100
   * - Target tokens trained
     - 4,455
     - 8,295
   * - Target tokens per trained row
     - 44.55
     - 82.95

The 256-token configuration reduced filtering by 24.86 percentage points and
nearly doubled the supervised target-token volume without destabilizing the
single-GPU run. It was therefore selected for the 500-step validation.

.. admonition:: Screenshot K8 — 128-token final progress

   Capture the final ``stage=dataset_mix_progress`` event and the following
   ``stage=max_steps_reached`` line from the 128-token run. The source-specific
   scheduled, filtered, trained, and token counts must be readable. Suggested
   filename: ``kaggle-08-token-limit-128.png``.

.. admonition:: Screenshot K9 — 256-token final progress

   Capture the equivalent final progress and completion lines from the
   256-token run. Suggested filename:
   ``kaggle-09-token-limit-256.png``.

9. Inspect the 500-step result
------------------------------

The final progress event reported:

.. list-table::
   :header-rows: 1
   :widths: 34 22

   * - Metric
     - Result
   * - Optimizer steps
     - 500
   * - Scheduled rows consumed
     - 1,171
   * - Filtered rows
     - 171
   * - Overall filter rate
     - 14.60%
   * - Trained rows
     - 1,000
   * - Target tokens trained
     - 77,533
   * - Mean target tokens per trained row
     - 77.533
   * - Final completion stage
     - ``max_steps_reached``

Per-source effective contribution was:

.. list-table::
   :header-rows: 1
   :widths: 16 16 16 16 18 18

   * - Source
     - Scheduled
     - Filtered
     - Trained
     - Trained share
     - Token share
   * - ``alpaca``
     - 675
     - 142
     - 533
     - 53.3%
     - 62.32%
   * - ``stanford``
     - 366
     - 5
     - 361
     - 36.1%
     - 24.92%
   * - ``gpt4``
     - 130
     - 24
     - 106
     - 10.6%
     - 12.76%

The source acceptance rates differ: Stanford Alpaca filtered fewer rows than
the other two sources. This explains why trained-row proportions differ from
scheduled sample weights. Token proportions differ again because accepted
response lengths are source-dependent. The progress event makes both effects
observable rather than silently claiming that scheduled weights equal loss
contribution.

.. admonition:: Screenshot K10 — 500-step completion

   Capture the ``step=499`` ``train_stats`` line, the final
   ``stage=dataset_mix_progress`` object, and ``step=500
   stage=max_steps_reached``. Suggested filename:
   ``kaggle-10-500-step-result.png``.

10. Evaluate training stability
-------------------------------

Single-batch loss is noisy because batch size is two, so windowed statistics
are more informative than comparing only step zero and step 499.

.. list-table::
   :header-rows: 1
   :widths: 34 22 22

   * - Metric
     - First 50 steps
     - Last 50 steps
   * - Mean loss
     - 2.088
     - 1.530
   * - Median loss
     - 2.009
     - 1.521
   * - Mean gradient norm
     - 44.90
     - 26.92
   * - Mean train time per step
     - 0.927 s
     - 0.792 s

The first-to-last window mean loss decreased by approximately 26.7%. All
reported losses, gradients, learning rates, and timings remained finite. This
supports execution stability, but it must not be presented as a downstream
quality benchmark because no held-out evaluation set was used.

.. admonition:: Screenshot K11 — loss trend

   Prefer a TensorBoard loss chart covering all 500 steps. If TensorBoard is
   unavailable, capture notebook output from a small log-summary cell that
   prints the first-50 and last-50 mean/median losses. Do not use only two
   individual loss points as evidence. Suggested filename:
   ``kaggle-11-loss-trend.png``.

Evidence checklist
------------------

The recommended submission contains these eleven screenshots:

#. ``kaggle-01-environment.png`` — runtime, CUDA, and GPU identity;
#. ``kaggle-02-source-revision.png`` — branch, commit, and clean tree;
#. ``kaggle-03-areno-check.png`` — compiled extension and diagnostic;
#. ``kaggle-04-fp16-checkpoint.png`` — completed FP16 model preparation;
#. ``kaggle-05-config-and-plan.png`` — resolved config and human-readable plan;
#. ``kaggle-06-structured-plan.png`` — structured counts and hashes;
#. ``kaggle-07-deterministic-hashes.png`` — matching hashes across runs;
#. ``kaggle-08-token-limit-128.png`` — 128-token filtering result;
#. ``kaggle-09-token-limit-256.png`` — 256-token filtering result;
#. ``kaggle-10-500-step-result.png`` — final progress and completion;
#. ``kaggle-11-loss-trend.png`` — windowed or charted training-loss evidence.

If the PR should remain compact, K1, K2, K3, K5, K7, K10, and K11 form the
minimum defensible set. K4, K6, K8, and K9 provide useful reproduction and
parameter-selection detail.

Store approved images under
``docs/_static/kaggle-dataset-mixing/`` and replace each screenshot admonition
with a normal ``figure`` directive. Use lossless PNG for text-heavy notebook
output and crop large blank margins.

Artifacts and retained evidence
-------------------------------

The feature produces or references:

* ``dataset_mix_plan.<pid>.json`` — sample-free source plan and hashes;
* ``areno_run_config.<pid>.json`` — resolved run configuration;
* ``areno_run_config.<pid>.txt`` — human-readable run configuration;
* TensorBoard event files under the metrics directory;
* notebook output containing plan, progress, train statistics, and completion.

The validation intentionally did not retain a model checkpoint. A future
quality-evaluation run should configure ``--save-path`` and record the saved
checkpoint location and size separately.

Known limitations
-----------------

* V1 supports map-style datasets; streaming datasets are not accepted.
* The epoch index schedule is precomputed, so memory grows with planned rows.
* Weights are sample-based and apply before tokenization/length filtering.
* Source-specific filtering can change effective trained-row proportions.
* Source-specific response lengths can change target-token/loss contribution.
* The schedule hash covers source names and selected indices, not immutable
  verification of remote dataset contents.
* Exact deterministic replay starts at an epoch boundary. Mid-epoch optimizer
  state and dataset-cursor resume are outside the current contract.
* The Kaggle run validates single-GPU SFT. It does not validate multi-node,
  multi-GPU data parallelism, or model-quality generalization.
