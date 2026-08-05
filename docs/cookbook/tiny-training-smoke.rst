Tiny training smoke test
========================

This tutorial turns the smallest official AReno training command into a
step-by-step guide. It covers prerequisites, the command itself, what the
output looks like, how to inspect metrics with TensorBoard, and the failures
you are most likely to hit.

.. note::

   The smoke test verifies **wiring**, not model quality. It confirms that the
   CLI can load a model, build batches, execute rollout, score with a reward
   function, run one optimizer step, and write outputs — all on a single GPU
   with the smallest possible configuration. A successful run means your
   environment is correctly set up; it does **not** mean the model has learned
   anything meaningful.

Prerequisites
-------------

Hardware
~~~~~~~~

An NVIDIA GPU with CUDA support is **required**. The smoke test uses
``Qwen/Qwen3-0.6B`` with tensor-parallel size 1, so a single GPU with at least
8 GB of VRAM is sufficient.

.. warning::

   CPU-only machines **cannot** run the AReno training engine. You can install
   the package on a CPU-only machine for docs and metadata checks, but
   ``areno train`` will fail at model initialization. If you see an error like
   ``CUDA is not available`` or ``No NVIDIA GPU detected``, switch to a
   CUDA-capable machine.

Software
~~~~~~~~

Make sure AReno is installed and the environment passes diagnostics:

.. code-block:: bash

   areno check

``areno check`` prints ``OK``, ``WARN``, and ``FAIL`` statuses for common
setup issues such as missing CUDA, CPU-only PyTorch, missing ``CUDA_HOME``,
unavailable ``nvcc``, or a missing ``areno_accel`` extension. All items
should read ``OK`` (``WARN`` is acceptable for optional dependencies) before
you proceed.

If you have not installed AReno yet, follow the
:doc:`installation guide </getting-started/installation>`.

Repository checkout
~~~~~~~~~~~~~~~~~~~

The smoke test references files from the ``examples/math/`` directory, so you
need a local checkout of the AReno source tree:

.. code-block:: bash

   git clone https://github.com/inclusionAI/AReno.git
   cd AReno

Run the smoke test from the repository root so that the relative paths
``examples/math/dataset_loader.py`` and ``examples/math/math_verify_reward.py``
resolve correctly.

Run the smoke test
-------------------

Execute the following command from the repository root:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 1

What this command does
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Flag
     - Role in the smoke test
   * - ``--ckpt Qwen/Qwen3-0.6B``
     - Smallest official checkpoint; loads quickly and fits in a single GPU.
   * - ``--dataset-path gsm8k:main``
     - GSM8K math dataset, fetched from the default model hub (ModelScope).
       Add ``--model-hub hf`` to pull from Hugging Face instead.
   * - ``--dataset-loader-fn examples/math/dataset_loader.py``
     - Normalizes raw GSM8K rows into the prompt/response schema the trainer
       expects.
   * - ``--reward-fn-path examples/math/math_verify_reward.py``
     - Scores completions by extracting the answer inside ``\boxed{}`` and
       checking it against the ground truth.
   * - ``--algo gspo``
     - Group Sequential Policy Optimization — the default RL algorithm.
   * - ``--tp-size 1 --world-size 1``
     - Single-GPU configuration, no distributed setup needed.
   * - ``--batch-size 1``
     - One prompt per step — the absolute minimum for a wiring check.

Expected output
---------------

When the smoke test succeeds, you will see output similar to the following.
Exact numbers (token counts, timings, reward values) will vary.

Training config
~~~~~~~~~~~~~~~

AReno first prints the resolved training configuration so you can verify
all paths and parameters:

.. code-block:: text

   ========= AReno training config =========
   ckpt: Qwen/Qwen3-0.6B
   dataset_path: gsm8k:main
   dataset_loader_fn: examples/math/dataset_loader.py
   reward_fn_path: examples/math/math_verify_reward.py
   algo: gspo
   world_size: 1
   tp_size: 1
   batch_size: 1
   n_samples: 8
   ...
   =========================================

Rollout logs
~~~~~~~~~~~~

After model loading, the trainer enters the rollout phase. You should see
``rollout_start``, decode progress, and ``rollout_end``:

.. code-block:: text

   epoch=0 stage=epoch_start
   role=policy stage=rollout_start
   rollout decode progress: dp=0/1 active=8 cuda_graph=True tokens_per_second=58.3
   role=policy stage=rollout_end
   metric=reward_mean value=0.000000

The ``reward_mean`` of ``0.0`` is normal for a smoke test — the base model
rarely produces correctly formatted math answers on the first try. What
matters is that rollout completed without an exception.

Train logs
~~~~~~~~~~

After rollout, the optimizer step runs:

.. code-block:: text

   role=policy stage=train_start
   role=policy stage=train_end
   train_stats={'loss': 0.0, 'advantage_mean': 0.0, 'response_len': 745.0,
                'rollout_logprobs_mean': -0.1823, 'train_logprobs_mean': -0.2105,
                'logp_diff_mean': 0.0282, 'ratio_mean': 1.0, 'ratio_std': 0.0,
                'grad_norm': 0.0, 'lr': 1e-06,
                'step_rollout_time_s': 12.4, 'step_train_time_s': 3.1,
                'step_e2e_time_s': 15.5}

A successful smoke test reaches this ``train_stats=...`` line without raising
an exception. Even though ``loss`` may show ``0.0`` (common when normalized
advantages cancel at the initial ratio), the fact that the full loop —
rollout, reward, advantage, train — executed end to end is what you are
verifying.

See :doc:`/cli/observability` for the complete ``train_stats`` field
reference.

Step timing
~~~~~~~~~~~

The local backend also logs a compact timing line:

.. code-block:: text

   time rollout=12.413 train=3.127 total=15.540

This tells you how many seconds were spent in rollout versus training. For
the smoke test, absolute numbers do not matter — only that both phases ran.

Expected artifacts
------------------

Checkpoints
~~~~~~~~~~~

By default, the smoke test does **not** save checkpoints because
``--save-path`` is not set. If you want to verify that checkpoint writing
works, add ``--save-path``:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 1 \
     --save-path /tmp/areno-smoke-ckpt \
     --save-interval 1

After one step, ``/tmp/areno-smoke-ckpt`` should contain a model checkpoint
directory with tokenizer files, model weights, and an optimizer state.

Metrics (TensorBoard)
~~~~~~~~~~~~~~~~~~~~~

Pass ``--metrics-log-dir`` to write TensorBoard event files:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 1 \
     --metrics-log-dir /tmp/areno-smoke-tfevent

Then launch TensorBoard to inspect the metrics:

.. code-block:: bash

   tensorboard --logdir /tmp/areno-smoke-tfevent --port 6006

Open ``http://localhost:6006`` in your browser. You should see scalar charts
in three namespaces:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Namespace
     - Charts
   * - ``rollout/*``
     - ``rollout/rewards_mean``, ``rollout/rewards_std``,
       ``rollout/rewards_max``, ``rollout/rewards_min``,
       ``rollout/advantages_mean``, ``rollout/advantages_std``,
       ``rollout/logprobs_mean``, ``rollout/seq_len_mean``,
       ``rollout/response_len_mean``, ``rollout/num_sequences``.
   * - ``train/*``
     - ``train/loss``, ``train/advantage_mean``,
       ``train/ratio_mean``, ``train/ratio_std``,
       ``train/grad_norm``, ``train/lr``,
       ``train/rollout_logprobs_mean``, ``train/train_logprobs_mean``.
   * - ``time/*``
     - ``time/rollout``, ``time/reward``, ``time/advantage``,
       ``time/train``.

For the smoke test, each chart will have only one or two data points. The
goal is to confirm that the metrics pipeline (recording, writing, and
reading event files) works — not to analyze training dynamics.

See :doc:`/cli/observability` for the full metrics reference.

Common failures
---------------

The table below lists the failures most commonly encountered during a smoke
test and how to resolve them.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Cause and fix
   * - ``CUDA is not available`` or
       ``No NVIDIA GPU detected``
     - PyTorch was installed without CUDA support, or no NVIDIA driver is
       present. Reinstall PyTorch with the correct CUDA build
       (``pip install torch --index-url ...cu121``) and run ``areno check``.
   * - ``ModuleNotFoundError: No module named 'areno_accel'``
     - The CUDA extension was not compiled during installation. Ensure
       ``CUDA_HOME`` is set and ``nvcc`` is on ``PATH``, then reinstall with
       ``pip install -e . --no-build-isolation``. See
       :doc:`/troubleshooting/areno-accel`.
   * - ``ImportError: cannot import name 'flash_attn'``
     - ``flash-attn`` is not installed or is incompatible with your GPU.
       Tesla T4 and other Turing-era GPUs do not support ``flash-attn``;
       AReno automatically falls back to the ``native`` attention backend
       and prints a warning. No action is needed unless the fallback itself
       fails — see :doc:`/troubleshooting/flash-attn`.
   * - ``RuntimeError: CUDA out of memory``
     - The GPU ran out of VRAM during rollout or training. The smoke test
       is already minimal (``--batch-size 1``, ``--tp-size 1``), so OOM
       usually means another process is using the GPU. Run ``nvidia-smi``
       to check, or add ``--attn-backend native`` to reduce memory. See
       :doc:`/troubleshooting/oom-timeout`.
   * - ``FileNotFoundError: examples/math/dataset_loader.py``
     - You are not running the command from the AReno repository root.
       ``cd`` into the cloned ``AReno`` directory and retry.
   * - ``OSError: ... gsm8k:main`` or dataset download timeout
     - The remote dataset could not be fetched. By default AReno uses
       ModelScope; switch to Hugging Face with ``--model-hub hf``, or
       pre-download the dataset and pass a local path.
   * - Command hangs after printing the config
     - The model is downloading. The first run fetches ``Qwen/Qwen3-0.6B``
       from the remote hub, which can take several minutes depending on
       your network. Subsequent runs use the local cache.
   * - ``train_stats`` shows ``ratio_mean=1.0`` and
       ``grad_norm=0.0``
     - This is expected for a single-step smoke test with the base model.
       GSPO normalizes advantages within the group; when all sampled
       completions receive the same reward (e.g., all wrong), advantages
       are zero, so the ratio stays at 1.0 and the gradient vanishes. This
       is a property of the algorithm, not a bug in your setup. See
       :doc:`/concepts/training-loop` for how the RL loop produces
       non-trivial gradients over multiple steps.

What the smoke test does not verify
------------------------------------

The smoke test is a **wiring check**, not a quality benchmark. Specifically,
it does **not** tell you:

* Whether the model will learn the task with more steps.
* Whether your reward function is well-calibrated.
* Whether your hyperparameters are optimal for production training.
* Whether multi-GPU distributed training will work (the smoke test uses
  ``--world-size 1``).

Once the smoke test passes, move on to the :doc:`/cookbook/math-rlvr` recipe
for a more realistic RLVR run, or the :doc:`/getting-started/quickstart` page
for the agentic rollout path.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Check
     - Pass criterion
   * - ``areno check`` returns no ``FAIL``
     - Environment is ready.
   * - ``areno train`` reaches ``train_stats=...``
     - Full loop (rollout → reward → advantage → train) executed.
   * - No exception raised
     - CLI, dataset loader, reward function, and backend are wired correctly.
   * - (Optional) Checkpoint directory is populated
     - Checkpoint saving works when ``--save-path`` is set.
   * - (Optional) TensorBoard shows scalar charts
     - Metrics pipeline works when ``--metrics-log-dir`` is set.

If all checks pass, your AReno installation is verified and ready for real
training experiments.
