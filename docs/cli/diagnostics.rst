:orphan:

Diagnostics CLI reference
=========================

``areno env`` and ``areno check`` help diagnose setup problems before a user
hits low-level Python, CUDA, or PyTorch errors.

``areno env`` is a descriptive support report. It does not initialize the AReno
engine or load model weights. Use it when collecting information for an issue.

.. code-block:: bash

   areno env

For machine-readable issue reports:

.. code-block:: bash

   areno env --json

The report includes:

* AReno version
* Python version and executable
* OS, platform, and architecture
* PyTorch version, CUDA build, CUDA runtime, and CUDA availability
* CUDA driver information from ``nvidia-smi`` when available
* visible GPU count, names, and compute capability
* ``CUDA_HOME`` and inferred CUDA toolkit location
* ``nvcc`` path and version
* ``flash-attn`` import status and version
* ``flash-linear-attention`` import status and version
* ``areno_accel`` import status
* selected environment variables such as ``MAX_JOBS``,
  ``CUDA_VISIBLE_DEVICES``, and ``TORCH_CUDA_ARCH_LIST``

areno check
-----------

``areno check`` validates whether the machine is ready to run AReno training
and serving. It classifies each check as ``OK``, ``WARN``, or ``FAIL`` and
prints concrete next steps for failures.

.. code-block:: bash

   areno check

Example output:

.. code-block:: text

   AReno check: not ready

   OK   Python >= 3.10
        found 3.11.8
   OK   PyTorch CUDA build
        torch.version.cuda=12.4
   OK   CUDA_HOME
        not set (not required for runtime; areno_accel imports)

``CUDA_HOME`` and ``nvcc`` are only warnings when AReno needs to build its CUDA
extension. If the installed ``areno_accel`` extension imports successfully,
they are not required for runtime readiness.

Checks include:

* Python version
* supported platform
* PyTorch import and version
* PyTorch CUDA build
* ``torch.cuda.is_available()``
* NVIDIA GPU visibility
* ``CUDA_HOME`` and ``nvcc``
* optional runtime dependency imports
* ``areno_accel`` import
* writable cache/log locations

``WARN`` items usually indicate degraded or incomplete setup. ``FAIL`` items
mean AReno is not ready to run the CUDA training/inference engine.

areno token-report
------------------

``areno token-report`` computes token-length distributions and context capacity
for a JSONL dataset using the selected tokenizer. It reports min, p50, p90, p95,
p99, and max for prompt, response, and total token lengths, along with the
percentage of examples that exceed a configurable maximum context length.

.. code-block:: bash

   areno token-report --dataset-path ./data.jsonl --tokenizer Qwen/Qwen3-0.6B

For machine-readable output (e.g. piping to a script):

.. code-block:: bash

   areno token-report --dataset-path ./data.jsonl --tokenizer Qwen/Qwen3-0.6B --json

Options:

* ``--dataset-path`` (required): Path to a JSONL file. Each line must be a JSON
  object with at least a ``prompt`` field.
* ``--tokenizer`` (required): HuggingFace model path for tokenizer loading.
* ``--max-context`` (default: 4096): Maximum context length for over-context
  percentage calculation.
* ``--sample-ratio`` (default: 1.0): Fraction of examples to sample. Use a value
  between 0 and 1.0. ``1.0`` performs a full scan.
* ``--sample-seed`` (default: None): Random seed for deterministic sampling.
  Required for reproducible results when ``--sample-ratio`` is less than 1.0.
* ``--response-field`` (default: None): Dataset field to also measure response
  token length. When provided, ``--tokenizer`` is used to encode the response
  text.
* ``--prompt-field`` (default: "prompt"): Dataset field containing the prompt
  text.
* ``--json``: Emit machine-readable JSON instead of a human-readable table.

Output fields (JSON mode):

* ``total_samples``: Total number of records in the dataset.
* ``sampled``: Number of records actually analyzed (after sampling).
* ``sampling_seed``: The random seed used, or ``null`` for full scans.
* ``max_context``: The configured maximum context length.
* ``prompt_stats``: Object with ``count``, ``min``, ``p50``, ``p90``, ``p95``,
  ``p99``, ``max``, and ``mean`` for prompt token lengths.
* ``response_stats``: Same structure as ``prompt_stats`` for response lengths,
  or ``null`` when ``--response-field`` is not provided.
* ``total_stats``: Same structure for ``prompt + response`` total lengths, or
  ``null`` when ``--response-field`` is not provided.
* ``over_context_count``: Number of examples exceeding ``max_context``.
* ``over_context_pct``: Percentage of examples exceeding ``max_context``.
* ``retained_under_policy``: Object mapping each overlength policy to the
  number of retained examples. ``drop`` discards over-context examples;
  ``truncate`` keeps all examples (with truncation).

Limitations:

* The dataset must be a local JSONL file. HuggingFace dataset IDs are not
  supported by the CLI command directly.
* The tokenizer must be loadable by HuggingFace ``transformers``.
* The command loads the tokenizer but does not initialize the AReno engine or
  any GPU resources.

Example with deterministic sampling and response length:

.. code-block:: bash

   areno token-report \
     --dataset-path ./train.jsonl \
     --tokenizer Qwen/Qwen3-0.6B \
     --max-context 2048 \
     --sample-ratio 0.1 \
     --sample-seed 42 \
     --response-field answer \
     --json
