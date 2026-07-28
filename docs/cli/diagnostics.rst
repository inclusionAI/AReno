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

Preflight model references
--------------------------

``areno check --model-ref`` validates a model checkpoint directory or remote
hub reference *before* expensive model or worker initialisation. It checks
local directory readability, ``config.json`` parseability, tokenizer file
presence, and safetensors shard integrity — all without loading weights or
tokenizers.

.. code-block:: bash

   areno check --model-ref /path/to/model --model-hub modelscope

For remote references (repo IDs), it verifies the hub client
(``modelscope`` or ``huggingface_hub``) is installed and reports local cache
status without making network requests.

Use ``--json`` for machine-readable output:

.. code-block:: bash

   areno check --model-ref /path/to/model --json

Output fields:

* **status**: ``ok``, ``not_found``, ``permission``, ``network``, or ``format``
* **stage**: where the check stopped (``local``, ``config``, ``tokenizer``,
  ``weights``, ``remote``)
* **missing_artifacts**: exact list of missing files
* **next_step**: actionable suggestion to resolve the failure

Example failure output:

.. code-block:: text

   AReno check: not ready

   FAIL model preflight (tokenizer)  /path/to/model
        no valid tokenizer file set found | missing: tokenizer.json, ...
   Next:
     Download tokenizer files into /path/to/model

``areno train --preflight`` and ``areno serve --preflight`` run the same
validation automatically before starting. The flag is disabled by default to
preserve backward compatibility.
