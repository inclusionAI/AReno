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

Tokenizer alignment inspector
-----------------------------

``areno check --tokenizer-inspect`` renders token IDs, token pieces,
special-token markers, EOS placement, role labels, and loss-mask spans
side by side — without modifying the actual tokenizer path.

Plain text inspection:

.. code-block:: bash

   areno check --tokenizer-inspect /path/to/model --inspect-text "hello world"

Chat message inspection (pass messages as JSON):

.. code-block:: bash

   areno check --tokenizer-inspect /path/to/model --inspect-chat \
     --inspect-text '[{"role":"user","content":"hello"}]'

Tool-call inspection (pass messages and tools as JSON):

.. code-block:: bash

   areno check --tokenizer-inspect /path/to/model \
     --inspect-text '[{"role":"user","content":"What time is it?"}]' \
     --inspect-tools '[{"type":"function","function":{"name":"get_time"}}]'

Use ``--inspect-json`` for machine-readable output:

.. code-block:: bash

   areno check --tokenizer-inspect /path/to/model --inspect-text "hello" --inspect-json

Output fields per token:

- **index**: Position in the token sequence (0-based)
- **token_id**: Integer token ID from the tokenizer
- **token_piece**: Decoded text for this single token
- **is_special**: Whether this token is a special token
- **is_eos**: Whether this token is an EOS token
- **is_unknown**: Whether this token is the unknown token
- **role**: ``system``, ``user``, ``assistant``, ``tool``, ``prompt``, or ``generation_prompt``
- **in_loss**: Whether this token participates in policy loss

Report-level fields:

- **round_trip_lossless**: Whether ``encode(decode(token_ids)) == token_ids``
- **warnings**: Truncation, round-trip loss, unknown tokens
