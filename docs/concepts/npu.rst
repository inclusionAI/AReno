Ascend NPU
==========

AReno provides an Ascend NPU backend for SFT, DPO, GRPO, GSPO and PPO,
as well as rollout, scoring and OpenAI-compatible serving.

Installation
------------

Use Linux/aarch64 with Ascend hardware and a CANN-compatible installation of
PyTorch and ``torch_npu``. The target stack is CANN 9.0.0, PyTorch 2.10.0+cpu
and ``torch_npu`` 2.10.0.post2. Source the CANN toolkit's ``set_env.sh`` before
building, and make sure CMake and the CANN development tools are available.

From the AReno repository root:

.. code-block:: bash

   python -m pip install -e . --no-build-isolation

The installer detects ``torch_npu`` and selects the NPU dependencies.
Keep the PyTorch and ``torch_npu`` versions supplied with your CANN environment.

Training
--------

Use the standard ``areno train`` command with ``--backend npu``. Select the
training method with ``--algo sft``, ``dpo``, ``grpo``, ``gspo`` or ``ppo``
and use the corresponding dataset and reward configuration.

Set ``--world-size`` to the number of devices and ``--tp-size`` to the tensor
parallel size. The CLI also selects NPU automatically on Linux when
``torch_npu`` is installed.

See :doc:`../reference/cli` for training options and
:doc:`../getting-started/quickstart` for the standard training workflow.

Serving
-------

Start an OpenAI-compatible server with a local checkpoint:

.. code-block:: bash

   areno serve --backend npu --model-path /path/to/local/checkpoint \
     --world-size 1 --tp-size 1 --port 8000

Use ``http://localhost:8000/v1`` as the base URL for your OpenAI-compatible
client. Add ``--attn-backend native`` to select native NPU attention for
training or serving.
