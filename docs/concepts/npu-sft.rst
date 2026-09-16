Ascend NPU integration
======================

The target environment is Linux/aarch64, Ascend 910, CANN 9.0.0,
PyTorch 2.10.0+cpu and torch_npu 2.10.0.post2. The CPU-tagged PyTorch
installation is retained; torch_npu provides the NPU device and operators.

Installation detects ``torch_npu`` without importing it and selects
``requirements/npu.txt``. CUDA and MLX retain their existing dependencies and
builders. The NPU extension uses the installed PyTorch C++ build tooling and
dispatches to torch_npu's compiled CANN operators. No CUDA compiler or
AReno-specific accelerator environment exports are needed.

.. code-block:: bash

   python -m pip install -e . --no-build-isolation
   python -m pip install pytest
   python -m pytest -q tests/test_npu_activation.py

Current validation boundary
---------------------------

This is **not complete NPU training/serving support**. The compiled extension
currently exposes six entries: SiLU, sigmoid and softplus, with their backward
operators. They preserve the public accel wrapper signatures and execute
through native CANN dispatch; they are not newly fused Ascend C kernels.

The backend source reuses the CUDA workflows for training, generation, losses,
optimizers, checkpoints and serving, with Ascend device initialization, HCCL
and memory probes. The backend directory contains only ``__init__.py`` and
``backend.py``. The shared TP/DP rank layout is reused. These paths have not
run on Ascend. Worker startup rejects the incomplete native extension before
starting a training or serving job.

Remaining native families include gated activations, normalization, dense and
grouped linear, attention, embedding, convolution, routing/MoE, recurrent
operators, and FP32-master/8-bit/4-bit AdamW. The existing opt-in
``tests/test_npu_end_to_end.py`` becomes the SFT/rollout/checkpoint acceptance
test once these kernels are complete; it is not expected to pass yet.

Development validation checks packaging, device dispatch and shared workflows
on CPU. NPU extension compilation and numerical execution must be validated in
the target CANN/torch_npu environment before extending the operator set.
