:orphan:

CUDA and PyTorch issues
=======================

AReno training and serving require a CUDA-capable NVIDIA GPU and a compatible
PyTorch install. CPU-only environments can run docs and lightweight tests, but
they cannot run the training or serving engine.

Check GPU visibility:

.. code-block:: bash

   python -c "import torch; print('GPU:', torch.cuda.is_available())"

If the result is ``False``:

* Confirm the machine has an NVIDIA GPU.
* Confirm the driver and CUDA runtime match the installed PyTorch wheel.
* Confirm the Python environment is the one used to install AReno.
* Run ``areno env --json`` and attach it to issue reports.

If CUDA is visible but training fails, move next to
:doc:`oom-timeout` or :doc:`areno-accel` depending on the error.
