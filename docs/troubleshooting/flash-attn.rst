:orphan:

FlashAttention issues
=====================

FlashAttention is optional unless you select an attention backend that needs
it. If FlashAttention is unavailable or unsupported on the local GPU, switch
to native attention while debugging:

.. code-block:: bash

   areno train \
     --attn-backend native \
     ...

Common checks:

* Confirm the installed FlashAttention package matches PyTorch and CUDA.
* Confirm the GPU architecture is supported by the package.
* Try the native attention backend before changing algorithm settings.
* Keep the failing command and ``areno env --json`` output for issue reports.
