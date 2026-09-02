Using 4-bit AdamW
=================

AReno provides an opt-in packed 4-bit AdamW optimizer for CUDA training. It
changes optimizer-state storage only; the model checkpoint and training data
format are unchanged.

Command line
------------

Add ``--adam-4bit`` to any CUDA ``areno train`` command:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --world-size 1 \
     --tp-size 1 \
     --batch-size 2 \
     --n-samples 2 \
     --mini-bs 1 \
     --adam-4bit

Keep the existing learning-rate and Adam settings unless an experiment calls
for different values:

.. code-block:: bash

   areno train \
     ... \
     --adam-4bit \
     --lr 1e-6 \
     --adam-beta1 0.9 \
     --adam-beta2 0.999

``--adam-4bit`` and ``--adam-8bit`` are mutually exclusive. Passing both
causes configuration validation to fail before training starts.

Optimizer-state offload
-----------------------

The 4-bit optimizer supports the same CUDA optimizer-state residency options
as the other CUDA AdamW implementations.

Keep state on the training device:

.. code-block:: bash

   areno train ... --adam-4bit

Offload state to CPU memory between train calls:

.. code-block:: bash

   areno train ... \
     --adam-4bit \
     --optimizer-state-offload cpu

Stream state through a local disk directory:

.. code-block:: bash

   areno train ... \
     --adam-4bit \
     --optimizer-state-offload disk \
     --optimizer-state-offload-dir /local/nvme/areno-optimizer \
     --optimizer-state-offload-batch-size 1

Use a fast local NVMe path for disk offload. Runtime mmap files are temporary
scratch files and are not restartable checkpoints.

Trainer configuration
---------------------

Set ``adam_4bit=True`` on a CLI trainer configuration:

.. code-block:: python

   from areno.api.trainer_config import PolicyTrainerConfig

   config = PolicyTrainerConfig(
       algo="gspo",
       ckpt="Qwen/Qwen3-0.6B",
       dataset_path="gsm8k:main",
       dataset_loader_fn="examples/math/dataset_loader.py",
       reward_fn_path="examples/math/math_verify_reward.py",
       backend="cuda",
       world_size=1,
       tp_size=1,
       adam_4bit=True,
   )

For the lower-level Trainer SDK, pass the optimizer option through
``CudaConfig``:

.. code-block:: python

   from areno import Trainer
   from areno.api import CUDA, CudaConfig

   trainer = Trainer(
       world_size=1,
       model_path="Qwen/Qwen3-0.6B",
       backend_type=CUDA,
       custom_config=CudaConfig(
           tp_size=1,
           optimizer={
               "adam_4bit": True,
               "lr": 1e-6,
               "betas": (0.9, 0.999),
               "weight_decay": 0.01,
           },
       ),
   )

The equivalent engine-level setting is
``OptimizerConfig(adam_4bit=True)``.

Requirements and errors
-----------------------

* Use the CUDA backend. MLX configurations reject ``adam_4bit=True``.
* Do not enable ``adam_8bit`` at the same time.
* Rebuild or reinstall AReno after switching to a revision that adds the
  fused 4-bit optimizer kernel.
* A saved 4-bit optimizer state must be resumed with the 4-bit optimizer. The
  separately saved model weights remain usable without ``--adam-4bit``.

Confirm that the option is available with:

.. code-block:: bash

   areno train --help | grep adam-4bit
