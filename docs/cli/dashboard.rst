Dashboard launcher preview
==========================

Start the local dashboard with ``areno dashboard --start`` and open the
Launcher page. ``Preview command`` validates the form without initializing a
model, starting workers, or writing artifacts. ``Start`` runs the same preview
and only submits when it succeeds.

Validation covers required model and dataset references, local path existence,
positive numeric fields, tensor-parallel divisibility, explicit GPU-list
counts, option conflicts, ports, and malformed ``extra_args`` quoting. Remote
ModelScope and Hugging Face references produce warnings because preview does
not contact either service. Warnings must be acknowledged in the launcher
before submission.

The preview response contains:

* ``ok`` and ``requires_acknowledgement`` status flags
* field-scoped ``errors`` and ``warnings``
* the tokenized ``command`` used for submission
* a shell-safe ``shell_command`` rendered with Python ``shlex`` rules
* ``resolved_args``, the structured launcher inputs

Local API example
-----------------

This fixture exercises the successful path without a database, network
service, model weights, or GPU initialization:

.. code-block:: bash

   mkdir -p /tmp/areno-preview/model
   printf '{"prompt":"2 + 2"}\n' > /tmp/areno-preview/data.jsonl
   curl -s http://127.0.0.1:8765/api/launcher/preview \
     -H 'Content-Type: application/json' \
     -d '{
       "kind": "train",
       "config": {
         "algo": "sft",
         "ckpt": "/tmp/areno-preview/model",
         "dataset_path": "/tmp/areno-preview/data.jsonl",
         "dataset_loader_fn": "examples/sft/alpaca/dataset_loader.py",
         "world_size": 1,
         "tp_size": 1,
         "epochs": 1,
         "batch_size": 1,
         "mini_bs": 1,
         "score_micro_bs": 1,
         "max_prompt_tokens": 32,
         "max_new_tokens": 16,
         "save_path": "/tmp/areno-preview/output"
       }
     }'

For a deterministic invalid boundary, change ``tp_size`` to ``2`` while
leaving ``world_size`` at ``1``. The response contains a ``tp_size`` error and
an empty ``shell_command``. Preview never checks remote repository contents or
available device memory; those remain runtime checks.
