CLI Reference
=============

Use these pages when you need exact command options, dataset loader shapes, or
runtime observability fields.

Command pages:

* :doc:`/cli/training`
* :doc:`/cli/inference`
* :doc:`/cli/agent`
* :doc:`/cli/dataset_loaders`
* :doc:`/cli/observability`
* :doc:`/cli/diagnostics`

Sequence-parallel precedence
----------------------------

For ``areno train``, ``--sequence-parallel`` and
``--no-sequence-parallel`` explicitly override the checkpoint model
configuration. If neither flag is present, AReno uses the checkpoint's
``sequence_parallel`` value. TP1 execution never activates sequence
parallelism. See :doc:`/cli/training` for the complete training option list.
