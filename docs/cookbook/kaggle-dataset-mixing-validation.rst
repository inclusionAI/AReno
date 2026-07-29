Kaggle dataset-mixing validation
================================

This page will document the reproducible Kaggle GPU validation for
deterministic weighted SFT dataset mixing introduced by Issue #198. It is
intended to capture the environment, exact command, observable artifacts, and
acceptance evidence without embedding dataset samples or credentials.

.. note::

   The validation has been run successfully. The detailed report below is a
   placeholder and will be finalized after the test-report scope and
   presentation are agreed.

Environment
-----------

``[TODO: Record the Kaggle accelerator, GPU count, CUDA version, PyTorch
version, Python version, and AReno commit.]``

Setup
-----

``[TODO: Add the minimal repository checkout, dependency installation, CUDA
extension build, and environment-check commands.]``

Test configuration
------------------

``[TODO: Record the model checkpoint, named dataset sources, normalized sample
weights, seed, exhaustion policy, samples-per-epoch budget, token limits,
batch sizes, and step count.]``

Validation procedure
--------------------

``[TODO: Describe the deterministic replay check, schedule-plan inspection,
training-progress inspection, and checkpoint or artifact checks.]``

Results
-------

``[TODO: Add the schedule and mix-spec hashes, planned and observed source
proportions, scheduled/filtered/trained row counts, target-token counts,
training-loss summary, step timing, and completion status.]``

Artifacts
---------

``[TODO: List the sample-free dataset-mix plan, metrics directory, relevant
log excerpts, and checkpoint location when checkpoint saving is enabled.]``

Known limitations
-----------------

``[TODO: Document that weights are sample-based, post-tokenization filtering
can change effective row proportions, token contribution can differ by source,
and exact mid-epoch resume is outside the current contract.]``
