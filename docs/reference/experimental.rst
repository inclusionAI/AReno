Experimental and auxiliary APIs
===============================

Experimental or auxiliary APIs should stay out of the new-user path until
their contracts are stable.

When documenting an experimental surface:

* Say which AReno version or branch it applies to.
* Link to the owning module or example.
* Mark the expected stability level.
* Keep migration notes close to the page that users will find by search.

Stable public surfaces should graduate into the relevant Reference page once
the contract is ready.

World-model VLA post-training
-----------------------------

Introduced after AReno v0.0.8, this experimental, restartable workflow covers
the frozen-Wan LIBERO-Spatial experiment. Its stability level is
**experimental**: configuration and state formats can change before this moves
into the stable CLI.

This workflow covers the frozen-world-model result documented in RLinf's
`Wan example <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/wan.html>`_.
It does not implement world-model/policy co-evolution from the
`PACE paper <https://arxiv.org/abs/2602.13977>`_.

The first backend deliberately separates orchestration from model execution:

* AReno validates assets, plans checkpoint stages, submits Slurm jobs, manages
  Ray, records state and metrics, resumes checkpoints, and launches the final
  real-environment evaluation.
* A compatible RLinf checkout runs the three colocated GPU roles: frozen Wan
  world-model environment, OpenVLA-OFT rollout policy, and FSDP actor.
* The policy is optimized with the official GRPO example. The world model is
  frozen. This targets the documented 77.5% LIBERO-Spatial result; it is not
  the paper's full world-model/policy co-evolution method.

The implementation is owned by
``areno.experimental.world_model_vla`` and does not modify ``TrainerConfig`` or
the stable ``areno`` command.

Inputs and compatibility
~~~~~~~~~~~~~~~~~~~~~~~~

The workflow requires:

* a source checkout containing RLinf's
  ``wan_libero_spatial_grpo_openvlaoft`` example;
* the Python and Ray executables from an environment with RLinf's Wan and
  OpenVLA-OFT dependencies;
* local Wan and OpenVLA-OFT snapshots, each with a
  ``snapshot_manifest.json`` file containing file paths, sizes, and optional
  SHA-256 values;
* a Slurm partition capable of allocating the configured number of colocated
  GPUs; the reference experiment uses eight.

GPU memory must cover the actor's optimizer state after the first update as
well as the colocated rollout and world-model workers. Validate at least two
steps on the target accelerator before scheduling a full run; 48 GiB devices
may be insufficient even with ``actor_micro_batch_size`` set to one.

The default path uses RLinf's published example without requiring source
patches. A low-memory compatibility patchset was needed for the tested modern
Transformers environment: configurable FSDP ``sync_module_states``, explicit
OpenVLA SDPA support and selection, and sequential embodied-worker
initialization. These changes are not part of a released RLinf version at the
time of writing. Set ``require_rlinf_compatibility_patchset`` only with a
checkout containing them; preflight then verifies all four capabilities before
submission. The backend adds the RLinf checkout and a ``wan/`` source directory
adjacent to the selected virtual environment to ``PYTHONPATH`` when present.

The integration invokes RLinf through a subprocess and does not add it as an
AReno runtime dependency. All caches are redirected below ``cache_root``.
Large model snapshots, checkpoints, logs, and caches should therefore point to
shared storage rather than a small home filesystem. Download new assets through
ModelScope when the required repositories are available there; this workflow
never falls back to another model hub implicitly.

Configuration
~~~~~~~~~~~~~

Start from ``examples/vla/wan_libero_spatial_world_model.json``. Relative paths
are resolved against the JSON file, not the shell's working directory. Update
the paths and Slurm fields when the checkouts, assets, or cluster layout differ.
``partition`` is required; ``train_qos``, ``eval_qos``, ``constraint``, and
``account`` are optional because their availability is cluster-specific.

Create manifests after downloading snapshots. SHA-256 is the default; use
``--sizes-only`` only when integrity is established by another mechanism:

.. code-block:: bash

   areno-world-model-vla manifest --snapshot "$MODEL_ROOT/RLinf-Wan-LIBERO-Spatial"
   areno-world-model-vla manifest --snapshot "$MODEL_ROOT/Openvla-oft-SFT-libero-spatial-traj1"

The default stage boundaries are ``2, 20, 40, 60, 80, 100``. A successful
stage saves the complete FSDP actor and optimizer state, then submits only its
immediate ``afterok`` successor. This avoids exceeding clusters that limit the
number of queued long-running jobs. The first two-step stage is the minimum GPU
validation gate; do not treat a one-step run as proof that peak memory is safe.

Use the separate experimental CLI. When running directly from a source
checkout, the equivalent module entry point is
``python -m areno.experimental.world_model_vla``:

.. code-block:: bash

   CONFIG=examples/vla/wan_libero_spatial_world_model.json

   areno-world-model-vla plan --config "$CONFIG"
   areno-world-model-vla verify --config "$CONFIG" --sha256
   areno-world-model-vla submit --config "$CONFIG" --dry-run
   areno-world-model-vla submit --config "$CONFIG"
   areno-world-model-vla status --config "$CONFIG"

``submit`` persists a normalized copy as ``workflow_config.json`` below the
output directory. Later jobs use that copy, so moving or editing the original
file cannot change a running chain. Reusing an output directory with a
different configuration is rejected.

Recovery and evaluation
~~~~~~~~~~~~~~~~~~~~~~~

Each stage is complete only when its
``actor/model_state_dict/full_weights.pt`` exists. To recover after a failed or
timed-out job, inspect status and submit the first stage without a complete
checkpoint:

.. code-block:: bash

   areno-world-model-vla status --config "$CONFIG"
   areno-world-model-vla resume --config "$CONFIG"

Recorded jobs are not resubmitted by default. Use ``--force`` only after
confirming that the prior Slurm job is no longer running or queued.

After step 100, the chain evaluates the latest full actor weights over 496
fixed-reset LIBERO-Spatial trajectories in the real simulator. Training
``env/success_once`` measures imagined Wan rollouts; only the final evaluation
``success_once`` can be compared with the 77.5% target. Workflow status and the
raw RLinf ``metrics.log`` retain both values.
