GaLore rank sweep and independent reproduction
==============================================

Scope and reproducibility
-------------------------

This is an incremental, standalone reference for issue #593.  It does not
integrate with or modify either mainline GaLore implementation.  The source
baseline was ``db3dbdf`` and the experiment used Python 3.12.7 with
``torch 2.7.1+cpu``; no GPU training or serving was run.

The artifacts are ``areno/optim/galore_reference.py``,
``tools/galore_sweep.py``, and ``tests/test_galore_reference_cpu.py``.
Reproduce the reported run from the repository root with::

   C:/Users/AORUS/anaconda3/python tools/galore_sweep.py
   C:/Users/AORUS/anaconda3/python -m pytest tests/test_galore_reference_cpu.py -q

The sweep uses seed 593, 40 by 40 matrices, 60 steps, ranks 4/8/16/32, and
learning rates 0.003/0.01/0.03.  It runs a separable quadratic and a small
full-batch linear regression, with identical initial values for every method.
The two runs of the default command produced byte-identical output
(SHA-256 ``7782b15933b88ab3d708eebf7e4f5168a21e5242929671b372587f5de712ec42``).
The test command completed with ``8 passed``.

Reference semantics and correctness anchor
------------------------------------------

For a wide matrix gradient ``G`` of shape ``(m, n)``, the reference computes
its first-gradient SVD and retains ``U_r``.  It stores ``R = U_r^T G`` and two
Adam moment buffers of shape ``(r, n)``.  For a tall matrix it retains ``V_r``
and stores ``R = G V_r`` with shape ``(m, r)``.  AdamW moment updates occur in
that low-rank space; the normalized update is projected back with ``U_r`` or
``V_r^T``.  Decoupled weight decay is applied to the original full parameter.
The projector is frozen after the first gradient, so this experiment isolates
rank rather than SVD refresh policy.

At ``rank == min(m, n)`` the reference deliberately uses identity coordinates,
the uncompressed control path.  It then matches single-tensor
``torch.optim.AdamW`` at every step, including nonzero weight decay, to an
absolute tolerance of ``1e-6`` in the CPU tests.  This special case is not
just an optimization: AdamW's second moment and epsilon are coordinate-wise,
and an arbitrary SVD orthogonal rotation does not generally commute with them.
Thus one must not claim that an arbitrary rotated full-rank GaLore update is
numerically identical to AdamW.

Measured final losses
---------------------

The following is the complete final-loss table at step 60.  Lower is better;
all runs use zero weight decay to isolate projection rank.

.. list-table::
   :header-rows: 1

   * - Objective
     - lr
     - AdamW
     - rank 4
     - rank 8
     - rank 16
     - rank 32
   * - quadratic
     - 0.003
     - 8.1356e-01
     - 9.5887e-01
     - 9.2913e-01
     - 8.8362e-01
     - 8.3937e-01
   * - quadratic
     - 0.01
     - 5.0727e-01
     - 8.8967e-01
     - 8.0404e-01
     - 6.7879e-01
     - 5.7160e-01
   * - quadratic
     - 0.03
     - 1.2477e-01
     - 7.6689e-01
     - 5.9385e-01
     - 3.6290e-01
     - 2.0036e-01
   * - linear
     - 0.003
     - 2.4554e+00
     - 4.4400e+00
     - 3.9809e+00
     - 3.4607e+00
     - 3.1419e+00
   * - linear
     - 0.01
     - 5.1189e-01
     - 3.6031e+00
     - 2.6862e+00
     - 1.6380e+00
     - 1.0170e+00
   * - linear
     - 0.03
     - 3.9031e-02
     - 3.2120e+00
     - 2.1464e+00
     - 8.9260e-01
     - 1.1624e-01

The script also prints sampled curves at steps 0/1/5/10/20/30/40/60.  For
example, at lr 0.03 the quadratic AdamW losses at steps 0/10/20/40/60 are
0.9943/0.7060/0.4970/0.2470/0.1248, versus rank-4 losses
0.9943/0.9366/0.8881/0.8151/0.7669.  In both objectives and at every tested
learning rate, final loss decreases as rank increases; rank 32 is closest to
AdamW.  Increasing the learning rate improves all methods in this short run,
but does not remove the low-rank gap.

Interpretation and memory
-------------------------

These synthetic gradients are generally full-rank and their useful subspace
can change as the parameters move.  A small fixed projector discards gradient
components outside its initial subspace, which explains the clear disadvantage
for ranks 4--16 under a fixed 60-step budget.  GaLore can therefore be
significantly worse than full AdamW when the gradient is high-rank or rapidly
rotating, the rank is small relative to the matrix, or comparison is made at
the same short step budget without retuning hyperparameters.  This experiment
does not establish a universal ranking for low-rank gradients, longer training,
or a memory-constrained setting.

For two FP32 moments, the theoretical persistent storage is ``8*m*n`` bytes
for AdamW and ``4*2*r*max(m,n)`` bytes for GaLore.  The ratio is
``r/min(m,n)``; at saturated rank there is no moment-storage saving.

.. list-table::
   :header-rows: 1

   * - Method
     - rank
     - moment scalars
     - KiB
     - % of AdamW
   * - AdamW
     - full
     - 3200
     - 12.500
     - 100.0
   * - GaLore
     - 4
     - 320
     - 1.250
     - 10.0
   * - GaLore
     - 8
     - 640
     - 2.500
     - 20.0
   * - GaLore
     - 16
     - 1280
     - 5.000
     - 40.0
   * - GaLore
     - 32
     - 2560
     - 10.000
     - 80.0

CPU-only limitations
--------------------

The memory table is analytical, not an allocator or peak-memory measurement;
it excludes parameters, gradients, activations, projector storage, and SVD
workspace.  CPU SVD and small-matrix runtime do not predict GPU kernel time,
SVD overhead, communication, or end-to-end throughput.  These full-batch
synthetic objectives do not represent LLM gradient spectra, stochastic
minibatches, mixed precision, distributed sharding, or checkpoint behavior.
No periodic projector refresh, production trainer integration, or actual GPU
memory saving was validated.  Those questions require separate GPU and model
experiments; this report intentionally makes no such extrapolation.
