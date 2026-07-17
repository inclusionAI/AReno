Example Gallery
===============

Use these examples to choose a runnable path before you move to larger
experiments. Training and serving examples require a CUDA-capable NVIDIA GPU;
CPU-only and macOS machines are suitable for reading the examples, checking
metadata, and running lightweight CPU tests.

.. list-table::
   :header-rows: 1
   :widths: 24 18 38 20

   * - Example
     - Path
     - What it shows
     - Good first step for
   * - :doc:`Quickstart <quickstart>`
     - Training smoke test
     - A small local workflow to verify installation and runtime setup.
     - New users
   * - :doc:`Math RLVR <../cookbook/math-rlvr>`
     - RLVR
     - Reward-based training on a structured reasoning task.
     - RL and math training
   * - :doc:`TicTacToe Agentic RL <../cookbook/tictactoe-agentic-rl>`
     - Agentic RL
     - Trajectory collection through an agent function and reward design.
     - Agent beginners
   * - :doc:`DuelGrid Visual Agent <../cookbook/duelgrid-visual-agent>`
     - Visual agentic RL
     - Before and after behavior for a multi-action browser-game demo.
     - Advanced agent workflows

For environment setup, start with :doc:`Installation <installation>`. For data
preparation, see :doc:`Dataset Formats <../concepts/dataset-formats>`. For CUDA,
PyTorch, extension, or runtime issues, see
:doc:`Troubleshooting <../troubleshooting/index>`.
