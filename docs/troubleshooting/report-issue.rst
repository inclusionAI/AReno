How to report an issue
======================

Good issue reports include enough environment, command, and task context for
another person to reproduce the failure.

Include:

* The exact command.
* The relevant stack trace or error message.
* ``areno check`` output.
* ``areno env --json`` output.
* GPU type and count.
* PyTorch, CUDA, FlashAttention, and AReno versions.
* Whether ``ARENO_BUILD_EXT=0`` was used.
* Dataset loader, reward function, and agent function paths when relevant.
* The smallest batch size or task seed that reproduces the issue.

For private data, replace examples with a minimal synthetic record that keeps
the same shape and failure.
