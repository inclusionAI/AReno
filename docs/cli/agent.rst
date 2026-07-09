:orphan:

Operations agent CLI
====================

``areno agent``

Run a local operations assistant for AReno training and serving tasks. The
agent uses an OpenAI-compatible chat model, inspects the current checkout, reads
command help, runs diagnostics, and can produce or execute AReno commands for
the current machine.

Configure the agent
-------------------

Store the endpoint, model, and API key once:

.. code-block:: bash

   areno agent --set \
     --base-url http://127.0.0.1:8000/v1 \
     --model deepseek-v4-flash \
     --api-key "$OPENAI_API_KEY"

The config is stored under ``~/.areno``. After this, normal agent runs do not
need ``--base-url``, ``--model``, or ``--api-key`` on the command line.

Run an agent task
-----------------

Pass the requested job as one natural-language argument:

.. code-block:: bash

   areno agent "Give me a complete command to run the math demo with n-samples=8, fitting the current GPU and using as much GPU memory as practical."

The agent can inspect GPUs, read example files, run ``areno check`` and
``areno train --help``, ask follow-up questions through the terminal when a
required value is missing, and stream command output while it works.

From a source checkout, use the repository-local wrapper when AReno is not
installed:

.. code-block:: bash

   ./agent.sh "Give me a complete command to run the math demo with n-samples=8, fitting the current GPU and using as much GPU memory as practical."

Refresh agent knowledge
-----------------------

The built-in operations knowledge tells the model how to reason about AReno
train and serve commands, GPU memory, smoke checks, ModelScope defaults, and
common recovery steps. Refresh the local copy when CLI behavior or examples
change:

.. code-block:: bash

   areno agent --refresh-knowledge

When to use it
--------------

Use ``areno agent`` when you want help choosing runnable train or serve
parameters for the current machine, especially when GPU memory, tensor
parallelism, dataset loaders, reward functions, or agentic rollout settings are
unclear. For deterministic scripts and CI, prefer explicit ``areno train`` or
``areno serve`` commands after the agent has helped you settle on parameters.
