Integrate a harness with Harbor
===============================

A harness owns the agent loop: it sends model requests, executes tools and
decides when the task is finished. This guide connects that loop to AReno
through Harbor. You will select or implement a Harbor agent, register a
harness configuration and expose the three hooks used by ``areno train``.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Task adaptation <harbor/tasks>
   Pi SVG walkthrough <harbor/pi>

Prerequisites
-------------

Use Python 3.12+ and Harbor 0.24.0. This guide describes the integration in
`AReno PR #621 <https://github.com/inclusionAI/AReno/pull/621>`_; running it
requires a checkout containing ``areno.api.harness`` and the SVG version of
``examples/agentic/pi/``. Prepare AReno using
:doc:`/getting-started/installation`; reuse
the PyTorch and attention packages matched to your accelerator environment.
AReno's dependencies include Harbor on supported Python versions. Check the
installation from the repository root:

.. code-block:: bash

   python -c 'from importlib.metadata import version; print(version("harbor"))'
   python -c 'from areno.api import HarnessRunner; print(HarnessRunner.__name__)'
   areno check

The harness must support an OpenAI-compatible **Chat Completions** endpoint,
including its tool-call format. The sample proxy currently serves
``POST /v1/chat/completions`` with ``n=1``; streaming is supported. A harness
that only speaks Responses or Anthropic Messages needs a protocol adapter.

1. Select an existing agent
---------------------------

Start with a native Harbor agent when one already wraps your harness. For
example, Pi can be composed with the SVG task adapter:

.. code-block:: python

   import os

   from areno.api import HarnessRunner
   from examples.agentic.pi import svg_dataset  # Registers "svg_icon".

   workflow = HarnessRunner(
       harness="pi",
       dataset="svg_icon",
       environment="docker",
       harness_options={
           "kwargs": {"model_api": "openai-completions", "thinking": "off"},
       },
       environment_options={"image": os.environ["ARENO_HARBOR_IMAGE"]},
       require_agent_success=False,
   )

``harness_options`` contains native Harbor ``AgentConfig`` fields, including
``kwargs``, ``env``, ``skills`` and ``mcp_servers``. Set agent-specific options
inside ``kwargs``. ``list_harnesses()`` from ``areno.api`` lists native names
and locally registered aliases.

Agents and tools are preinstalled by default. ``preinstalled=True`` skips
the native installed agent's package installation while retaining its
execution, session handling and trajectory conversion. Use
``preinstalled=False`` if Harbor should install the agent in its environment.
Custom external agents still receive their ``setup`` hook.

2. Wrap a new harness
---------------------

If Harbor does not already support the harness, implement its ``BaseAgent``
contract. Keep the existing agent loop and tools; adapt its client to the
sample's model connection. For a CLI installed in a container, Harbor's
``BaseInstalledAgent`` additionally supplies command execution and log
collection. See `Harbor's agent guide
<https://www.harborframework.com/docs/agents>`_ for that interface.

Here is a small external shell harness illustrating the complete connection.
Save it as ``my_harness.py`` at the repository root. Its tool runs through
Harbor's environment, so the same loop can operate on different tasks.

.. code-block:: python

   import json

   from harbor.agents.base import BaseAgent
   from harbor.agents.model_connection import ModelConnectionSpec
   from harbor.agents.options import AgentOptions
   from openai import AsyncOpenAI
   from pydantic import Field


   class ShellOptions(AgentOptions):
       max_turns: int = Field(default=8, ge=1)


   class ShellAgent(BaseAgent):
       options_model = ShellOptions
       MODEL_CONNECTION = ModelConnectionSpec(default_provider="openai")

       @staticmethod
       def name():
           return "my-shell"

       def version(self):
           return "0.1.0"

       async def setup(self, environment):
           pass  # The environment already supplies the shell and task tools.

       async def run(self, instruction, environment, context):
           connection = self.model_connection
           messages = [{"role": "user", "content": instruction}]
           tools = [{
               "type": "function",
               "function": {
                   "name": "shell",
                   "description": "Run a shell command in the task workspace.",
                   "parameters": {
                       "type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"],
                       "additionalProperties": False,
                   },
               },
           }]
           async with AsyncOpenAI(
               base_url=connection.base_url,
               api_key=connection.api_key,
               max_retries=0,
           ) as client:
               for _ in range(self.options.max_turns):
                   response = await client.chat.completions.create(
                       model=self.model_name.removeprefix("openai/"),
                       messages=messages,
                       tools=tools,
                   )
                   message = response.choices[0].message
                   messages.append(message.model_dump(exclude_none=True))
                   if not message.tool_calls:
                       break
                   for call in message.tool_calls:
                       try:
                           command = json.loads(call.function.arguments)["command"]
                           if call.function.name != "shell" or not isinstance(command, str):
                               raise ValueError("expected a shell command string")
                       except (ValueError, KeyError, TypeError) as error:
                           output = f"Invalid tool arguments: {error}"
                       else:
                           result = await environment.exec(command, timeout_sec=30)
                           output = json.dumps({
                               "returncode": result.return_code,
                               "stdout": result.stdout,
                               "stderr": result.stderr,
                           })
                       messages.append({
                           "role": "tool", "tool_call_id": call.id, "content": output,
                       })
           (self.logs_dir / "messages.json").write_text(json.dumps(messages))

``run`` receives the task instruction, a Harbor environment and an
``AgentContext`` for execution metadata. ``MODEL_CONNECTION`` makes the
runner-injected ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` available through
``self.model_connection``. Every policy request must reach the sample proxy
to become training data. Create the model client per attempt and close it
before returning.

Preserve assistant tool calls and their corresponding tool observations in
order. Do not repair malformed model output by fabricating calls. The example
reports argument errors to the model and lets it try again. Its message log
is useful for inspection; ATIF export requires an agent-specific converter,
such as the one Harbor's native Pi agent supplies.

3. Register the harness and choose an environment
-------------------------------------------------

Save the configuration in an importable module, ``my_integration.py``:

.. code-block:: python

   import os

   from harbor.models.trial.config import AgentConfig

   from areno.api import HarnessRunner, register_harness
   from task_adapter import FileTasks  # Registers "file-tasks"; see the task guide.


   def shell_factory(attempt, **options):
       return AgentConfig(import_path="my_harness:ShellAgent", **options)


   register_harness("my-shell", shell_factory, replace=True)

   workflow = HarnessRunner(
       harness="my-shell",
       dataset="file-tasks",
       environment="docker",
       environment_options={"image": os.environ["ARENO_HARBOR_IMAGE"]},
       result_dir="runs/my-harness/results",
       require_agent_success=False,
   )

A harness factory receives ``HarnessAttempt`` and ``harness_options`` and
returns native ``AgentConfig``. The attempt identifies the sample and its
record; its environment is created later. You can also pass
``harness="my_harness:ShellAgent"`` directly without registering an alias.
``replace=True`` allows CLI hook files that are loaded independently to
import the same registration safely.

The image must contain the task tools and use the task workspace as its
working directory. Native environment options are supplied through
``environment_options``. ``image`` sets the task's ``docker_image``;
``task_env`` sets task environment variables; provider constructor options go
inside ``kwargs``. Set ``environment="local"`` for the external shell example
to use per-sample host workspaces instead of containers, and omit ``image``.

For a reusable environment preset, use
``register_environment(name, factory)``. Its factory receives the attempt and
options and returns ``EnvironmentSpec``. Reuse
``native_environment_spec`` from ``areno.agent.harness_trial`` for native
Harbor providers, or pass a provider import path directly as ``environment``.
Keep environment selection independent of the task adapter.

For example, add this Docker alias to ``my_integration.py`` and select it
without modifying the harness or dataset:

.. code-block:: python

   from areno.agent.harness_trial import native_environment_spec
   from areno.api import register_environment


   def docker_factory(attempt, **options):
       return native_environment_spec("docker", **options)


   register_environment("team-docker", docker_factory, replace=True)
   workflow.environment = "team-docker"

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Provider
     - Requirements and routing
   * - Docker
     - Reachable daemon and preinstalled agent/tools. The proxy binds to
       ``0.0.0.0`` and the agent connects through ``host.docker.internal`` by
       default. Override ``proxy_bind_host`` and ``proxy_connect_host`` when
       the daemon uses another network route.
   * - Local
     - Preinstalled host tools. Uses loopback and separate workspaces, but
       shares the host filesystem. AReno's local provider reaps the process
       groups it starts; it does not provide OS isolation.

Native installed agents with fixed configuration paths require an isolated
provider. The Pi example supplies a compatible local wrapper with private
home/config directories; use its ``create_runner`` rather than a generic
``HarnessRunner(harness="pi", environment="local", ...)``.

4. Expose AReno's training hooks
--------------------------------

Create ``hooks.py`` and use the same workflow for loading, rollout and reward:

.. code-block:: python

   from my_integration import workflow


   def load_training_dataset(dataset_path, *, default_loader, **kwargs):
       return workflow.dataset_loader(
           dataset_path, default_loader=default_loader, **kwargs,
       )


   async def run_agent(ctx, batch):
       return await workflow(ctx, batch)


   def reward_fn(record):
       return workflow.reward_fn(record)

Wire the module into the CLI with ``--dataset-loader-fn hooks.py``,
``--agent-fn hooks.py`` and ``--reward-fn-path hooks.py``. The dataset adapter
in :doc:`harbor/tasks` completes this example; :doc:`harbor/pi` provides the
end-to-end training command and checks.

How the connection works
------------------------

For each sampled task, ``HarnessRunner`` creates an independent record,
environment configuration and authenticated model proxy, then starts a native
Harbor ``Trial``. The task adapter prepares the workspace on ``AGENT_START``
and grades it on ``AGENT_END``. Harbor's built-in verifier is disabled;
AReno's reward hook reads the task adapter's scalar result.

.. code-block:: text

   loader -> task records -> HarnessRunner -> Harbor Trial -> agent -> tools
                                  ^                            |
                                  +-- sample model proxy <-----+
                                      -> AReno policy
   task adapter -> reward -> AReno optimizer -> checkpoint

The proxy records AReno's exact input/output token IDs and rollout log
probabilities. Training turns come from this trace, not from re-tokenizing
logs or importing an ATIF file. AReno owns context and generation limits;
harness ``max_tokens`` settings do not override the training configuration.
See :doc:`/reference/agentic-rollout-api` for the underlying contract.

Before scaling up, verify one complete tool interaction, successful and failed
task grades, two concurrent attempts with distinct workspaces, and cleanup
after a timeout. Inspect a saved trace to confirm that policy requests passed
through AReno. Then check two optimizer updates as described in :doc:`harbor/pi`.
