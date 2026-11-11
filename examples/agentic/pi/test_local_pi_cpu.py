"""Local Pi keeps native Harbor execution with private concurrent session paths."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("harbor", reason="Harbor integration requires Python 3.12+ and Harbor 0.24")

from harbor.agents.installed.pi import Pi
from harbor.models.agent.context import AgentContext

from areno.agent.harness_env import local_environment, local_spec
from areno.agent.harness_trial import native_environment_spec
from areno.api.agentic import AgentBatch
from areno.api.harness import HarnessAttempt
from examples.agentic.pi.local_pi import LocalPi
from examples.agentic.pi.run_agent import create_runner


def test_local_runner_uses_native_pi_adapter_and_filters_controller_environment(monkeypatch):
    monkeypatch.setenv("ARENO_HARBOR_ENVIRONMENT", "local")
    workflow = create_runner()
    assert workflow.harness == "examples.agentic.pi.local_pi:LocalPi"
    assert workflow.environment == "local"
    assert workflow.harness_options["kwargs"]["thinking"] == "off"
    assert "PATH" in workflow.environment_options["inherit_env"]
    assert "ARENO_VLM_API_KEY" not in workflow.environment_options["inherit_env"]


@pytest.mark.parametrize("selected", ["docker", "local", " Docker ", " LOCAL "])
def test_environment_variable_switches_providers_with_a_docker_image_present(tmp_path, monkeypatch, selected):
    monkeypatch.setenv("ARENO_HARBOR_ENVIRONMENT", selected)
    monkeypatch.setenv("ARENO_HARBOR_IMAGE", "preinstalled-pi")
    monkeypatch.delenv("ARENO_HARBOR_ENVIRONMENT_OPTIONS", raising=False)
    workflow = create_runner()
    item = next(AgentBatch([{}], ["draw"], [[0]], 1).iter_samples())
    attempt = HarnessAttempt(item, tmp_path, 30, 10)
    if selected.strip().lower() == "docker":
        spec = native_environment_spec(workflow.environment, **workflow.environment_options)
        assert workflow.harness == "pi" and spec.config.type.value == "docker"
        assert spec.task_options["docker_image"] == "preinstalled-pi" and spec.isolated
    else:
        spec = local_spec(attempt, **workflow.environment_options)
        assert workflow.harness == "examples.agentic.pi.local_pi:LocalPi"
        assert "image" not in workflow.environment_options
        assert spec.workspace == str(tmp_path / "workspace") and not spec.isolated


def test_explicit_environment_overrides_environment_variable(monkeypatch):
    monkeypatch.setenv("ARENO_HARBOR_ENVIRONMENT", "local")
    monkeypatch.setenv("ARENO_HARBOR_IMAGE", "preinstalled-pi")
    workflow = create_runner(environment="docker")
    assert workflow.environment == "docker" and workflow.harness == "pi"
    assert workflow.environment_options["image"] == "preinstalled-pi"


def test_custom_environment_import_path_keeps_case(monkeypatch):
    monkeypatch.setenv("ARENO_HARBOR_ENVIRONMENT", "custom_provider:CustomEnvironment")
    workflow = create_runner()
    spec = native_environment_spec(workflow.environment, **workflow.environment_options)
    assert spec.config.import_path == "custom_provider:CustomEnvironment"


def test_native_local_pi_has_independent_config_home_and_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENO_VLM_API_KEY", "controller-only-secret")
    tools = tmp_path / "bin"
    tools.mkdir()
    script = tools / "pi"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os\nfrom pathlib import Path\n"
        "config = Path(os.environ['PI_CODING_AGENT_DIR'])\n"
        "Path('observed.json').write_text(json.dumps({\n"
        " 'config': str(config), 'models': json.loads((config/'models.json').read_text()),\n"
        " 'home': os.environ['HOME'], 'cuda': os.environ['CUDA_VISIBLE_DEVICES'],\n"
        " 'judge_key_inherited': 'ARENO_VLM_API_KEY' in os.environ,\n"
        " 'policy_key': os.environ['OPENAI_API_KEY'],\n"
        " 'turn_extension': (config/'max-turns.ts').exists()}))\n"
        "print(json.dumps({'type': 'agent_end'}))\n"
    )
    script.chmod(0o755)

    async def run(index):
        root = tmp_path / "paths with spaces" / str(index)
        root.mkdir(parents=True)
        item = next(AgentBatch([{}], ["draw"], [[0]], 1).iter_samples())
        attempt = HarnessAttempt(item, root, 30, 10)
        handle = local_environment(attempt)
        environment = handle.environment
        environment.workspace = handle.workspace
        environment.inherit_env = ["PATH"]
        environment.task_env["PATH"] = str(tools) + os.pathsep + os.environ["PATH"]
        await environment.start()
        logs = root / "logs"
        logs.mkdir()
        agent = LocalPi(
            logs_dir=logs,
            environment_logs_dir=logs,
            model_name="openai/policy",
            extra_env={"OPENAI_BASE_URL": f"http://policy-{index}.invalid/v1", "OPENAI_API_KEY": f"key-{index}"},
            thinking="off",
            model_api="openai-completions",
            max_turns=8,
        )
        assert isinstance(agent.delegate, Pi)
        try:
            await agent.setup(environment)
            await agent.run("Draw an SVG icon", environment, AgentContext())
        finally:
            await environment.stop()
        observed = json.loads((Path(handle.workspace) / "observed.json").read_text())
        config = Path(observed["config"])
        assert config.is_relative_to(root) and (config / "models.json").stat().st_mode & 0o777 == 0o600
        assert observed["models"]["providers"]["harbor-endpoint"]["baseUrl"] == f"http://policy-{index}.invalid/v1"
        assert observed["policy_key"] == f"key-{index}"
        assert observed["cuda"] == "" and not observed["judge_key_inherited"]
        assert observed["turn_extension"]
        return observed

    async def check():
        first, second = await asyncio.gather(run(0), run(1))
        assert first["config"] != second["config"] and first["home"] != second["home"]

    asyncio.run(check())
