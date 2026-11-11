"""Reuse native Harbor Pi with separate config and home paths for local trials.

Local execution shares the host filesystem; this adapter separates concurrent
sessions but does not provide an OS sandbox. Pi and Node must be preinstalled.
"""

from pathlib import Path
from shlex import quote

from harbor.agents.installed.pi import Pi
from harbor.models.trial.config import AgentConfig

from areno.agent.harbor_agent import PreinstalledAgent


class PiWorkspace:
    def __init__(self, environment, workspace):
        self.environment = environment
        self.workspace = Path(workspace)
        self.config = self.workspace.parent / "pi-config"
        self.home = self.workspace.parent / "pi-home"
        self.temp = self.workspace.parent / "pi-tmp"
        for directory in (self.config, self.home / ".nvm", self.temp):
            directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        # Native Pi sources nvm before execution; Node is already on PATH.
        (self.home / ".nvm" / "nvm.sh").write_text(":\n")

    def __getattr__(self, name):
        return getattr(self.environment, name)

    async def upload_file(self, source_path, target_path):
        target_path = str(target_path).replace("/tmp/harbor-pi-agent", str(self.config))
        await self.environment.upload_file(source_path, target_path)

    async def exec(self, command, cwd=None, env=None, **kwargs):
        command = command.replace("/tmp/harbor-pi-agent", quote(str(self.config)))
        return await self.environment.exec(
            command,
            cwd=cwd or str(self.workspace),
            env={**(env or {}), "HOME": str(self.home), "TMPDIR": str(self.temp), "CUDA_VISIBLE_DEVICES": ""},
            **kwargs,
        )


class LocalPi(PreinstalledAgent):
    options_model = Pi.options_model

    def __init__(self, *args, **kwargs):
        config = AgentConfig(
            name="pi",
            model_name=kwargs.get("model_name"),
            env=kwargs.get("extra_env", {}),
            kwargs={name: kwargs[name] for name in Pi.options_model.model_fields if name in kwargs},
        )
        super().__init__(
            *args, native_config=config.model_dump(mode="json", context={"redact_sensitive_env": False}), **kwargs
        )

    @staticmethod
    def name():
        return "local-pi"

    async def _workspace(self, environment):
        location = await environment.exec("pwd")
        if location.return_code != 0 or not location.stdout:
            raise RuntimeError("could not resolve the local Pi workspace")
        return PiWorkspace(environment, location.stdout.strip())

    async def run(self, instruction, environment, context):
        await super().run(instruction, await self._workspace(environment), context)

    async def resume(self, instruction, environment, context):
        await super().resume(instruction, await self._workspace(environment), context)

    async def load(self, instruction, environment, context):
        await super().load(instruction, await self._workspace(environment), context)
