"""Reuse Harbor agents with an explicit policy for preinstalled tools.

Harbor's installed-agent setup performs package installation. This wrapper skips
that setup only when tools are supplied by the environment, while delegating
execution, resume, model configuration, error handling and ATIF conversion to
the original agent. Custom agents still receive their setup hook.
"""

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.trial.config import AgentConfig


class PreinstalledAgent(BaseAgent):
    def __init__(self, *args, native_config: dict, **kwargs):
        super().__init__(*args, **kwargs)
        config = AgentConfig.model_validate(native_config)
        self.delegate = AgentFactory.create_agent_from_config(
            config,
            logs_dir=self.logs_dir,
            logger=self.logger,
            environment_logs_dir=self.environment_logs_dir,
            mcp_servers=self.mcp_servers,
            skills_dir=self.skills_dir,
        )
        self.capabilities = self.delegate.capabilities

    @staticmethod
    def name():
        return "harbor-preinstalled"

    def version(self):
        return self.delegate.version()

    def to_agent_info(self):
        return self.delegate.to_agent_info()

    @property
    def model_connection(self):
        return self.delegate.model_connection

    def _sync_identity(self):
        self.delegate.session_id = self.session_id
        self.delegate.context_id = self.context_id
        self.delegate.environment_logs_dir = self.environment_logs_dir

    async def setup(self, environment):
        self._sync_identity()
        if not isinstance(self.delegate, BaseInstalledAgent):
            await self.delegate.setup(environment)

    async def run(self, instruction, environment, context):
        self._sync_identity()
        await self.delegate.run(instruction, environment, context)

    async def resume(self, instruction, environment, context):
        self._sync_identity()
        await self.delegate.resume(instruction, environment, context)

    async def load(self, instruction, environment, context):
        self._sync_identity()
        await self.delegate.load(instruction, environment, context)

    def populate_context_post_run(self, context):
        self.delegate.populate_context_post_run(context)
