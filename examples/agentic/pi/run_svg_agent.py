"""Run the native Harbor Pi agent with the SVG example adapter."""

from examples.agentic.pi import svg_dataset  # noqa: F401 -- register the example adapter
from examples.agentic.pi.run_agent import create_runner


async def run_agent(ctx, batch):
    return await create_runner(dataset="svg_icon")(ctx, batch)
