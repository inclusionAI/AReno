"""Command-line interface for the installed areno tool."""

from __future__ import annotations

import sys
from pathlib import Path

import click

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class ArenoCli(click.Group):
    """Click group that imports heavy subcommands only when selected."""

    _COMMANDS = {
        "train": ("areno.cli.train", "train_command", "Run SFT, DPO, GSPO, GRPO, or PPO training."),
        "serve": ("areno.cli.serve", "serve_command", "Serve an OpenAI-compatible chat API."),
    }

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(self._COMMANDS)

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        target = self._COMMANDS.get(name)
        if target is None:
            return None
        module_name, attr_name, _ = target
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = [(name, self._COMMANDS[name][2]) for name in self.list_commands(ctx)]
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.group(
    cls=ArenoCli,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run local LLM post-training jobs or start the OpenAI-compatible areno server.",
)
def main() -> None:
    """Top-level areno command group."""


if __name__ == "__main__":
    main()
