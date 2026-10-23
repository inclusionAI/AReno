"""Share the core task reward across each attempt's model turns."""

from areno.api.harness import HarnessRunner


def reward_fn(record):
    return HarnessRunner.reward_fn(record)
