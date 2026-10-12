"""Task-level reward, shared across each attempt's model turns by AReno."""


def reward_fn(record):
    return float(record.source_record["pi_result"]["reward"])
