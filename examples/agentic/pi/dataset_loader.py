"""Load Pi tasks using the adapter registered by this example.

For another source format, pass a callable or ``file.py:read_rows`` as
HarnessRunner.loader_fn, with source-specific options in loader_kwargs.
The callable returns task records; it can read CSV, a database, an SDK, etc.
CLI callers can also supply their own --dataset-loader-fn file directly.
"""

from areno.api.harness import HarnessRunner
from examples.agentic.pi import svg_dataset  # noqa: F401 -- register the example adapter

_workflow = HarnessRunner(harness="pi", dataset="svg_icon")


def load_training_dataset(dataset_path, *, default_loader, **kwargs):
    return _workflow.dataset_loader(dataset_path, default_loader=default_loader, **kwargs)
