"""Load Pi tasks using the adapter registered by this example.

For another source format, pass a callable or ``file.py:read_rows`` as
HarnessRunner.loader_fn, with source-specific options in loader_kwargs.
The callable returns task records; it can read CSV, a database, an SDK, etc.
CLI callers can also supply their own --dataset-loader-fn file directly.
"""

from pathlib import Path
from runpy import run_path

from areno.api.harness import HarnessRunner

run_path(str(Path(__file__).with_name("frontend_dataset.py")))
_workflow = HarnessRunner(harness="pi", dataset="frontend")


def load_training_dataset(dataset_path, *, default_loader, **kwargs):
    return _workflow.dataset_loader(dataset_path, default_loader=default_loader, **kwargs)
