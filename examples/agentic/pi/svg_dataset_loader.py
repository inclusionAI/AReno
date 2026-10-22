"""Normalize local SVG task records using the registered example adapter."""

from areno.api.harness import HarnessRunner
from examples.agentic.pi import svg_dataset  # noqa: F401 -- register the example adapter


def load_training_dataset(dataset_path, *, default_loader, **kwargs):
    return HarnessRunner(harness="pi", dataset="svg_icon").dataset_loader(
        dataset_path, default_loader=default_loader, **kwargs
    )
