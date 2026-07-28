"""Pass through already-normalized rows for the deterministic mix example."""


def load_training_dataset(dataset_path, *, default_loader, **_kwargs):
    return default_loader(dataset_path)
