from __future__ import annotations

import types


class PatchedContext:
    def __init__(self, module, **attrs):
        self.module = module
        self.attrs = attrs
        self.old = {}

    def __enter__(self):
        for name, value in self.attrs.items():
            self.old[name] = getattr(self.module, name)
            setattr(self.module, name, value)

    def __exit__(self, exc_type, exc, tb):
        for name, value in self.old.items():
            setattr(self.module, name, value)


def single_tp_context():
    return types.SimpleNamespace(rank=0, world_size=1, group=None)
