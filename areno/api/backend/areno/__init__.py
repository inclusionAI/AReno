"""areno backend package.

Importing this module triggers `ArenoBackend`'s `@register_backend`
decorator so the `Trainer` factory can resolve `BackendType.Areno`.
"""

from areno.api.backend.areno.backend import ArenoBackend

__all__ = ["ArenoBackend"]
