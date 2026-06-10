"""Execution backends for `areno.api.Trainer`.

Concrete backends live in submodules and register themselves with the
`Backend` registry in `base.py` via `@register_backend`. The factory in
`base.get_backend_cls` imports them lazily so picking one backend does not
force the other's dependency stack to load.
"""
