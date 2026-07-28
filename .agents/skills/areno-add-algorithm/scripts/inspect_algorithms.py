#!/usr/bin/env python3
"""Print the active AReno algorithm registry as JSON."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    argparse.ArgumentParser().parse_args()
    try:
        from areno.api.algorithms import list_algorithms

        rows = []
        for name, spec in sorted(list_algorithms().items()):
            trainer = spec.resolve_trainer_cls()
            rows.append(
                {
                    "name": name,
                    "requires_rollout": spec.requires_rollout,
                    "experimental": spec.experimental,
                    "trainer": f"{trainer.__module__}.{trainer.__name__}",
                    "loss": f"{spec.default_loss_fn.__module__}.{spec.default_loss_fn.__name__}",
                    "custom_loss_factory": spec.loss_fn_factory is not None,
                }
            )
        result = {"ok": True, "algorithms": rows}
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
