#!/usr/bin/env python3
"""Print the active AReno algorithm registry as JSON."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import Result, build_parser, skill_main


@skill_main
def main() -> Result:
    build_parser("Print the active AReno algorithm registry as JSON.").parse_args()

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
    return Result(ok=True, data={"algorithms": rows})


if __name__ == "__main__":
    raise SystemExit(main())