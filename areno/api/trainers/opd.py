"""On-Policy Distillation trainer."""

from areno.api.trainers.policy_only import (
    PolicyOnlyTrainer,
)


class OPDTrainer(PolicyOnlyTrainer):
    """OPD trainer; implementation follows in Stages 2–4."""

    def fit(self) -> None:
        raise NotImplementedError(
            "OPD training is not yet implemented. "
            "Complete Stages 2–4 first."
        )