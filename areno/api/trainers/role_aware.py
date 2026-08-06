"""Shared lifecycle and scoring helpers for trainers with auxiliary model roles.

Role-aware algorithms initialize backend-owned models before training, score
exact token rows through those models, and leave placement/offload decisions to
the backend. DPO uses this for its frozen reference policy; PPO uses it for its
reference, critic, and optional reward roles. Online distillation algorithms can
reuse the same boundary for a frozen teacher.
"""

from __future__ import annotations

from areno.api.trainers.policy_only import PolicyOnlyTrainer


class RoleAwareTrainerMixin:
    """Provide role initialization and validated log-probability scoring.

    Subclasses must define ``areno``, ``config``, ``logger``, ``roles``, and an
    ``_fit_initialized()`` method. The mixin deliberately does not manage model
    placement itself; ``ensure_roles`` and scoring stay behind the backend API.
    """

    def fit(self) -> None:
        self.areno.init()
        try:
            self._ensure_roles()
            self._fit_initialized()
        finally:
            self.areno.close()

    def _ensure_roles(self) -> None:
        for role in self.roles.values():
            self.logger.info(
                "role=%s stage=init_start trainable=%s path=%s",
                role.name,
                role.trainable,
                role.path,
            )

        self.areno.ensure_roles(self.roles)

        for role in self.roles.values():
            self.logger.info(
                "role=%s stage=init_end trainable=%s",
                role.name,
                role.trainable,
            )

    def _score_logprobs(
        self,
        role: str,
        token_rows: list[list[int]],
    ) -> list[list[float]]:
        rows = self.areno.score_logprobs(
            role,
            token_rows,
            microbatch_size=self.config.score_micro_bs,
        )

        if len(rows) != len(token_rows):
            raise ValueError(f"role {role!r} returned {len(rows)} logprob rows for {len(token_rows)} token rows")

        for row_idx, (tokens, logprobs) in enumerate(zip(token_rows, rows, strict=True)):
            if len(logprobs) != len(tokens):
                raise ValueError(
                    f"role {role!r} returned {len(logprobs)} logprobs for token row {row_idx} with {len(tokens)} tokens"
                )

        return rows


class RoleAwarePolicyTrainer(
    RoleAwareTrainerMixin,
    PolicyOnlyTrainer,
):
    """Policy rollout loop with backend-owned auxiliary model roles."""


__all__ = [
    "RoleAwarePolicyTrainer",
    "RoleAwareTrainerMixin",
]
