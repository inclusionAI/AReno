"""Sample-utilization funnel accounting for AReno training updates.

A "funnel" reconciles how many samples survive each stage of one training
update::

    loaded -> contract-valid -> generated -> length-valid
           -> trainable-token-valid -> actually-trained

`drop_reasons` explains the delta between adjacent stages. This module is
intentionally pure and free of GPU/runtime dependencies so it can be unit-tested
on CPU and reused by both the metric recorder (persistence) and the
``areno funnel`` CLI (rendering). It never carries sample *contents* -- only
integer counts and short reason codes -- so funnel artifacts stay safe to print.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Ordered funnel stages, from raw dataset rows to sequences that reached the
# backend training step. The CLI renders them in this order.
STAGE_ORDER = (
    "loaded",
    "contract_valid",
    "generated",
    "length_valid",
    "trainable_token_valid",
    "trained",
)

# Stages that may legitimately be unset (``None``) for trainers without a
# rollout phase (SFT/DPO have no generation step). ``reconcile`` uses this to
# distinguish "untracked stage" from "broken accounting".
_OPTIONAL_STAGES = frozenset({"generated", "length_valid"})

# The funnel spans two unit domains. The prompt-domain stages count dataset
# rows; the completion-domain stages count rollouts (which fan out by
# ``n_samples``). A monotonic "later <= earlier" check is only meaningful
# *within* a domain -- ``generated`` legitimately exceeds ``contract_valid`` by
# the rollout fan-out factor, so the two domains must not be compared.
_STAGE_CHAINS = (
    ("loaded", "contract_valid"),  # prompt-domain
    ("generated", "length_valid", "trainable_token_valid", "trained"),  # completion-domain
)


@dataclass
class FunnelCounters:
    """Per-update sample funnel counters.

    Each count field is either a non-negative int or ``None`` ("not tracked for
    this trainer path"). ``drop_reasons`` maps a stage name to short reason
    codes -- never sample text.
    """

    step: int
    source: str  # "sft" | "dpo" | "online_rl"
    loaded: int | None = None
    contract_valid: int | None = None
    generated: int | None = None
    length_valid: int | None = None
    trainable_token_valid: int | None = None
    trained: int | None = None
    drop_reasons: dict[str, list[str]] = field(default_factory=dict)


def build_funnel(counters: FunnelCounters) -> dict:
    """Serialize a ``FunnelCounters`` to a JSON-able dict.

    Only count fields and reason codes are emitted; any stray sample content is
    dropped by construction -- the dataclass has no field for it.
    """

    return {
        "step": int(counters.step),
        "source": counters.source,
        "stages": {
            "loaded": counters.loaded,
            "contract_valid": counters.contract_valid,
            "generated": counters.generated,
            "length_valid": counters.length_valid,
            "trainable_token_valid": counters.trainable_token_valid,
            "trained": counters.trained,
        },
        "drop_reasons": {
            stage: [str(reason) for reason in reasons] for stage, reasons in counters.drop_reasons.items()
        },
    }


def reconcile(counters: FunnelCounters) -> list[str]:
    """Return human-readable warnings for funnel inconsistencies.

    Never raises: a malformed or partial funnel yields warnings instead. Stages
    in ``_OPTIONAL_STAGES`` are allowed to be unset; other unset stages and
    negative counts always warn. Monotonic "later <= earlier" checks run only
    *within* a unit domain (see ``_STAGE_CHAINS``) -- the prompt-domain and
    completion-domain counts are not compared because rollout fan-out makes a
    later completion count legitimately larger than the prior prompt count.
    """

    warnings: list[str] = []
    stages = build_funnel(counters)["stages"]
    for name in STAGE_ORDER:
        value = stages[name]
        if value is None:
            if name not in _OPTIONAL_STAGES:
                warnings.append(f"stage '{name}' is not tracked for source '{counters.source}'")
            continue
        if value < 0:
            warnings.append(f"stage '{name}' has negative count {value}")
    for chain in _STAGE_CHAINS:
        last_name: str | None = None
        last_val: int | None = None
        for name in chain:
            value = stages[name]
            if value is None:
                continue
            if last_val is not None and value > last_val:
                warnings.append(
                    f"stage '{name}' ({value}) exceeds prior tracked stage '{last_name}' ({last_val}) "
                    f"in the {chain[0]} chain; samples cannot appear at a later stage"
                )
            last_name = name
            last_val = value
    return warnings
