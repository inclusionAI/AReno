"""Token-level On-Policy Distillation loss."""


def opd_loss_fn(
    data_pack,
    logprobs,
    *,
    opd_coef: float = 1.0,
):
    raise NotImplementedError(
        "OPD loss will be implemented in Stage 4."
    )