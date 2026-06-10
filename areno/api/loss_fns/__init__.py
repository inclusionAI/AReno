"""Loss functions consumed by `Trainer.train`.

Five flavours are bundled:
    dpo_loss_fn  - pairwise preference loss over chosen/rejected rows.
    grpo_loss_fn - PPO-style clipping applied per response token (GRPO).
    gspo_loss_fn - sequence-level clipping using a length-averaged log ratio.
    ppo_loss_fn  - clipped actor loss with optional KL penalty, matching the
                   form used by verl for actor-critic PPO.
    sft_loss_fn  - supervised next-token NLL on assistant/target tokens.
Each function accepts the same packed/padded `data_pack` dictionary so the trainer can
swap algorithms without re-shaping its data path.
"""

from areno.api.loss_fns.dpo import dpo_loss_fn
from areno.api.loss_fns.grpo import grpo_loss_fn
from areno.api.loss_fns.gspo import gspo_loss_fn
from areno.api.loss_fns.ppo import ppo_loss_fn
from areno.api.loss_fns.sft import sft_loss_fn

__all__ = ["dpo_loss_fn", "grpo_loss_fn", "gspo_loss_fn", "ppo_loss_fn", "sft_loss_fn"]
