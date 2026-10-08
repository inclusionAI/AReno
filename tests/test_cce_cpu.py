"""CPU contracts for CCE configuration and packed loss alignment."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from areno.api.trainer_config import PolicyTrainerConfig, TrainerConfig
from areno.engine.config import RuntimeConfig
from areno.engine.runtime.logprobs import packed_cut_logprobs
from tests.helpers import single_tp_context


def test_cce_default_and_backend_propagation():
    assert RuntimeConfig().cce
    for cls in (TrainerConfig, PolicyTrainerConfig):
        for enabled in (True, False):
            config = cls(
                backend="cuda", algo="gspo", ckpt="dummy", dataset_path="dummy", cce=enabled, world_size=1, tp_size=1
            )
            assert config.backend_config().runtime["cce"] is enabled


@pytest.mark.parametrize("cap", [0.0, 3.0])
def test_cce_packed_boundaries_and_gradient_mask(cap):
    torch.manual_seed(1)
    x = torch.randn(1, 8, 5, requires_grad=True)
    w = torch.randn(11, 5, requires_grad=True)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    cu = torch.tensor([0, 3, 4, 8])
    positions = torch.tensor([0, 1, 4, 5, 6])
    labels = tokens.flatten()[positions + 1]
    logits = F.linear(x[0, positions], w)
    if cap:
        logits = cap * (logits / cap).tanh()
    expected = logits.log_softmax(-1).gather(1, labels[:, None]).flatten()

    def reference(hidden, weight, selected, **kwargs):
        z = F.linear(hidden, weight)
        if kwargs["softcap"]:
            z = kwargs["softcap"] * (z / kwargs["softcap"]).tanh()
        return z.log_softmax(-1).gather(1, selected[:, None]).flatten()

    with (
        patch("areno.accel.cce.cut_logprobs", reference),
        patch("areno.engine.runtime.logprobs.get_tp_context", single_tp_context),
    ):
        actual = packed_cut_logprobs(x, tokens, cu, SimpleNamespace(weight=w, vocab_start=0), logit_softcap=cap)
    torch.testing.assert_close(actual, expected)
    scale = torch.tensor([1.0, 0.0, -0.5, 0.0, 2.0])
    ref = torch.autograd.grad((expected * scale).sum(), (x, w))
    got = torch.autograd.grad((actual * scale).sum(), (x, w))
    for a, b in zip(got, ref, strict=True):
        torch.testing.assert_close(a, b)
