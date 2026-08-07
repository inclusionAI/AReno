from __future__ import annotations

import pytest
import torch

from areno.api.loss_fns.ipo import (
    _ipo_from_sequence_logps,
    ipo_loss_fn,
)

def test_ipo_sequence_loss_matches_hand_calculation():
    # Policy:
    # chosen - rejected = -2 - (-4) = 2
    #
    # Reference:
    # chosen - rejected = -3 - (-4) = 1
    #
    # delta = 2 - 1 = 1
    #
    # beta = 0.25 -> target = 1 / (2 * 0.25) = 2
    #
    # loss = (1 - 2)^2 = 1
    policy_seq_logps = torch.tensor([-2.0, -4.0])
    ref_seq_logps = torch.tensor([-3.0, -4.0])
    response_lens = torch.tensor([2.0, 2.0])

    loss, metrics = _ipo_from_sequence_logps(
        policy_seq_logps,
        ref_seq_logps,
        response_lens,
        beta=0.25,
    )

    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(metrics["ipo_delta"], torch.tensor(1.0))
    torch.testing.assert_close(
        metrics["ipo_target_error"],
        torch.tensor(1.0),
    )


def test_ipo_loss_is_zero_at_target():
    # beta = 0.1 -> target = 5
    #
    # Policy margin = 5
    # Reference margin = 0
    # delta = 5
    policy_seq_logps = torch.tensor([5.0, 0.0])
    ref_seq_logps = torch.tensor([0.0, 0.0])
    response_lens = torch.tensor([1.0, 1.0])

    loss, metrics = _ipo_from_sequence_logps(
        policy_seq_logps,
        ref_seq_logps,
        response_lens,
        beta=0.1,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(metrics["ipo_delta"], torch.tensor(5.0))
    torch.testing.assert_close(
        metrics["ipo_target_error"],
        torch.tensor(0.0),
    )


@pytest.mark.parametrize(
    ("delta", "expected_gradient_sign"),
    [
        (3.0, -1),
        (5.0, 0),
        (7.0, 1),
    ],
)
def test_ipo_gradient_moves_delta_toward_target(
    delta,
    expected_gradient_sign,
):
    # beta = 0.1 -> target = 5.
    #
    # delta < 5 should be pushed upward.
    # delta = 5 should have zero gradient.
    # delta > 5 should be pushed downward.
    policy_seq_logps = torch.tensor(
        [delta, 0.0],
        requires_grad=True,
    )
    ref_seq_logps = torch.zeros(2)
    response_lens = torch.ones(2)

    loss, _ = _ipo_from_sequence_logps(
        policy_seq_logps,
        ref_seq_logps,
        response_lens,
        beta=0.1,
    )
    loss.backward()

    chosen_gradient = policy_seq_logps.grad[0].item()

    if expected_gradient_sign < 0:
        assert chosen_gradient < 0
    elif expected_gradient_sign > 0:
        assert chosen_gradient > 0
    else:
        assert abs(chosen_gradient) < 1.0e-6


def test_ipo_loss_uses_sequence_sums_not_length_averages():
    # Chosen has two response tokens:
    #
    # policy sum = -2
    # ref sum = -3
    # policy-vs-ref shift = 1
    #
    # Rejected has one response token:
    #
    # policy sum = -3
    # ref sum = -3.5
    # policy-vs-ref shift = 0.5
    #
    # delta = 1 - 0.5 = 0.5
    #
    # beta = 1 -> target = 0.5, so sequence-sum IPO has zero loss.
    #
    # A length-averaged implementation would produce a different delta.
    logprobs = torch.tensor(
        [
            [0.0, -1.0, -1.0],
            [0.0, -3.0, 100.0],
        ]
    )

    data_pack = {
        "prompt_mask": torch.tensor(
            [
                [True, True, False, False],
                [True, True, False, True],
            ]
        ),
        "ref_logprobs": torch.tensor(
            [
                [0.0, 0.0, -1.5, -1.5],
                [0.0, 0.0, -3.5, 100.0],
            ]
        ),
    }

    loss, metrics = ipo_loss_fn(
        data_pack,
        logprobs,
        beta=1.0,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(
        metrics["ipo_delta"],
        torch.tensor(0.5),
    )


def test_ipo_requires_even_number_of_sequences():
    logprobs = torch.zeros((3, 2))

    data_pack = {
        "prompt_mask": torch.tensor(
            [
                [True, False, False],
                [True, False, False],
                [True, False, False],
            ]
        ),
        "ref_logprobs": torch.zeros((3, 3)),
    }

    with pytest.raises(
        ValueError,
        match="IPO requires an even number of sequences",
    ):
        ipo_loss_fn(
            data_pack,
            logprobs,
            beta=0.1,
        )


@pytest.mark.parametrize("beta", [0.0, -0.1])
def test_ipo_requires_positive_beta(beta):
    with pytest.raises(
        ValueError,
        match="IPO beta must be positive",
    ):
        ipo_loss_fn(
            {},
            torch.zeros((2, 1)),
            beta=beta,
        )