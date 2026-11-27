import pytest
import torch
from torch import nn

from areno.engine.parallel.collectives import sequence_parallel_region
from areno.engine.runtime.decoder import project_lm_head


@pytest.mark.parametrize("positions", [[], [3], [2, 0, 2]])
def test_selected_head_projects_only_requested_rows(positions):
    torch.manual_seed(29)
    head = nn.Linear(8, 16, bias=False)
    states = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    indices = torch.tensor(positions, dtype=torch.long)
    expected = head(states.float()).index_select(1, indices)
    calls = []
    head.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape))
    assert project_lm_head(head, states, defer=True, logits_indices=indices) is None
    selected = project_lm_head(head, states, defer=False, logits_indices=indices, dtype=torch.float32)
    torch.testing.assert_close(selected, expected)
    assert calls == [(2, len(indices), 8)]


def test_sample_indices_address_global_sequence_parallel_positions():
    class GatherHead(nn.Module):
        def forward(self, local):
            return torch.cat([local, local + 10], dim=1)

    local = torch.tensor([[[1.0], [2.0]]])
    with sequence_parallel_region(True):
        selected = project_lm_head(GatherHead(), local, defer=False, logits_indices=torch.tensor([3, 0]))
    torch.testing.assert_close(selected, torch.tensor([[[12.0], [1.0]]]))
