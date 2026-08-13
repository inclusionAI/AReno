from types import SimpleNamespace
from unittest.mock import patch

import torch

from areno.engine.protocol import ScorePayload
from areno.engine.roles import RoleManager


def test_score_logprobs_omits_empty_feature_rows_for_text_model():
    class TextModel:
        def __call__(self, *, input_ids, train_meta):
            del train_meta
            return SimpleNamespace(logits_shard=torch.zeros((*input_ids.shape, 8)))

    manager = RoleManager(SimpleNamespace(device=torch.device("cpu")))
    payload = ScorePayload(
        role="ref",
        token_rows_by_dp=[[[1, 2, 3]]],
        features_by_dp=[[]],
        pad_token_id=0,
    )

    with patch("areno.engine.roles.next_token_logprobs", return_value=torch.zeros((1, 2))):
        rows = manager._score_logprob_rows(TextModel(), [[1, 2, 3]], payload, features=[None])

    assert rows == [[0.0, 0.0, 0.0]]
