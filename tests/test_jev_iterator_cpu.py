"""Resumed iteration must match uninterrupted training across epoch boundaries."""

import json

import pytest

from areno.api.jev.data import QuestionIterator


def test_resumed_batches_match_uninterrupted_across_short_batch_and_epoch():
    dataset = [{"id": i} for i in range(19)]
    uninterrupted = QuestionIterator(dataset, seed=17, batch_size=8)
    expected = [uninterrupted.take() for _ in range(9)]
    segmented = QuestionIterator(dataset, seed=17, batch_size=8)
    actual = []
    for _ in range(3):
        actual.extend(segmented.take() for _ in range(3))
        state = json.loads(json.dumps(segmented.state_dict()))
        segmented = QuestionIterator(dataset, seed=17, batch_size=8, state=state)
    assert actual == expected
    assert segmented.state_dict() == uninterrupted.state_dict()
    assert sorted(row["id"] for batch in actual[:3] for row in batch) == list(range(19))


@pytest.mark.parametrize("change", ["dataset", "seed", "batch_size"])
def test_resume_rejects_changed_data_or_order(change):
    dataset, seed, batch_size = [{"id": i} for i in range(19)], 17, 8
    iterator = QuestionIterator(dataset, seed=seed, batch_size=batch_size)
    iterator.take()
    state = iterator.state_dict()
    if change == "dataset":
        dataset = list(reversed(dataset))
    elif change == "seed":
        seed += 1
    else:
        batch_size = 4
    with pytest.raises(ValueError, match="same dataset"):
        QuestionIterator(dataset, seed=seed, batch_size=batch_size, state=state)
