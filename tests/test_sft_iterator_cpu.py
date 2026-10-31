import copy

import numpy as np
import pytest

from areno.api.trainers.sft_iterator import SFTBatchIterator


def _convert(value):
    return None if value % 3 == 0 else value


@pytest.mark.parametrize("shuffle", [False, True])
def test_iterator_resume_preserves_order_with_filtered_rows_and_final_batch(shuffle):
    rows = list(range(13))
    expected = SFTBatchIterator(rows, _convert, 3, np.random.RandomState(91), shuffle=shuffle)
    expected_batches = list(expected)

    interrupted = SFTBatchIterator(rows, _convert, 3, np.random.RandomState(91), shuffle=shuffle)
    first = next(interrupted)
    state = interrupted.state_dict()
    resumed = SFTBatchIterator(rows, _convert, 3, np.random.RandomState(300), shuffle=shuffle, state=state)

    assert [first, *resumed] == expected_batches
    assert resumed.state_dict() == expected.state_dict()
    assert len(expected_batches[-1]) < 3


def test_iterator_resume_at_end_does_not_report_empty_dataset():
    iterator = SFTBatchIterator([1, 3], _convert, 2, np.random.RandomState(1), shuffle=False)
    assert next(iterator) == [1]
    resumed = SFTBatchIterator(
        [1, 3], _convert, 2, np.random.RandomState(1), shuffle=False, state=iterator.state_dict()
    )
    assert list(resumed) == []


def test_iterator_rejects_invalid_progress():
    iterator = SFTBatchIterator([1, 2], _convert, 1, np.random.RandomState(1), shuffle=False)
    next(iterator)
    state = copy.deepcopy(iterator.state_dict())
    state["order"] = [0, 0]
    with pytest.raises(ValueError, match="invalid SFT dataset iterator state"):
        SFTBatchIterator([1, 2], _convert, 1, np.random.RandomState(1), shuffle=False, state=state)
