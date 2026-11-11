from areno.engine.inference import _cancel_stop_token


def test_cancel_stop_token_prefers_explicit_stop_ids():
    assert _cancel_stop_token((7, 9), (1, 2)) == 7


def test_cancel_stop_token_falls_back_to_eos_tuple():
    assert _cancel_stop_token((), (1, 2)) == 1


def test_cancel_stop_token_scalar_eos():
    assert _cancel_stop_token((), 5) == 5


def test_cancel_stop_token_empty_eos_tuple_from_ignore_eos():
    # ignore_eos=True makes the CUDA backend pass eos_token_id=() with no
    # stop tokens; this must not raise (regression: int(()) TypeError).
    assert _cancel_stop_token((), ()) == 0


def test_cancel_stop_token_none_eos():
    assert _cancel_stop_token((), None) == 0
