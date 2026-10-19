"""Strict payment scoring regressions; no model, network, or accelerator needed."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "recipes" / "ling-tiny-fine-tuning" / "recipe.py"
_SPEC = importlib.util.spec_from_file_location("ling_recipe", _PATH)
recipe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recipe)


@pytest.fixture
def answer():
    return {
        "order_id": "example", "order_amount": 100, "status": "SUCCESS",
        "cashier_receipt": {"payable_amount": 90, "applied_promotions": ["A", "B"]},
        "recommended_option": None, "error_message": None,
    }


def test_key_order_and_whitespace_do_not_change_score(answer):
    reordered = dict(reversed(list(answer.items())))
    assert recipe.strict_score(json.dumps(reordered, indent=2), answer) == {
        "valid_json": True, "exact_match": True,
    }


@pytest.mark.parametrize("amount", [100.0, True, "100", 99])
def test_amounts_require_exact_integer_values(answer, amount):
    actual = {**answer, "order_amount": amount}
    assert not recipe.strict_score(json.dumps(actual), answer)["exact_match"]


def test_booleans_are_not_integer_cents():
    assert not recipe.strict_score('{"amount": true}', {"amount": 1})["exact_match"]


@pytest.mark.parametrize("text", ['{"a": 1, "a": 1}', '{"a": NaN}', '```json\n{"a": 1}\n```',
                                    'Answer: {"a": 1}', '{"a": 1} trailing'])
def test_malformed_or_wrapped_json_is_rejected(text):
    assert recipe.strict_score(text, {"a": 1}) == {"valid_json": False, "exact_match": False}


def test_extra_missing_nested_fields_and_array_order_fail(answer):
    extra = {**answer, "explanation": "ok"}
    missing = {k: v for k, v in answer.items() if k != "order_id"}
    nested = {**answer, "cashier_receipt": {"payable_amount": 90}}
    reordered = {**answer, "cashier_receipt": {"payable_amount": 90, "applied_promotions": ["B", "A"]}}
    for actual in (extra, missing, nested, reordered):
        assert not recipe.strict_score(json.dumps(actual), answer)["exact_match"]


def test_error_responses_are_scored_in_full():
    expected = {"status": "ERROR", "cashier_receipt": None, "error_message": "CHANNEL_NOT_FOUND"}
    assert recipe.strict_score(json.dumps(expected), expected)["exact_match"]
    assert not recipe.strict_score(json.dumps({**expected, "error_message": "INSUFFICIENT_FUNDS"}), expected)["exact_match"]


def test_valid_json_is_not_necessarily_a_correct_answer(answer):
    assert recipe.strict_score("[]", answer) == {"valid_json": True, "exact_match": False}
