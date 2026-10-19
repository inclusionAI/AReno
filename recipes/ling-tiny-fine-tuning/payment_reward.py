"""Payment Reward Function and Verifier.

Implements 5-dimension stepped scoring (0.0 to 1.0) compatible with:
1. AReno post-training reward contract: reward_fn(record: RewardRecord) -> float
2. Offline testing and evaluation harness: verify_payment(...) -> tuple[float, dict]
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Robustly extract JSON payload from text, stripping <think> tags and markdown fences."""
    # 1. Remove thinking process if present
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    # 2. Extract from markdown code fence ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass

    # 3. Search for the outermost matching curly braces
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = cleaned[start_idx : end_idx + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 4. Final attempt: parse direct string
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def verify_payment(
    request: dict[str, Any],
    completion: str | dict[str, Any],
    ground_truth: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Evaluate completion against request and ground truth across 5 dimensions.

    Returns:
        (total_score, details_dict) where total_score is in [0.0, 1.0]
    """
    details: dict[str, Any] = {
        "dim1_json_syntax": 0.0,
        "dim2_conservation": 0.0,
        "dim3_promotion_rules": 0.0,
        "dim4_cashier_match": 0.0,
        "dim5_recommendation_match": 0.0,
        "error": None,
    }

    # Dimension 1: JSON Syntax & Basic Structure (0.20)
    if isinstance(completion, dict):
        pred_dict = completion
        details["dim1_json_syntax"] = 0.20
    else:
        pred_dict = extract_json_object(completion)
        if pred_dict is None or not isinstance(pred_dict, dict):
            details["error"] = "Failed to parse JSON completion"
            return 0.0, details
        if "order_id" not in pred_dict or "status" not in pred_dict:
            details["error"] = "Missing required top-level keys in JSON"
            return 0.0, details
        details["dim1_json_syntax"] = 0.20

    expected_status = ground_truth.get("status")
    pred_status = pred_dict.get("status")

    # If expected error status (e.g. INSUFFICIENT_FUNDS or CHANNEL_NOT_FOUND)
    if expected_status != "SUCCESS":
        if pred_status == expected_status:
            details["dim2_conservation"] = 0.20
            details["dim3_promotion_rules"] = 0.20
            details["dim4_cashier_match"] = 0.20
            details["dim5_recommendation_match"] = 0.20
            total_score = round(sum(details[k] for k in details if k.startswith("dim")), 4)
            return total_score, details
        else:
            details["error"] = f"Expected status {expected_status}, but got {pred_status}"
            total_score = round(details["dim1_json_syntax"], 4)
            return total_score, details

    if pred_status != "SUCCESS":
        details["error"] = f"Expected status SUCCESS, but got {pred_status}"
        return details["dim1_json_syntax"], details

    cashier = pred_dict.get("cashier_receipt")
    recommendation = pred_dict.get("recommended_option")

    if not isinstance(cashier, dict) or not isinstance(recommendation, dict):
        details["error"] = "cashier_receipt or recommended_option is missing or not a dict"
        return details["dim1_json_syntax"], details

    channels_map = {c["channel_id"]: c["balance"] for c in request.get("available_channels", [])}
    order_amount = int(request.get("order", {}).get("order_amount", 0))

    # Dimension 2: Balance Constraint & Arithmetic Conservation (0.20)
    def _check_plan_conservation(plan: dict[str, Any], is_cashier: bool) -> bool:
        ch_key = "primary_channel_id" if is_cashier else "recommended_channel_id"
        primary_ch = plan.get(ch_key)
        p_deduct = plan.get("primary_deduct_amount")
        s_ch = plan.get("secondary_channel_id")
        s_deduct = plan.get("secondary_deduct_amount")
        payable = plan.get("payable_amount")

        if not isinstance(p_deduct, int) or not isinstance(s_deduct, int) or not isinstance(payable, int):
            return False
        if p_deduct < 0 or s_deduct < 0 or payable < 0:
            return False
        if p_deduct + s_deduct != payable:
            return False
        if primary_ch not in channels_map or p_deduct > channels_map[primary_ch]:
            return False
        if s_ch is not None:
            if s_ch not in channels_map or s_deduct > channels_map[s_ch]:
                return False
        else:
            if s_deduct != 0:
                return False
        return True

    if _check_plan_conservation(cashier, True) and _check_plan_conservation(recommendation, False):
        details["dim2_conservation"] = 0.20
    else:
        details["error"] = "Conservation equation or channel balance violation"

    # Dimension 3: Promotion Math and Bounds (0.20)
    cashier_discount = cashier.get("total_discount_amount")
    cashier_payable = cashier.get("payable_amount")
    rec_discount = recommendation.get("total_discount_amount")
    rec_payable = recommendation.get("payable_amount")

    if (
        isinstance(cashier_discount, int)
        and isinstance(cashier_payable, int)
        and isinstance(rec_discount, int)
        and isinstance(rec_payable, int)
        and cashier_payable == max(0, order_amount - cashier_discount)
        and rec_payable == max(0, order_amount - rec_discount)
        and 0 <= cashier_discount <= order_amount
        and 0 <= rec_discount <= order_amount
    ):
        details["dim3_promotion_rules"] = 0.20
    else:
        if details["error"] is None:
            details["error"] = "Payable amount does not equal order_amount - total_discount_amount"

    # Dimension 4: Cashier Receipt Exact Match (0.20)
    exp_cashier = ground_truth.get("cashier_receipt", {})
    if (
        cashier.get("primary_channel_id") == exp_cashier.get("primary_channel_id")
        and cashier.get("primary_deduct_amount") == exp_cashier.get("primary_deduct_amount")
        and cashier.get("secondary_channel_id") == exp_cashier.get("secondary_channel_id")
        and cashier.get("secondary_deduct_amount") == exp_cashier.get("secondary_deduct_amount")
        and cashier.get("total_discount_amount") == exp_cashier.get("total_discount_amount")
        and cashier.get("payable_amount") == exp_cashier.get("payable_amount")
        and sorted(cashier.get("applied_promotions", [])) == sorted(exp_cashier.get("applied_promotions", []))
    ):
        details["dim4_cashier_match"] = 0.20
    else:
        if details["error"] is None:
            details["error"] = "Cashier receipt values do not match ground truth"

    # Dimension 5: Recommendation Exact Match & Tie-breaker (0.20)
    exp_rec = ground_truth.get("recommended_option", {})
    if (
        recommendation.get("recommended_channel_id") == exp_rec.get("recommended_channel_id")
        and recommendation.get("primary_deduct_amount") == exp_rec.get("primary_deduct_amount")
        and recommendation.get("secondary_channel_id") == exp_rec.get("secondary_channel_id")
        and recommendation.get("secondary_deduct_amount") == exp_rec.get("secondary_deduct_amount")
        and recommendation.get("total_discount_amount") == exp_rec.get("total_discount_amount")
        and recommendation.get("payable_amount") == exp_rec.get("payable_amount")
        and sorted(recommendation.get("applied_promotions", [])) == sorted(exp_rec.get("applied_promotions", []))
    ):
        details["dim5_recommendation_match"] = 0.20
    else:
        if details["error"] is None:
            details["error"] = "Recommendation values do not match ground truth"

    total_score = round(
        details["dim1_json_syntax"]
        + details["dim2_conservation"]
        + details["dim3_promotion_rules"]
        + details["dim4_cashier_match"]
        + details["dim5_recommendation_match"],
        4,
    )
    return total_score, details


def reward_fn(record: Any) -> float:
    """AReno standard reward function contract."""
    # record is an areno RewardRecord
    prompt = getattr(record, "prompt", "")
    completion = getattr(record, "completion", "")
    ground_truth = getattr(record, "answer", None)

    # If answer is None, check source_record
    if ground_truth is None and hasattr(record, "source_record"):
        ground_truth = record.source_record.get("ground_truth")

    # If ground_truth is string, parse json
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except Exception:
            return 0.0

    if not isinstance(ground_truth, dict):
        return 0.0

    # Extract request payload from prompt or source_record
    request_data = None
    if hasattr(record, "source_record") and "request" in record.source_record:
        request_data = record.source_record["request"]
    else:
        # Sniff JSON from prompt
        request_data = extract_json_object(prompt)

    if not isinstance(request_data, dict):
        request_data = {}

    score, _ = verify_payment(request_data, completion, ground_truth)
    return float(score)
