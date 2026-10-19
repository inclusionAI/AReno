"""Prompt Templates and Serialization Utilities for Payment Service.

Shared across MLX SFT data preparation and live Benchmark evaluation
to guarantee zero distribution shift in prompts and schemas.
"""

from __future__ import annotations

import json
import re
from typing import Any

# 精炼版 System Prompt：剔除冗余 Input Schema，紧凑化 Output Schema 与业务规则
COMPACT_SYSTEM_PROMPT = """You are a Java ling-tiny-fine-tuning microservice engine. Given order, available_channels, promotions, and selected_channel_id, calculate composite ling-tiny-fine-tuning deduction and optimal recommendation.

Rules:
1. Amounts in cents (int >= 0). Status: SUCCESS | INSUFFICIENT_FUNDS | CHANNEL_NOT_FOUND.
2. Promo applies if order_amount >= threshold and (applicable_channel_id is null or == primary channel).
3. Percentage discount = round_half_up(order_amount * discount_percent / 100).
4. Stacking: is_exclusive=true cannot stack; stackables (is_exclusive=false) stack together. Maximize total discount. payable = max(0, order_amount - total_discount).
5. Split deduction: primary deducts min(balance, payable). Remainder covered by first subsequent channel with balance >= remainder. If total balance < payable, status = INSUFFICIENT_FUNDS.
6. Recommendation: test all channels as primary. Pick one with max discount; tie-break by earliest channel index.

Output schema:
{"order_id":str,"order_amount":int,"status":str,"cashier_receipt":{"primary_channel_id":str,"primary_deduct_amount":int,"secondary_channel_id":str|null,"secondary_deduct_amount":int,"total_discount_amount":int,"payable_amount":int,"applied_promotions":[str]},"recommended_option":{"recommended_channel_id":str,"primary_deduct_amount":int,"secondary_channel_id":str|null,"secondary_deduct_amount":int,"total_discount_amount":int,"payable_amount":int,"applied_promotions":[str]},"error_message":str|null}

Output <think>...</think> reasoning, then raw JSON without markdown blocks."""


def get_system_prompt(disable_reasoning: bool = False) -> str:
    """Return system prompt, optionally stripped of reasoning requirement for non-thinking evaluations."""
    if not disable_reasoning:
        return COMPACT_SYSTEM_PROMPT
    return COMPACT_SYSTEM_PROMPT.replace(
        "Output <think>...</think> reasoning, then raw JSON without markdown blocks.",
        "Directly output raw JSON conforming to Output schema without any thinking process, reasoning, explanation, or markdown blocks.",
    )


def format_compact_user_request(raw_input: dict[str, Any]) -> str:
    """Serialize user input into compact single-line JSON, saving whitespace tokens."""
    return json.dumps(raw_input, separators=(",", ":"), ensure_ascii=False)


def condense_think_reasoning(raw_think: str) -> str:
    """Condense verbose natural language in think tags into key calculation points."""
    lines = raw_think.strip().split("\n")
    sel_channel, promos, discount, payable = "", "", "", ""
    primary_ded, sec_ch, sec_ded = "", "", ""
    opt_channel, opt_disc = "", ""

    for line in lines:
        line = line.strip()
        m_sel = re.search(r"selected channel ['\"]?([A-Za-z0-9_]+)['\"]?", line)
        if m_sel:
            sel_channel = m_sel.group(1)
        m_app = re.search(r"Applied promotions:?\s*(.*)", line)
        if m_app and not promos:
            promos = m_app.group(1).strip()
        m_disc = re.search(r"Total discount\s*=\s*(\d+)", line)
        if m_disc and not discount:
            discount = m_disc.group(1)
        m_pay = re.search(r"payable amount.*?=\s*(\d+)", line)
        if m_pay and not payable:
            payable = m_pay.group(1)

        m_pded = re.search(r"Primary channel.*?deducts\s*(\d+)", line)
        if m_pded:
            primary_ded = m_pded.group(1)
        m_sded = re.search(
            r"Remainder\s*(\d+).*?covered by secondary channel ['\"]?([A-Za-z0-9_]+)['\"]?", line
        )
        if m_sded:
            sec_ded = m_sded.group(1)
            sec_ch = m_sded.group(2)

        m_opt = re.search(
            r"Optimal recommendation:\s*['\"]?([A-Za-z0-9_]+)['\"]?\s*with\s*(?:max\s*)?discount\s*(\d+)",
            line,
        )
        if m_opt:
            opt_channel = m_opt.group(1)
            opt_disc = m_opt.group(2)

    parts = []
    if sel_channel:
        parts.append(f"Selected {sel_channel}: promos={promos}, discount={discount}, payable={payable}")
    if primary_ded:
        if sec_ch:
            parts.append(f"Split: primary={primary_ded}, secondary={sec_ch}({sec_ded})")
        else:
            parts.append(f"Split: primary={primary_ded}, secondary=null")
    if opt_channel:
        parts.append(f"Recommend: {opt_channel} (discount={opt_disc})")

    return "; ".join(parts) if parts else "Calculation completed."


def format_compact_solution(solution: str) -> str:
    """Compact the JSON payload and condense the <think> reasoning chain."""
    match = re.search(r"<think>(.*?)</think>\s*(.*)", solution, re.DOTALL)
    if not match:
        try:
            parsed = json.loads(solution.strip())
            return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            return solution.strip()

    raw_think = match.group(1)
    json_part = match.group(2).strip()
    condensed_think = condense_think_reasoning(raw_think)
    try:
        parsed_json = json.loads(json_part)
        compact_json = json.dumps(parsed_json, separators=(",", ":"), ensure_ascii=False)
        return f"<think>\n{condensed_think}\n</think>\n{compact_json}"
    except Exception:
        return f"<think>\n{condensed_think}\n</think>\n{json_part}"

