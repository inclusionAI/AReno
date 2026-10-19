"""Advanced Synthetic Dataset Generator for Composite Payment Microservice.

Generates high-difficulty, multi-channel, extreme price & promotion scenarios:
1. data/eval_testset_500.jsonl (500 samples, strictly isolated golden evaluation set)
2. data/sft_payment_train.jsonl (1,500 samples, includes synthetic <think> CoT draft)
3. data/rl_payment_prompts.jsonl (10,000 samples, raw input + ground truth for RLVR)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from recipes.payment.payment_service.models import (
    OrderInfo,
    PaymentChannel,
    PaymentRequest,
    PaymentResponse,
    PaymentStatus,
    Promotion,
    PromotionType,
)
from recipes.payment.payment_service.service import PaymentCalculationService

CHANNEL_PREFIXES = [
    "BANK_ICBC_DEBIT",
    "BANK_CCB_CREDIT",
    "BANK_BOC_CARD",
    "BANK_ABC_EXPRESS",
    "BANK_COMM_PAY",
    "BANK_CMB_GOLD",
    "BANK_SPDB_CARD",
    "BANK_PINGAN_PAY",
    "WALLET_ALIPAY_PLUS",
    "WALLET_WECHAT_PAY",
    "WALLET_MEITUAN_CASH",
    "WALLET_JD_PAY",
    "DIGITAL_RMB_WALLET",
    "CREDIT_ANT_HUABEI",
    "CREDIT_BAITIAO_PLUS",
    "UNIONPAY_QUICK_PASS",
    "REWARD_POINTS_VIP",
    "GIFT_VOUCHER_CARD",
    "PAYPAL_GLOBAL_DIRECT",
    "APPLE_WALLET_NFC",
]


def _generate_random_channel_ids(count: int) -> list[str]:
    """Generate `count` distinct randomized channel identifiers."""
    picked_prefixes = random.sample(CHANNEL_PREFIXES, min(count, len(CHANNEL_PREFIXES)))
    if count > len(CHANNEL_PREFIXES):
        # Allow reuse of prefix with different suffixes if count exceeds prefix count
        picked_prefixes.extend(random.choices(CHANNEL_PREFIXES, k=count - len(CHANNEL_PREFIXES)))
    
    channel_ids: list[str] = []
    used = set()
    for prefix in picked_prefixes:
        for _ in range(50):
            suffix = f"{random.randint(1000, 9999):04d}"
            cid = f"{prefix}_{suffix}"
            if cid not in used:
                used.add(cid)
                channel_ids.append(cid)
                break
    return channel_ids


def _request_fingerprint(req_dict: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 fingerprint for a request payload."""
    serialized = json.dumps(req_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def generate_cot_reasoning(req: PaymentRequest, res: PaymentResponse) -> str:
    """Synthesize 100% accurate step-by-step reasoning draft for SFT training."""
    order = req.order
    selected = req.selected_channel_id
    receipt = res.cashier_receipt
    rec = res.recommended_option

    lines = ["<think>"]
    lines.append(f"1. Order Context: order_id={order.order_id}, order_amount={order.order_amount} cents.")
    lines.append(f"2. User selected primary channel: {selected}.")
    lines.append(f"   Available channels count: {len(req.available_channels)}.")

    if res.status == PaymentStatus.CHANNEL_NOT_FOUND:
        lines.append(f"3. Validation check: selected channel '{selected}' not found in available channels list.")
        lines.append(f"4. Result: return status = {res.status.value}, cashier_receipt and recommended_option are null.")
        lines.append("</think>")
        return "\n".join(lines)

    if res.status == PaymentStatus.INSUFFICIENT_FUNDS:
        lines.append(f"3. Balance & promotion check for selected channel '{selected}':")
        if receipt and receipt.applied_promotions:
            lines.append(f"   - Eligible promotions: {', '.join(receipt.applied_promotions)} (total discount: {receipt.total_discount_amount} cents).")
            lines.append(f"   - Net payable amount = {receipt.payable_amount} cents.")
        else:
            lines.append(f"   - No promotions apply. Net payable amount = {order.order_amount} cents.")
        lines.append("   - Insufficient funds check: combined balance of primary and available secondary channels cannot cover payable amount.")
        lines.append(f"4. Result: return status = {res.status.value}.")
        lines.append("</think>")
        return "\n".join(lines)

    # SUCCESS branch
    lines.append(f"3. Promotion evaluation for selected channel '{selected}':")
    if receipt and receipt.applied_promotions:
        lines.append(f"   - Applied promotions: {', '.join(receipt.applied_promotions)}.")
        lines.append(f"   - Total discount = {receipt.total_discount_amount} cents.")
    else:
        lines.append("   - No promotions eligible or applicable. Total discount = 0 cents.")
    lines.append(f"   - Net payable amount = {order.order_amount} - {receipt.total_discount_amount} = {receipt.payable_amount} cents.")

    lines.append("4. Calculate cashier 2-leg split deduction:")
    lines.append(f"   - Primary channel '{selected}' deducts {receipt.primary_deduct_amount} cents.")
    if receipt.secondary_channel_id:
        lines.append(f"   - Primary balance insufficient. Remainder {receipt.secondary_deduct_amount} cents covered by secondary channel '{receipt.secondary_channel_id}'.")
    else:
        lines.append("   - Primary channel balance covers payable amount. Secondary channel = null.")

    lines.append("5. Global optimization search across all available channels for recommendation:")
    lines.append(f"   - Evaluated all {len(req.available_channels)} channels as candidate primary channels.")
    lines.append(f"   - Optimal recommendation: '{rec.recommended_channel_id}' with max discount {rec.total_discount_amount} cents.")
    if rec.applied_promotions:
        lines.append(f"   - Applied promotions for recommendation: {', '.join(rec.applied_promotions)}.")
    if rec.total_discount_amount == receipt.total_discount_amount and rec.recommended_channel_id != receipt.primary_channel_id:
        lines.append("   - Tie-breaker applied: identical discount found, selected earliest appearing channel in input list.")
    elif rec.total_discount_amount > receipt.total_discount_amount:
        lines.append(f"   - Recommendation saves additional {rec.total_discount_amount - receipt.total_discount_amount} cents over user choice.")

    lines.append("6. Assemble final JSON response.")
    lines.append("</think>")
    return "\n".join(lines)


def _sample_extreme_order_amount() -> int:
    """Generate high-variance, non-round order amounts across 4 distinct price tiers."""
    tier = random.choices(["micro", "mid_odd", "large_irregular", "boundary"], weights=[15, 45, 25, 15], k=1)[0]
    if tier == "micro":
        # 1 to 999 cents (0.01 to 9.99 Yuan)
        return random.choice([1, 2, 7, 13, 29, 49, 99, 101, 250, 499, 888, 999])
    elif tier == "mid_odd":
        # 1,000 to 99,999 cents (10.00 to 999.99 Yuan) with arbitrary primes / non-round cents
        base = random.randint(10, 999) * 100
        offset = random.choice([1, 3, 7, 11, 13, 23, 37, 49, 53, 77, 89, 99])
        return base + offset
    elif tier == "large_irregular":
        # 100,000 to 5,000,000 cents (1,000.00 to 50,000.00 Yuan)
        base = random.randint(1000, 50000) * 100
        offset = random.randint(1, 99)
        return base + offset
    else:  # boundary
        # Specific boundary cents
        return random.choice([9999, 10000, 10001, 49999, 50000, 50001, 99999, 100000, 100001, 999999])


def sample_single_request(bucket: str, order_idx: int) -> PaymentRequest:
    """Generate a single stochastic PaymentRequest matching specific bucket requirements."""
    order_id = f"ORD_{order_idx:07d}"

    # 4 to 8 channels per request (high density multi-channel scenario)
    num_channels = random.randint(4, 8)
    channel_ids = _generate_random_channel_ids(num_channels)

    order_amount = _sample_extreme_order_amount()

    promotions: list[Promotion] = []
    p_id_counter = 1

    # Default balance generation across channels: high variance
    channel_balances: dict[str, int] = {}
    for cid in channel_ids:
        # Some accounts have 0 balance, some tiny, some rich
        b_type = random.choices(["empty", "low", "medium", "rich"], weights=[10, 25, 45, 20], k=1)[0]
        if b_type == "empty":
            channel_balances[cid] = 0
        elif b_type == "low":
            channel_balances[cid] = random.randint(1, max(2, order_amount // 4))
        elif b_type == "medium":
            channel_balances[cid] = random.randint(order_amount // 2, order_amount * 2)
        else:
            channel_balances[cid] = random.randint(order_amount * 2, order_amount * 10 + 50000)

    selected_channel = random.choice(channel_ids)

    if bucket == "SPLIT_PAYMENT":
        # Primary channel has non-zero balance but strictly less than order amount
        # E.g. balance is order_amount - 1 (requires exactly 1 cent remainder) or small fraction
        if random.random() < 0.3:
            channel_balances[selected_channel] = max(1, order_amount - 1)  # 1 cent gap!
        else:
            channel_balances[selected_channel] = random.randint(1, max(2, order_amount // 3))

        # Intentionally place 1 or 2 other channels with insufficient balance before the paying secondary channel
        other_channels = [c for c in channel_ids if c != selected_channel]
        for c in other_channels[:-1]:
            channel_balances[c] = random.randint(0, max(1, order_amount // 5))
        # Ensure the last other channel has sufficient balance
        channel_balances[other_channels[-1]] = order_amount * 3

        # Add 2 to 3 stackable promotions
        for _ in range(random.randint(2, 3)):
            p_thresh = random.randint(0, order_amount // 2)
            promotions.append(
                Promotion(
                    promotion_id=f"PROMO_STACK_{p_id_counter}",
                    promotion_type=PromotionType.FIXED_DISCOUNT,
                    threshold=p_thresh,
                    discount_amount=max(1, int(order_amount * random.uniform(0.02, 0.08))),
                    is_exclusive=False,
                )
            )
            p_id_counter += 1

    elif bucket == "RECOMMEND_SWITCH":
        # Another channel gets a high exclusive or channel discount
        candidate_rec = random.choice([c for c in channel_ids if c != selected_channel])
        channel_balances[candidate_rec] = order_amount * 3
        # Selected channel discount is small
        promotions.append(
            Promotion(
                promotion_id=f"PROMO_SMALL_{p_id_counter}",
                promotion_type=PromotionType.FIXED_DISCOUNT,
                threshold=order_amount // 4,
                discount_amount=max(1, int(order_amount * 0.03)),
                is_exclusive=False,
            )
        )
        p_id_counter += 1
        # Recommendation channel gets fat discount
        promotions.append(
            Promotion(
                promotion_id=f"PROMO_FAT_CHANNEL_{p_id_counter}",
                promotion_type=PromotionType.CHANNEL_DISCOUNT,
                threshold=order_amount // 3,
                discount_amount=max(2, int(order_amount * random.uniform(0.15, 0.35))),
                applicable_channel_id=candidate_rec,
                is_exclusive=random.choice([True, False]),
            )
        )
        p_id_counter += 1
        # Add 2 more noise promotions for other channels
        for extra_c in channel_ids:
            if extra_c not in (selected_channel, candidate_rec) and random.random() < 0.5:
                promotions.append(
                    Promotion(
                        promotion_id=f"PROMO_EXTRA_{p_id_counter}",
                        promotion_type=PromotionType.CHANNEL_DISCOUNT,
                        threshold=order_amount // 2,
                        discount_amount=max(1, int(order_amount * 0.05)),
                        applicable_channel_id=extra_c,
                    )
                )
                p_id_counter += 1

    elif bucket == "EXCLUSIVE_VS_STACK":
        # Multi-coupon battle: 2 Exclusive vs 2-3 Stackable
        # With discount differences down to 1-5 cents
        stack_sum = 0
        for _ in range(random.randint(2, 3)):
            disc = max(10, int(order_amount * random.uniform(0.04, 0.09)))
            promotions.append(
                Promotion(
                    promotion_id=f"PROMO_STACK_{p_id_counter}",
                    promotion_type=PromotionType.FIXED_DISCOUNT,
                    threshold=order_amount // 3,
                    discount_amount=disc,
                    is_exclusive=False,
                )
            )
            stack_sum += disc
            p_id_counter += 1

        # Exclusive 1: close to stack_sum (within 1-10 cents)
        diff = random.choice([-5, -1, 1, 5, 50, -50])
        ex_disc_1 = max(1, stack_sum + diff)
        promotions.append(
            Promotion(
                promotion_id=f"PROMO_EX_A_{p_id_counter}",
                promotion_type=PromotionType.FIXED_DISCOUNT,
                threshold=order_amount // 3,
                discount_amount=ex_disc_1,
                is_exclusive=True,
            )
        )
        p_id_counter += 1

        # Exclusive 2: another close competitor
        ex_disc_2 = max(1, stack_sum + random.choice([-2, 2, 10]))
        promotions.append(
            Promotion(
                promotion_id=f"PROMO_EX_B_{p_id_counter}",
                promotion_type=PromotionType.FIXED_DISCOUNT,
                threshold=order_amount // 2,
                discount_amount=ex_disc_2,
                is_exclusive=True,
            )
        )
        p_id_counter += 1
        channel_balances[selected_channel] = order_amount * 3

    elif bucket == "ROUND_HALF_UP_BOUNDARY":
        # Fractional cents percentage discount with prime percentages (3%, 7%, 13%, 17%, 29%, 33%, 67%)
        pct = random.choice([3, 7, 13, 17, 29, 33, 67])
        promotions.append(
            Promotion(
                promotion_id=f"PROMO_PCT_{p_id_counter}",
                promotion_type=PromotionType.PERCENTAGE_DISCOUNT,
                threshold=0,
                discount_percent=pct,
                is_exclusive=False,
            )
        )
        p_id_counter += 1
        # Add a fixed stackable coupon
        promotions.append(
            Promotion(
                promotion_id=f"PROMO_STACK_CENTS_{p_id_counter}",
                promotion_type=PromotionType.FIXED_DISCOUNT,
                threshold=order_amount // 4,
                discount_amount=random.choice([33, 77, 125, 299]),
                is_exclusive=False,
            )
        )
        p_id_counter += 1
        channel_balances[selected_channel] = order_amount * 3

    elif bucket == "TIE_BREAKER":
        # 3 to 5 channels yield the EXACT same discount and have ample balance
        # Tests strict tie-breaking across multi-channel index ordering
        shared_discount = max(10, int(order_amount * 0.10))
        promotions.append(
            Promotion(
                promotion_id=f"PROMO_UNIVERSAL_EQUAL_{p_id_counter}",
                promotion_type=PromotionType.FIXED_DISCOUNT,
                threshold=order_amount // 4,
                discount_amount=shared_discount,
                is_exclusive=False,
            )
        )
        p_id_counter += 1
        for cid in channel_ids:
            channel_balances[cid] = order_amount * 4
        # User selected a channel deep in the list (e.g. index 3 or last)
        selected_channel = channel_ids[-1]

    elif bucket == "INSUFFICIENT_FUNDS":
        # Total combined balance of all channels is strictly less than order amount
        # E.g. sum(balances) = order_amount - 1 cent
        max_total = max(1, order_amount - random.randint(1, 100))
        per_channel = max_total // len(channel_ids)
        for cid in channel_ids:
            channel_balances[cid] = per_channel

    elif bucket == "CHANNEL_NOT_FOUND":
        selected_channel = "BANK_NONEXISTENT_GHOST_CHANNEL_9999"

    else:  # SINGLE_CHANNEL_NORMAL
        channel_balances[selected_channel] = order_amount * 3
        promotions.append(
            Promotion(
                promotion_id=f"PROMO_FIXED_{p_id_counter}",
                promotion_type=PromotionType.FIXED_DISCOUNT,
                threshold=order_amount // 2,
                discount_amount=max(1, int(order_amount * 0.12)),
            )
        )
        p_id_counter += 1
        # Add 1 channel-specific discount
        promotions.append(
            Promotion(
                promotion_id=f"PROMO_CH_{p_id_counter}",
                promotion_type=PromotionType.CHANNEL_DISCOUNT,
                threshold=order_amount // 3,
                discount_amount=max(1, int(order_amount * 0.05)),
                applicable_channel_id=selected_channel,
            )
        )

    available_channels = [
        PaymentChannel(channel_id=name, balance=channel_balances.get(name, 10000))
        for name in channel_ids
    ]

    return PaymentRequest(
        order=OrderInfo(order_id=order_id, order_amount=order_amount),
        available_channels=available_channels,
        promotions=promotions,
        selected_channel_id=selected_channel,
    )


def generate_dataset(
    target_count: int,
    start_id: int,
    is_sft: bool = False,
    isolated_hashes: set[str] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Synthesize dataset records and guarantee zero cross-set leakage via hash tracking."""
    seen_hashes: set[str] = set(isolated_hashes or [])
    current_hashes: set[str] = set()
    generated_records: list[dict[str, Any]] = []

    bucket_choices = [
        "SPLIT_PAYMENT",
        "RECOMMEND_SWITCH",
        "EXCLUSIVE_VS_STACK",
        "ROUND_HALF_UP_BOUNDARY",
        "TIE_BREAKER",
        "SINGLE_CHANNEL_NORMAL",
        "INSUFFICIENT_FUNDS",
        "CHANNEL_NOT_FOUND",
    ]
    bucket_weights = [25, 20, 15, 15, 10, 10, 3, 2]

    current_idx = start_id
    while len(generated_records) < target_count:
        bucket = random.choices(bucket_choices, weights=bucket_weights, k=1)[0]
        req = sample_single_request(bucket, current_idx)
        req_dict = req.to_dict()

        h = _request_fingerprint(req_dict)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        current_hashes.add(h)

        res = PaymentCalculationService.calculate(req)
        res_dict = res.to_dict()

        if is_sft:
            cot = generate_cot_reasoning(req, res)
            solution_text = f"{cot}\n{json.dumps(res_dict, indent=2, ensure_ascii=False)}"
            record = {
                "case_id": f"SFT_{len(generated_records)+1:06d}",
                "bucket": bucket,
                "input": req_dict,
                "ground_truth": res_dict,
                "solution": solution_text,
                "prompt": (
                    "You are an authoritative Java ling-tiny-fine-tuning microservice engine. "
                    "Calculate composite ling-tiny-fine-tuning deduction and optimal recommendation.\n\n"
                    f"User Input:\n{json.dumps(req_dict, indent=2, ensure_ascii=False)}"
                ),
            }
        else:
            record = {
                "case_id": f"CASE_{len(generated_records)+1:06d}",
                "bucket": bucket,
                "input": req_dict,
                "ground_truth": res_dict,
            }

        generated_records.append(record)
        current_idx += 1

    return generated_records, current_hashes


def main():
    parser = argparse.ArgumentParser(description="Generate high-difficulty synthetic datasets for composite ling-tiny-fine-tuning.")
    parser.add_argument("--eval-count", type=int, default=500, help="Number of eval test cases")
    parser.add_argument("--sft-count", type=int, default=1500, help="Number of SFT training samples")
    parser.add_argument("--rl-count", type=int, default=10000, help="Number of RL prompt samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="data", help="Output data directory")

    args = parser.parse_args()
    random.seed(args.seed)

    recipe_root = Path(__file__).resolve().parent
    out_dir = recipe_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting high-difficulty dataset synthesis with seed={args.seed}...")

    # 1. Generate Eval Testset (500)
    print(f"[1/3] Generating {args.eval_count} isolated high-difficulty evaluation samples...")
    eval_records, eval_hashes = generate_dataset(args.eval_count, start_id=1, is_sft=False)
    eval_path = out_dir / "eval_testset_500.jsonl"
    with open(eval_path, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"      Saved {len(eval_records)} samples to {eval_path}")

    # 2. Generate SFT Training Set (1,500) with synthetic CoT
    print(f"[2/3] Generating {args.sft_count} SFT cold-start samples with <think> reasoning...")
    sft_records, sft_hashes = generate_dataset(
        args.sft_count, start_id=10000, is_sft=True, isolated_hashes=eval_hashes
    )
    sft_path = out_dir / "sft_payment_train.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for r in sft_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"      Saved {len(sft_records)} samples to {sft_path}")

    # 3. Generate RL Prompts Set (10,000)
    combined_isolated = eval_hashes | sft_hashes
    print(f"[3/3] Generating {args.rl_count} RL exploration samples (zero overlap with eval)...")
    rl_records, _ = generate_dataset(
        args.rl_count, start_id=100000, is_sft=False, isolated_hashes=combined_isolated
    )
    rl_path = out_dir / "rl_payment_prompts.jsonl"
    with open(rl_path, "w", encoding="utf-8") as f:
        for r in rl_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"      Saved {len(rl_records)} samples to {rl_path}")

    print("\n[✔] All high-difficulty datasets synthesized successfully!")
    print(f"    Eval Set: {eval_path} ({eval_path.stat().st_size / 1024:.1f} KB)")
    print(f"    SFT Set:  {sft_path} ({sft_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"    RL Set:   {rl_path} ({rl_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
