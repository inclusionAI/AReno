"""Promotion Engine for evaluating discounts, exclusivity and channel eligibility."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from .models import Promotion, PromotionType


def round_half_up(value: Decimal) -> int:
    """Standard financial half-up rounding to integer cents."""
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


class PromotionEngine:
    """Calculates optimal discount amount and applied promotion set for a given channel."""

    @classmethod
    def calculate_discount_for_channel(
        cls,
        order_amount: int,
        primary_channel_id: str,
        promotions: list[Promotion],
    ) -> tuple[int, list[str]]:
        """Calculate maximum discount achievable under mutual exclusivity and stacking rules.

        Returns:
            (total_discount_amount, list_of_applied_promotion_ids)
        """
        # 1. Evaluate single exclusive promotions (is_exclusive=True)
        best_exclusive_discount = 0
        best_exclusive_promo_id: str | None = None

        for p in promotions:
            if not p.is_exclusive:
                continue
            amt = cls._calculate_single_promotion(order_amount, primary_channel_id, p)
            if amt > best_exclusive_discount:
                best_exclusive_discount = amt
                best_exclusive_promo_id = p.promotion_id

        # 2. Evaluate stackable promotions (is_exclusive=False)
        # Business rule: at most 1 best general promotion (fixed or percentage) + all applicable channel promotions
        best_general_discount = 0
        best_general_promo_id: str | None = None

        for p in promotions:
            if p.is_exclusive:
                continue
            if p.promotion_type in (PromotionType.FIXED_DISCOUNT, PromotionType.PERCENTAGE_DISCOUNT):
                amt = cls._calculate_single_promotion(order_amount, primary_channel_id, p)
                if amt > best_general_discount:
                    best_general_discount = amt
                    best_general_promo_id = p.promotion_id

        channel_discount_total = 0
        applied_channel_promos: list[str] = []
        for p in promotions:
            if p.is_exclusive:
                continue
            if p.promotion_type == PromotionType.CHANNEL_DISCOUNT:
                amt = cls._calculate_single_promotion(order_amount, primary_channel_id, p)
                if amt > 0:
                    channel_discount_total += amt
                    applied_channel_promos.append(p.promotion_id)

        stackable_total = best_general_discount + channel_discount_total
        stackable_promo_ids: list[str] = []
        if best_general_promo_id:
            stackable_promo_ids.append(best_general_promo_id)
        stackable_promo_ids.extend(applied_channel_promos)

        # 3. Decision: Pick higher discount between best exclusive vs stacked
        if best_exclusive_discount > stackable_total:
            final_discount = min(best_exclusive_discount, order_amount)
            return final_discount, [best_exclusive_promo_id] if best_exclusive_promo_id else []
        else:
            final_discount = min(stackable_total, order_amount)
            return final_discount, stackable_promo_ids

    @classmethod
    def _calculate_single_promotion(
        cls,
        order_amount: int,
        primary_channel_id: str,
        promo: Promotion,
    ) -> int:
        """Calculate discount amount in cents for a single promotion."""
        if order_amount < promo.threshold:
            return 0

        if promo.promotion_type == PromotionType.FIXED_DISCOUNT:
            return min(promo.discount_amount, order_amount)

        if promo.promotion_type == PromotionType.PERCENTAGE_DISCOUNT:
            # discount_percent: 1 to 99, e.g. 15 means 15% off
            discount_dec = (Decimal(order_amount) * Decimal(promo.discount_percent)) / Decimal(100)
            return min(round_half_up(discount_dec), order_amount)

        if promo.promotion_type == PromotionType.CHANNEL_DISCOUNT:
            if promo.applicable_channel_id == primary_channel_id:
                return min(promo.discount_amount, order_amount)
            return 0

        return 0
