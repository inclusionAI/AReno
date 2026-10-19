"""Core Payment Calculation and Recommendation Service (Authoritative Oracle)."""

from __future__ import annotations

from .models import (
    CashierReceipt,
    OrderInfo,
    PaymentChannel,
    PaymentRequest,
    PaymentResponse,
    PaymentStatus,
    Promotion,
    RecommendedOption,
)
from .promotion_engine import PromotionEngine


class PaymentCalculationService:
    """Authoritative reference implementation of composite ling-tiny-fine-tuning and cashier recommendation."""

    @classmethod
    def calculate(cls, request: PaymentRequest) -> PaymentResponse:
        order = request.order
        channels = request.available_channels
        promotions = request.promotions
        selected_id = request.selected_channel_id

        channel_map = {c.channel_id: c for c in channels}
        if selected_id not in channel_map:
            return PaymentResponse(
                order_id=order.order_id,
                order_amount=order.order_amount,
                status=PaymentStatus.CHANNEL_NOT_FOUND,
                error_message=f"Selected channel {selected_id} is not in available channels",
            )

        # 1. Compute cashier receipt for user selected channel
        receipt = cls._calculate_plan_for_channel(
            order=order,
            primary_channel_id=selected_id,
            channels=channels,
            promotions=promotions,
        )

        if receipt is None:
            return PaymentResponse(
                order_id=order.order_id,
                order_amount=order.order_amount,
                status=PaymentStatus.INSUFFICIENT_FUNDS,
                error_message="Selected channel and available secondary channels cannot cover payable amount",
            )

        # 2. Iterate through all available channels to find optimal recommendation
        best_candidate: tuple[int, int, RecommendedOption] | None = None
        for idx, ch in enumerate(channels):
            candidate_receipt = cls._calculate_plan_for_channel(
                order=order,
                primary_channel_id=ch.channel_id,
                channels=channels,
                promotions=promotions,
            )
            if candidate_receipt is None:
                continue

            # Tie-breaking key: 1) higher total_discount_amount, 2) smaller original input index
            score = (candidate_receipt.total_discount_amount, -idx)
            rec_option = RecommendedOption(
                recommended_channel_id=candidate_receipt.primary_channel_id,
                primary_deduct_amount=candidate_receipt.primary_deduct_amount,
                secondary_channel_id=candidate_receipt.secondary_channel_id,
                secondary_deduct_amount=candidate_receipt.secondary_deduct_amount,
                total_discount_amount=candidate_receipt.total_discount_amount,
                payable_amount=candidate_receipt.payable_amount,
                applied_promotions=candidate_receipt.applied_promotions,
            )

            if best_candidate is None or score > (best_candidate[0], best_candidate[1]):
                best_candidate = (candidate_receipt.total_discount_amount, -idx, rec_option)

        recommended = best_candidate[2] if best_candidate is not None else None

        return PaymentResponse(
            order_id=order.order_id,
            order_amount=order.order_amount,
            status=PaymentStatus.SUCCESS,
            cashier_receipt=receipt,
            recommended_option=recommended,
        )

    @classmethod
    def _calculate_plan_for_channel(
        cls,
        order: OrderInfo,
        primary_channel_id: str,
        channels: list[PaymentChannel],
        promotions: list[Promotion],
    ) -> CashierReceipt | None:
        """Compute discount and 2-leg split deduction for a given primary channel."""
        total_discount, applied_promos = PromotionEngine.calculate_discount_for_channel(
            order_amount=order.order_amount,
            primary_channel_id=primary_channel_id,
            promotions=promotions,
        )

        payable_amount = max(0, order.order_amount - total_discount)
        primary_channel = next(c for c in channels if c.channel_id == primary_channel_id)

        # Case A: Primary channel covers whole payable amount
        if primary_channel.balance >= payable_amount:
            return CashierReceipt(
                primary_channel_id=primary_channel_id,
                primary_deduct_amount=payable_amount,
                secondary_channel_id=None,
                secondary_deduct_amount=0,
                total_discount_amount=total_discount,
                payable_amount=payable_amount,
                applied_promotions=applied_promos,
            )

        # Case B: Primary channel pays full balance, secondary channel covers remaining needed
        primary_deduct = primary_channel.balance
        needed_remainder = payable_amount - primary_deduct

        # Search for first eligible secondary channel in input order
        secondary_chosen: PaymentChannel | None = None
        for ch in channels:
            if ch.channel_id == primary_channel_id:
                continue
            if ch.balance >= needed_remainder:
                secondary_chosen = ch
                break

        if secondary_chosen is None:
            # Cannot cover remaining ling-tiny-fine-tuning with a single secondary channel
            return None

        return CashierReceipt(
            primary_channel_id=primary_channel_id,
            primary_deduct_amount=primary_deduct,
            secondary_channel_id=secondary_chosen.channel_id,
            secondary_deduct_amount=needed_remainder,
            total_discount_amount=total_discount,
            payable_amount=payable_amount,
            applied_promotions=applied_promos,
        )
