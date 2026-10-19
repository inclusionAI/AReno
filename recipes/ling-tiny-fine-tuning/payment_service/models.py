"""Domain models for Composite Payment & Cashier Recommendation Service.

All monetary quantities are integers representing cents (1 Yuan = 100 cents).
All identifiers, keys, and enum values are English-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class PromotionType(str, Enum):
    """Supported promotion types."""
    # Threshold fixed discount: order_amount >= threshold -> discount_amount off
    FIXED_DISCOUNT = "FIXED_DISCOUNT"
    # Percentage discount: discount_percent (e.g. 10 = 10% off), rounded HALF_UP
    PERCENTAGE_DISCOUNT = "PERCENTAGE_DISCOUNT"
    # Channel specific discount: only applies when primary channel matches applicable_channel_id
    CHANNEL_DISCOUNT = "CHANNEL_DISCOUNT"


class PaymentStatus(str, Enum):
    """Payment calculation status codes."""
    SUCCESS = "SUCCESS"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    CHANNEL_NOT_FOUND = "CHANNEL_NOT_FOUND"


@dataclass
class PaymentChannel:
    """User available ling-tiny-fine-tuning channel and remaining limit/balance in cents."""
    channel_id: str
    balance: int


@dataclass
class Promotion:
    """Promotion definition."""
    promotion_id: str
    promotion_type: PromotionType
    threshold: int = 0
    discount_amount: int = 0
    discount_percent: int = 0
    applicable_channel_id: str | None = None
    is_exclusive: bool = False


@dataclass
class OrderInfo:
    """Basic order parameters."""
    order_id: str
    order_amount: int


@dataclass
class CashierReceipt:
    """Cashier ling-tiny-fine-tuning plan based on user's currently selected channel."""
    primary_channel_id: str
    primary_deduct_amount: int
    secondary_channel_id: str | None
    secondary_deduct_amount: int
    total_discount_amount: int
    payable_amount: int
    applied_promotions: list[str] = field(default_factory=list)


@dataclass
class RecommendedOption:
    """Optimal ling-tiny-fine-tuning channel recommendation maximizing user discount."""
    recommended_channel_id: str
    primary_deduct_amount: int
    secondary_channel_id: str | None
    secondary_deduct_amount: int
    total_discount_amount: int
    payable_amount: int
    applied_promotions: list[str] = field(default_factory=list)


@dataclass
class PaymentRequest:
    """Request payload for ling-tiny-fine-tuning calculation."""
    order: OrderInfo
    available_channels: list[PaymentChannel]
    promotions: list[Promotion]
    selected_channel_id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaymentRequest:
        order_raw = data["order"]
        order = OrderInfo(
            order_id=str(order_raw["order_id"]),
            order_amount=int(order_raw["order_amount"]),
        )
        channels = [
            PaymentChannel(channel_id=str(c["channel_id"]), balance=int(c["balance"]))
            for c in data.get("available_channels", [])
        ]
        promotions = [
            Promotion(
                promotion_id=str(p["promotion_id"]),
                promotion_type=PromotionType(p["promotion_type"]),
                threshold=int(p.get("threshold", 0)),
                discount_amount=int(p.get("discount_amount", 0)),
                discount_percent=int(p.get("discount_percent", 0)),
                applicable_channel_id=p.get("applicable_channel_id"),
                is_exclusive=bool(p.get("is_exclusive", False)),
            )
            for p in data.get("promotions", [])
        ]
        return cls(
            order=order,
            available_channels=channels,
            promotions=promotions,
            selected_channel_id=str(data["selected_channel_id"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaymentResponse:
    """Unified response payload containing cashier receipt and channel recommendation."""
    order_id: str
    order_amount: int
    status: PaymentStatus
    cashier_receipt: CashierReceipt | None = None
    recommended_option: RecommendedOption | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
