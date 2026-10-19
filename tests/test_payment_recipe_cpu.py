import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recipes.payment.payment_service.models import (
    OrderInfo,
    PaymentChannel,
    PaymentRequest,
    PaymentStatus,
    Promotion,
    PromotionType,
)
from recipes.payment.payment_service.service import PaymentCalculationService
from recipes.payment.payment_reward import verify_payment, extract_json_object


class TestPaymentSystemAndVerifier(unittest.TestCase):
    def test_system_calculation_and_verifier_perfect_score(self):
        """Verify system calculates 2-leg split + recommendation correctly and verifier awards 1.00."""
        req = PaymentRequest(
            order=OrderInfo(order_id="ORD_TEST_01", order_amount=10000),  # 100.00
            available_channels=[
                PaymentChannel(channel_id="ALIPAY", balance=4000),
                PaymentChannel(channel_id="WECHAT", balance=8000),
                PaymentChannel(channel_id="BANK_CARD", balance=20000),
            ],
            promotions=[
                Promotion(
                    promotion_id="PROMO_FIXED_10",
                    promotion_type=PromotionType.FIXED_DISCOUNT,
                    threshold=5000,
                    discount_amount=1000,
                ),
                Promotion(
                    promotion_id="PROMO_BANK_SPECIAL",
                    promotion_type=PromotionType.CHANNEL_DISCOUNT,
                    threshold=6000,
                    discount_amount=1500,
                    applicable_channel_id="BANK_CARD",
                ),
            ],
            selected_channel_id="ALIPAY",
        )
        # 1. Run system calculation
        res = PaymentCalculationService.calculate(req)
        self.assertEqual(res.status, PaymentStatus.SUCCESS)

        req_dict = req.to_dict()
        gt_dict = res.to_dict()

        # 2. Simulate model output with <think> reasoning tags and markdown fence
        completion_text = f"""
        <think>
        Let's calculate step by step:
        Order amount = 10000 cents.
        Selected channel is ALIPAY.
        ALIPAY qualifies for PROMO_FIXED_10 (1000 cents off).
        Payable = 9000 cents.
        ALIPAY balance = 4000 cents, so it pays 4000.
        Remaining needed = 5000 cents.
        WECHAT balance is 8000 >= 5000, so WECHAT pays 5000.
        For recommendation: BANK_CARD qualifies for both PROMO_FIXED_10 (1000) and PROMO_BANK_SPECIAL (1500), total discount = 2500 cents.
        Payable = 7500 cents. BANK_CARD balance 20000 >= 7500, pays all.
        </think>
        ```json
        {json.dumps(gt_dict, indent=2)}
        ```
        """

        # 3. Verifier evaluate
        score, details = verify_payment(req_dict, completion_text, gt_dict)
        self.assertEqual(score, 1.0, f"Expected 1.0 but got {score}: {details}")
        self.assertEqual(details["dim1_json_syntax"], 0.20)
        self.assertEqual(details["dim2_conservation"], 0.20)
        self.assertEqual(details["dim3_promotion_rules"], 0.20)
        self.assertEqual(details["dim4_cashier_match"], 0.20)
        self.assertEqual(details["dim5_recommendation_match"], 0.20)

    def test_tie_breaker_in_recommendation_and_verifier(self):
        """Verify tie-breaker preserves original channel input order and verifier checks it."""
        req = PaymentRequest(
            order=OrderInfo(order_id="ORD_TEST_02", order_amount=5000),
            available_channels=[
                PaymentChannel(channel_id="ALIPAY", balance=10000),
                PaymentChannel(channel_id="WECHAT", balance=10000),
            ],
            promotions=[
                Promotion(
                    promotion_id="PROMO_FIXED_5",
                    promotion_type=PromotionType.FIXED_DISCOUNT,
                    threshold=3000,
                    discount_amount=500,
                )
            ],
            selected_channel_id="WECHAT",
        )
        res = PaymentCalculationService.calculate(req)
        self.assertEqual(res.status, PaymentStatus.SUCCESS)
        self.assertEqual(res.cashier_receipt.primary_channel_id, "WECHAT")
        self.assertEqual(res.recommended_option.recommended_channel_id, "ALIPAY")

        score, details = verify_payment(req.to_dict(), res.to_dict(), res.to_dict())
        self.assertEqual(score, 1.0)

    def test_verifier_catches_conservation_violation(self):
        """Verifier must dock points if deduction does not equal payable amount."""
        req = PaymentRequest(
            order=OrderInfo(order_id="ORD_TEST_03", order_amount=10000),
            available_channels=[PaymentChannel(channel_id="ALIPAY", balance=15000)],
            promotions=[],
            selected_channel_id="ALIPAY",
        )
        res = PaymentCalculationService.calculate(req)
        gt_dict = res.to_dict()

        # Tamper completion: primary_deduct 9000 instead of 10000 (leaking 1000 cents)
        bad_dict = json.loads(json.dumps(gt_dict))
        bad_dict["cashier_receipt"]["primary_deduct_amount"] = 9000

        score, details = verify_payment(req.to_dict(), bad_dict, gt_dict)
        self.assertLess(score, 1.0)
        self.assertEqual(details["dim2_conservation"], 0.0)  # docked!
        self.assertEqual(details["dim4_cashier_match"], 0.0)  # docked!

    def test_verifier_catches_wrong_recommendation(self):
        """Verifier must dock points if recommendation missed tie-breaker or picked sub-optimal."""
        req = PaymentRequest(
            order=OrderInfo(order_id="ORD_TEST_04", order_amount=5000),
            available_channels=[
                PaymentChannel(channel_id="ALIPAY", balance=10000),
                PaymentChannel(channel_id="WECHAT", balance=10000),
            ],
            promotions=[],
            selected_channel_id="ALIPAY",
        )
        res = PaymentCalculationService.calculate(req)
        gt_dict = res.to_dict()

        # Tamper: set recommended channel to WECHAT instead of ALIPAY (tie-breaker failure)
        bad_dict = json.loads(json.dumps(gt_dict))
        bad_dict["recommended_option"]["recommended_channel_id"] = "WECHAT"

        score, details = verify_payment(req.to_dict(), bad_dict, gt_dict)
        self.assertLess(score, 1.0)
        self.assertEqual(details["dim5_recommendation_match"], 0.0)


if __name__ == "__main__":
    unittest.main()
