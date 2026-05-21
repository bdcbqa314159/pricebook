"""Tests for perpetual and step-up bonds."""

import pytest
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.fixed_income.perpetual import (
    PerpetualBond, StepUpBond, PerpetualPricingResult, price_perpetual,
)

REF = date(2024, 1, 15)


@pytest.fixture
def flat_curve():
    return DiscountCurve.flat(REF, 0.04)


class TestPerpetual:
    def test_basic_pricing(self, flat_curve):
        perp = PerpetualBond(coupon=0.05)
        r = price_perpetual(perp, flat_curve)
        assert r.clean_price > 100  # coupon > rate → premium

    def test_low_coupon_discount(self, flat_curve):
        perp = PerpetualBond(coupon=0.03)
        r = price_perpetual(perp, flat_curve)
        assert r.clean_price < 100

    def test_yield_current(self, flat_curve):
        perp = PerpetualBond(coupon=0.05)
        r = price_perpetual(perp, flat_curve)
        assert abs(r.yield_current - 0.05 * 100 / r.clean_price) < 0.001

    def test_callable_lower_price(self, flat_curve):
        """Callable perpetual should price lower than non-callable (issuer call option)."""
        plain = PerpetualBond(coupon=0.06)
        callable_ = PerpetualBond(coupon=0.06, first_call_years=5.0)
        p_plain = price_perpetual(plain, flat_curve)
        p_call = price_perpetual(callable_, flat_curve)
        assert p_call.clean_price <= p_plain.clean_price

    def test_call_value_positive(self, flat_curve):
        perp = PerpetualBond(coupon=0.06, first_call_years=5.0)
        r = price_perpetual(perp, flat_curve)
        assert r.call_value >= 0

    def test_step_up_increases_price(self, flat_curve):
        """Step-up coupon should increase not-called extension value."""
        no_step = PerpetualBond(coupon=0.05, first_call_years=5.0, step_up_bp=0)
        with_step = PerpetualBond(coupon=0.05, first_call_years=5.0, step_up_bp=100)
        r1 = price_perpetual(no_step, flat_curve, call_probability=0.5)
        r2 = price_perpetual(with_step, flat_curve, call_probability=0.5)
        assert r2.clean_price > r1.clean_price

    def test_credit_spread_lowers_price(self, flat_curve):
        perp = PerpetualBond(coupon=0.05)
        r1 = price_perpetual(perp, flat_curve, credit_spread=0.0)
        r2 = price_perpetual(perp, flat_curve, credit_spread=0.02)
        assert r2.clean_price < r1.clean_price

    def test_to_dict(self, flat_curve):
        perp = PerpetualBond(coupon=0.05)
        r = price_perpetual(perp, flat_curve)
        assert "clean_price" in r.to_dict()


class TestStepUpBond:
    def test_basic(self, flat_curve):
        bond = StepUpBond(REF, 10.0, 0.04, [(5, 0.05)])
        p = bond.price(flat_curve)
        assert p > 0

    def test_coupon_at(self):
        bond = StepUpBond(REF, 10.0, 0.04, [(5, 0.06), (8, 0.08)])
        assert bond.coupon_at(3.0) == 0.04
        assert bond.coupon_at(5.0) == 0.06
        assert bond.coupon_at(9.0) == 0.08

    def test_step_up_higher_price(self, flat_curve):
        """Step-up coupon → higher price than flat coupon."""
        flat = StepUpBond(REF, 10.0, 0.04, [])
        step = StepUpBond(REF, 10.0, 0.04, [(5, 0.06)])
        assert step.price(flat_curve) > flat.price(flat_curve)

    def test_to_dict(self):
        bond = StepUpBond(REF, 10.0, 0.04, [(5, 0.06)])
        d = bond.to_dict()
        assert d["initial_coupon"] == 0.04
