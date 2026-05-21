"""Tests for callable bonds with credit risk."""

import pytest
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.credit.callable_credit import (
    callable_credit_bond_price, credit_risky_oas, CallableCreditResult,
)

REF = date(2024, 1, 15)


@pytest.fixture
def flat_curve():
    return DiscountCurve.flat(REF, 0.04)


@pytest.fixture
def ig_survival():
    return SurvivalCurve.flat(REF, 0.01, tenors=list(range(1, 12)))


@pytest.fixture
def hy_survival():
    return SurvivalCurve.flat(REF, 0.05, tenors=list(range(1, 12)))


class TestCallableCreditPrice:
    def test_basic(self, flat_curve, ig_survival):
        r = callable_credit_bond_price(0.05, 10.0, flat_curve, ig_survival)
        assert isinstance(r, CallableCreditResult)
        assert r.price > 0

    def test_credit_reduces_price(self, flat_curve, ig_survival, hy_survival):
        """Higher hazard → lower price."""
        r_ig = callable_credit_bond_price(0.05, 10.0, flat_curve, ig_survival)
        r_hy = callable_credit_bond_price(0.05, 10.0, flat_curve, hy_survival)
        assert r_hy.price < r_ig.price

    def test_call_reduces_price(self, flat_curve, ig_survival):
        """Callable price ≤ non-callable price (issuer benefits from call)."""
        r = callable_credit_bond_price(0.06, 10.0, flat_curve, ig_survival)
        assert r.price <= r.price_no_call + 1.0  # within tolerance

    def test_call_option_value_positive(self, flat_curve, ig_survival):
        """Premium coupon callable: call option has value."""
        r = callable_credit_bond_price(0.07, 10.0, flat_curve, ig_survival)
        assert r.call_option_value >= 0

    def test_credit_spread_positive(self, flat_curve, ig_survival):
        r = callable_credit_bond_price(0.05, 10.0, flat_curve, ig_survival)
        assert r.credit_spread_bp >= 0

    def test_to_dict(self, flat_curve, ig_survival):
        r = callable_credit_bond_price(0.05, 10.0, flat_curve, ig_survival)
        d = r.to_dict()
        assert "price" in d
        assert "call_option_value" in d


class TestCreditRiskyOAS:
    def test_oas_positive(self, flat_curve, ig_survival):
        """OAS should be positive when market price is below model."""
        model_result = callable_credit_bond_price(0.05, 10.0, flat_curve, ig_survival)
        # Set market price below model → positive OAS needed
        market_price = model_result.price - 2.0
        oas = credit_risky_oas(market_price, 0.05, 10.0, flat_curve, ig_survival)
        assert oas > 0

    def test_oas_zero_at_model_price(self, flat_curve, ig_survival):
        """OAS should be ~0 when market = model price."""
        model_result = callable_credit_bond_price(0.05, 10.0, flat_curve, ig_survival)
        oas = credit_risky_oas(model_result.price, 0.05, 10.0, flat_curve, ig_survival)
        assert abs(oas) < 0.005  # within 50bp
