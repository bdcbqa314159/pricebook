"""Tests for CoCo / AT1 bond pricing."""

import pytest
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.credit.coco import (
    CoCoBond, CoCoTriggerType, CoCoLossAbsorption,
    CoCoPricingResult, price_coco,
)


REF = date(2024, 1, 15)


@pytest.fixture
def flat_curve():
    return DiscountCurve.flat(REF, 0.04)


@pytest.fixture
def at1_bond():
    """Standard AT1 CoCo: 7% coupon, perpetual, 5.25% CET1 trigger, full write-down."""
    return CoCoBond(
        coupon=0.07, trigger_level=0.0525,
        trigger_type=CoCoTriggerType.CET1_RATIO,
        loss_absorption=CoCoLossAbsorption.FULL_WRITE_DOWN,
        maturity_years=None, first_call_years=5.0,
    )


@pytest.fixture
def t2_coco():
    """Tier 2 CoCo: 5% coupon, 10Y maturity, equity conversion."""
    return CoCoBond(
        coupon=0.05, trigger_level=0.07,
        loss_absorption=CoCoLossAbsorption.EQUITY_CONVERSION,
        maturity_years=10.0, first_call_years=5.0,
        conversion_price=10.0,
    )


class TestPricing:
    def test_basic_at1(self, flat_curve, at1_bond):
        result = price_coco(at1_bond, flat_curve, trigger_intensity=0.02)
        assert isinstance(result, CoCoPricingResult)
        assert 50 < result.clean_price < 120

    def test_zero_trigger_near_par(self, flat_curve, at1_bond):
        """With zero trigger risk, AT1 prices like a callable bond."""
        result = price_coco(at1_bond, flat_curve, trigger_intensity=0.0)
        assert result.clean_price > 100  # coupon > risk-free → premium

    def test_high_trigger_low_price(self, flat_curve, at1_bond):
        """High trigger intensity → low price."""
        r_low = price_coco(at1_bond, flat_curve, trigger_intensity=0.01)
        r_high = price_coco(at1_bond, flat_curve, trigger_intensity=0.10)
        assert r_high.clean_price < r_low.clean_price

    def test_trigger_loss_positive(self, flat_curve, at1_bond):
        result = price_coco(at1_bond, flat_curve, trigger_intensity=0.03)
        assert result.trigger_loss_pv > 0

    def test_coupon_pv_positive(self, flat_curve, at1_bond):
        result = price_coco(at1_bond, flat_curve, trigger_intensity=0.02)
        assert result.coupon_pv > 0

    def test_redemption_pv_positive(self, flat_curve, at1_bond):
        result = price_coco(at1_bond, flat_curve, trigger_intensity=0.02)
        assert result.redemption_pv > 0


class TestEquityConversion:
    def test_t2_conversion(self, flat_curve, t2_coco):
        result = price_coco(t2_coco, flat_curve, trigger_intensity=0.02)
        assert 60 < result.clean_price < 120

    def test_conversion_less_loss_than_writedown(self, flat_curve):
        """Equity conversion has less loss than full write-down."""
        wd = CoCoBond(coupon=0.06, trigger_level=0.05,
                       loss_absorption=CoCoLossAbsorption.FULL_WRITE_DOWN)
        ec = CoCoBond(coupon=0.06, trigger_level=0.05,
                       loss_absorption=CoCoLossAbsorption.EQUITY_CONVERSION)
        r_wd = price_coco(wd, flat_curve, trigger_intensity=0.03)
        r_ec = price_coco(ec, flat_curve, trigger_intensity=0.03)
        assert r_ec.clean_price > r_wd.clean_price


class TestAnalytics:
    def test_yield_to_call(self, flat_curve, at1_bond):
        result = price_coco(at1_bond, flat_curve, trigger_intensity=0.02)
        assert result.yield_to_call > 0

    def test_credit_spread(self, flat_curve, at1_bond):
        result = price_coco(at1_bond, flat_curve, trigger_intensity=0.03)
        assert result.credit_spread_bp > 0

    def test_trigger_prob(self, flat_curve, at1_bond):
        result = price_coco(at1_bond, flat_curve, trigger_intensity=0.02)
        assert 0 < result.trigger_prob_5y < 1
        # 2% intensity × 5Y ≈ 9.5% trigger prob
        assert abs(result.trigger_prob_5y - 0.095) < 0.01

    def test_coupon_skip(self, flat_curve, at1_bond):
        """Coupon skip probability reduces price."""
        r_no_skip = price_coco(at1_bond, flat_curve, 0.02, coupon_skip_prob=0.0)
        r_skip = price_coco(at1_bond, flat_curve, 0.02, coupon_skip_prob=0.10)
        assert r_skip.clean_price < r_no_skip.clean_price


class TestSerialization:
    def test_bond_to_dict(self, at1_bond):
        d = at1_bond.to_dict()
        assert d["coupon"] == 0.07
        assert d["loss_absorption"] == "full_write_down"

    def test_result_to_dict(self, flat_curve, at1_bond):
        r = price_coco(at1_bond, flat_curve, 0.02)
        d = r.to_dict()
        assert "clean_price" in d
        assert "trigger_prob_5y" in d
