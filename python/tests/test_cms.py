"""Tests for CMS swaps, caps, spread options, and range accruals."""

import math
import pytest
from datetime import date

from pricebook.cms import (
    cms_convexity_adjustment, CMSLeg,
    cms_cap, cms_spread_option, range_accrual,
)
from pricebook.black76 import OptionType
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency
from pricebook.swap import InterestRateSwap, SwapDirection


REF = date(2024, 1, 15)


def _curve(rate=0.05):
    return DiscountCurve.flat(REF, rate)


def _upward_curve():
    pillar_dates = [
        date(2025, 1, 15), date(2026, 1, 15), date(2029, 1, 15),
        date(2034, 1, 15), date(2054, 1, 15),
    ]
    rates = [0.03, 0.035, 0.04, 0.045, 0.05]
    dfs = [math.exp(-r * ((d - REF).days / 365.0)) for r, d in zip(rates, pillar_dates)]
    return DiscountCurve(REF, pillar_dates, dfs)


# ---- Convexity adjustment ----

class TestConvexityAdjustment:
    def test_positive_adjustment(self):
        """CMS convexity adjustment should be positive."""
        adj = cms_convexity_adjustment(
            forward_swap_rate=0.05,
            annuity=4.0,
            swap_tenor_years=10,
            vol=0.20,
            time_to_fixing=1.0,
        )
        assert adj > 0

    def test_zero_vol_zero_adjustment(self):
        adj = cms_convexity_adjustment(0.05, 4.0, 10, 0.0, 1.0)
        assert adj == 0.0

    def test_zero_time_zero_adjustment(self):
        adj = cms_convexity_adjustment(0.05, 4.0, 10, 0.20, 0.0)
        assert adj == 0.0

    def test_scales_with_vol(self):
        adj1 = cms_convexity_adjustment(0.05, 4.0, 10, 0.10, 1.0)
        adj2 = cms_convexity_adjustment(0.05, 4.0, 10, 0.20, 1.0)
        # adjustment ~ vol², so adj2 ≈ 4 × adj1
        assert adj2 == pytest.approx(4 * adj1, rel=0.01)

    def test_scales_with_time(self):
        adj1 = cms_convexity_adjustment(0.05, 4.0, 10, 0.20, 1.0)
        adj2 = cms_convexity_adjustment(0.05, 4.0, 10, 0.20, 2.0)
        assert adj2 == pytest.approx(2 * adj1, rel=0.01)


# ---- CMS Leg ----

class TestCMSLeg:
    def test_cms_rate_above_forward(self):
        """CMS rate should exceed forward swap rate (convexity premium)."""
        curve = _curve()
        leg = CMSLeg(REF, date(2029, 1, 15), cms_tenor=10, notional=10_000_000)
        cfs = leg.cashflows(curve, vol=0.20)
        assert len(cfs) > 0
        for cf in cfs:
            assert cf.cms_rate >= cf.forward_rate
            assert cf.convexity_adj >= 0

    def test_zero_vol_cms_equals_forward(self):
        """Without vol, CMS rate = forward swap rate."""
        curve = _curve()
        leg = CMSLeg(REF, date(2029, 1, 15), cms_tenor=10)
        cfs = leg.cashflows(curve, vol=0.0)
        for cf in cfs:
            assert cf.cms_rate == pytest.approx(cf.forward_rate)
            assert cf.convexity_adj == pytest.approx(0.0)

    def test_pv_positive(self):
        curve = _curve()
        leg = CMSLeg(REF, date(2029, 1, 15), cms_tenor=10, notional=10_000_000)
        pv = leg.pv(curve, vol=0.20)
        assert pv > 0

    def test_pv_with_spread(self):
        curve = _curve()
        leg_no_spread = CMSLeg(REF, date(2029, 1, 15), cms_tenor=10)
        leg_with_spread = CMSLeg(REF, date(2029, 1, 15), cms_tenor=10, spread=0.001)
        assert leg_with_spread.pv(curve) > leg_no_spread.pv(curve)


# ---- CMS cap/floor ----

class TestCMSCap:
    def test_cap_positive(self):
        curve = _curve()
        pv = cms_cap(REF, date(2029, 1, 15), 0.04, 10, curve, 0.20)
        assert pv > 0

    def test_deep_otm_cap_small(self):
        """Deep OTM cap (high strike) → small PV."""
        curve = _curve()
        itm = cms_cap(REF, date(2029, 1, 15), 0.03, 10, curve, 0.20)
        otm = cms_cap(REF, date(2029, 1, 15), 0.10, 10, curve, 0.20)
        assert otm < itm

    def test_floor_positive(self):
        curve = _curve()
        pv = cms_cap(REF, date(2029, 1, 15), 0.06, 10, curve, 0.20,
                     option_type=OptionType.PUT)
        assert pv > 0

    def test_higher_vol_higher_cap(self):
        curve = _curve()
        low = cms_cap(REF, date(2029, 1, 15), 0.05, 10, curve, 0.10)
        high = cms_cap(REF, date(2029, 1, 15), 0.05, 10, curve, 0.30)
        assert high > low


# ---- CMS spread option ----

class TestCMSSpread:
    def test_positive_spread_call(self):
        """Call on 10Y-2Y spread should be positive."""
        curve = _upward_curve()
        pv = cms_spread_option(
            REF, date(2029, 1, 15), 10, 2, 0.001,
            curve, 0.20, 0.20, 0.9,
        )
        assert pv > 0

    def test_higher_strike_lower_call(self):
        curve = _upward_curve()
        low_k = cms_spread_option(REF, date(2029, 1, 15), 10, 2, 0.001, curve, 0.20, 0.20)
        high_k = cms_spread_option(REF, date(2029, 1, 15), 10, 2, 0.02, curve, 0.20, 0.20)
        assert high_k < low_k

    def test_lower_correlation_higher_vol(self):
        """Lower correlation → higher spread vol → higher option value."""
        curve = _upward_curve()
        high_corr = cms_spread_option(REF, date(2029, 1, 15), 10, 2, 0.005,
                                      curve, 0.20, 0.20, 0.95)
        low_corr = cms_spread_option(REF, date(2029, 1, 15), 10, 2, 0.005,
                                     curve, 0.20, 0.20, 0.50)
        assert low_corr > high_corr


# ---- Range accrual ----

class TestRangeAccrual:
    def test_full_range_recovers_coupon(self):
        """Range [0, ∞) → always accruing → PV = full coupon stream."""
        curve = _curve()
        full = range_accrual(
            REF, date(2029, 1, 15), 0.05, 0.0, 1e6,
            curve, 0.20,
        )
        # Compare to a simple fixed coupon stream
        from pricebook.fixed_leg import FixedLeg
        fixed = FixedLeg(REF, date(2029, 1, 15), 0.05, Frequency.QUARTERLY,
                         notional=1_000_000)
        fixed_pv = fixed.pv(curve)
        assert full == pytest.approx(fixed_pv, rel=0.05)

    def test_impossible_range_zero(self):
        """Range [∞, ∞+1) → never accrues → PV = 0."""
        curve = _curve()
        pv = range_accrual(
            REF, date(2029, 1, 15), 0.05, 1e6, 2e6,
            curve, 0.20,
        )
        assert pv == pytest.approx(0.0, abs=1.0)

    def test_narrower_range_lower_pv(self):
        curve = _curve()
        wide = range_accrual(REF, date(2029, 1, 15), 0.05, 0.02, 0.08, curve, 0.20)
        narrow = range_accrual(REF, date(2029, 1, 15), 0.05, 0.04, 0.06, curve, 0.20)
        assert narrow < wide

    def test_higher_vol_lower_pv(self):
        """Higher vol → more likely to breach range → lower PV."""
        curve = _curve()
        low_v = range_accrual(REF, date(2029, 1, 15), 0.05, 0.03, 0.07, curve, 0.10)
        high_v = range_accrual(REF, date(2029, 1, 15), 0.05, 0.03, 0.07, curve, 0.40)
        assert high_v < low_v
