"""Tests for risky bond, Z-spread, and asset swap spread."""

import pytest
import math
from datetime import date

from pricebook.risky_bond import RiskyBond, z_spread, asset_swap_spread
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)
END = date(2029, 1, 15)


class TestRiskyBond:
    def test_risky_less_than_risk_free(self):
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)
        rb = RiskyBond(REF, END, coupon_rate=0.05)
        risky = rb.dirty_price(disc, surv)
        riskfree = rb.risk_free_price(disc)
        assert risky < riskfree

    def test_risky_positive(self):
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)
        rb = RiskyBond(REF, END, coupon_rate=0.05)
        assert rb.dirty_price(disc, surv) > 0

    def test_zero_hazard_equals_risk_free(self):
        """No default risk → risky price = risk-free price."""
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.0001)  # near-zero hazard
        rb = RiskyBond(REF, END, coupon_rate=0.05)
        risky = rb.dirty_price(disc, surv)
        riskfree = rb.risk_free_price(disc)
        assert risky == pytest.approx(riskfree, rel=0.01)

    def test_higher_hazard_lower_price(self):
        disc = make_flat_curve(REF, 0.04)
        surv_low = make_flat_survival(REF, 0.01)
        surv_high = make_flat_survival(REF, 0.05)
        rb = RiskyBond(REF, END, coupon_rate=0.05)
        assert rb.dirty_price(disc, surv_high) < rb.dirty_price(disc, surv_low)

    def test_higher_recovery_higher_price(self):
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.03)
        rb_low = RiskyBond(REF, END, coupon_rate=0.05, recovery=0.2)
        rb_high = RiskyBond(REF, END, coupon_rate=0.05, recovery=0.6)
        assert rb_high.dirty_price(disc, surv) > rb_low.dirty_price(disc, surv)


class TestZSpread:
    def test_z_spread_positive_below_par(self):
        """Below-par price → positive Z-spread."""
        disc = make_flat_curve(REF, 0.04)
        rb = RiskyBond(REF, END, coupon_rate=0.05)
        riskfree_price = rb.risk_free_price(disc)
        z = z_spread(rb, riskfree_price - 5.0, disc)
        assert z > 0

    def test_z_spread_zero_at_risk_free(self):
        """At risk-free price → Z-spread ≈ 0."""
        disc = make_flat_curve(REF, 0.04)
        rb = RiskyBond(REF, END, coupon_rate=0.05)
        riskfree_price = rb.risk_free_price(disc)
        z = z_spread(rb, riskfree_price, disc)
        assert z == pytest.approx(0.0, abs=0.001)

    def test_z_spread_round_trip(self):
        """Price → Z-spread → reprice recovers the market price."""
        disc = make_flat_curve(REF, 0.04)
        rb = RiskyBond(REF, END, coupon_rate=0.05)
        market_price = 95.0
        z = z_spread(rb, market_price, disc)

        # Reprice with bumped curve
        bumped = disc.bumped(z)
        pv = 0.0
        from pricebook.day_count import year_fraction
        for i in range(1, len(rb.schedule)):
            yf = year_fraction(rb.schedule[i-1], rb.schedule[i], rb.day_count)
            pv += rb.notional * rb.coupon_rate * yf * bumped.df(rb.schedule[i])
        pv += rb.notional * bumped.df(rb.end)
        assert pv == pytest.approx(market_price, abs=0.01)


class TestAssetSwapSpread:
    def test_asw_positive_below_par(self):
        disc = make_flat_curve(REF, 0.04)
        rb = RiskyBond(REF, END, coupon_rate=0.05)
        riskfree = rb.risk_free_price(disc)
        asw = asset_swap_spread(rb, riskfree - 5.0, disc)
        assert asw > 0

    def test_asw_zero_at_risk_free(self):
        disc = make_flat_curve(REF, 0.04)
        rb = RiskyBond(REF, END, coupon_rate=0.05)
        riskfree = rb.risk_free_price(disc)
        asw = asset_swap_spread(rb, riskfree, disc)
        assert asw == pytest.approx(0.0, abs=0.001)

    def test_asw_approx_z_spread(self):
        """ASW spread ≈ Z-spread for near-par bonds."""
        disc = make_flat_curve(REF, 0.04)
        rb = RiskyBond(REF, END, coupon_rate=0.05)
        market_price = 97.0
        z = z_spread(rb, market_price, disc)
        asw = asset_swap_spread(rb, market_price, disc)
        # They should be in the same ballpark
        assert abs(asw - z) < 0.01
