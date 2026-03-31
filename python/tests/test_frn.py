"""Tests for floating-rate note."""

import pytest
import math
from datetime import date

from pricebook.frn import FloatingRateNote
from pricebook.discount_curve import DiscountCurve
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
MATURITY = date(2029, 1, 15)


class TestConstruction:
    def test_basic(self):
        frn = FloatingRateNote(start=REF, end=MATURITY, spread=0.005)
        assert frn.spread == 0.005
        assert frn.notional == 1_000_000.0

    def test_zero_spread(self):
        frn = FloatingRateNote(start=REF, end=MATURITY, spread=0.0)
        assert frn.spread == 0.0


class TestDirtyPrice:
    def test_at_par_zero_spread_single_curve(self):
        """FRN with zero spread prices at par on its own curve."""
        curve = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=MATURITY, spread=0.0)
        price = frn.dirty_price(curve)
        assert price == pytest.approx(100.0, abs=0.5)

    def test_positive_spread_above_par(self):
        """Positive spread → price above par (more coupon than discount rate)."""
        curve = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=MATURITY, spread=0.01)
        price = frn.dirty_price(curve)
        assert price > 100.0

    def test_negative_spread_below_par(self):
        """Negative spread → price below par."""
        curve = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=MATURITY, spread=-0.01)
        price = frn.dirty_price(curve)
        assert price < 100.0

    def test_dual_curve(self):
        """Dual-curve pricing: different projection and discount curves."""
        disc = make_flat_curve(REF, 0.04)
        proj = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=MATURITY, spread=0.0)
        price = frn.dirty_price(disc, proj)
        # Projection rate > discount rate → above par
        assert price > 100.0

    def test_price_positive(self):
        curve = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=MATURITY, spread=0.005)
        assert frn.dirty_price(curve) > 0


class TestCleanPrice:
    def test_clean_leq_dirty_after_start(self):
        """Clean price ≤ dirty price (accrued is non-negative)."""
        curve = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=MATURITY, spread=0.005)
        dirty = frn.dirty_price(curve)
        clean = frn.clean_price(curve)
        assert clean <= dirty + 0.01  # allow small tolerance


class TestDiscountMargin:
    def test_dm_zero_at_par(self):
        """DM = 0 when market price = par and spread = 0."""
        curve = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=MATURITY, spread=0.0)
        par_price = frn.dirty_price(curve)
        dm = frn.discount_margin(par_price, curve)
        assert dm == pytest.approx(0.0, abs=0.001)

    def test_dm_negative_below_par(self):
        """Below-par price → negative DM (less spread needed to match lower price)."""
        curve = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=MATURITY, spread=0.005)
        par_price = frn.dirty_price(curve)
        dm = frn.discount_margin(par_price - 2.0, curve)
        assert dm < 0

    def test_dm_positive_above_par(self):
        """Above-par price → positive DM (more spread needed to match higher price)."""
        curve = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=MATURITY, spread=0.005)
        par_price = frn.dirty_price(curve)
        dm = frn.discount_margin(par_price + 2.0, curve)
        assert dm > 0

    def test_dm_round_trip(self):
        """Reprice with DM-adjusted spread recovers the market price."""
        curve = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=MATURITY, spread=0.005)
        market_price = 99.0
        dm = frn.discount_margin(market_price, curve)

        # Reprice with adjusted spread
        frn_adj = FloatingRateNote(start=REF, end=MATURITY, spread=0.005 + dm)
        repriced = frn_adj.dirty_price(curve)
        assert repriced == pytest.approx(market_price, abs=0.01)
