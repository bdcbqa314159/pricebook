"""Tests for equity index and commodity futures."""

import math
import pytest
from datetime import date

from pricebook.futures import (
    EquityFuture, CommodityFuture,
    contango_or_backwardation, roll_yield,
    calendar_spread, futures_strip_curve, implied_convenience_yield,
)


REF = date(2024, 1, 15)


# ---- Equity futures ----

class TestEquityFuture:
    def test_fair_price_no_dividend(self):
        """F = S × exp(r × T)."""
        ef = EquityFuture(spot=5000, expiry=date(2024, 7, 15), rate=0.05)
        fp = ef.fair_price(REF)
        T = (date(2024, 7, 15) - REF).days / 365.0
        assert fp == pytest.approx(5000 * math.exp(0.05 * T), rel=1e-6)

    def test_fair_price_with_dividend(self):
        ef = EquityFuture(spot=5000, expiry=date(2024, 7, 15), rate=0.05, div_yield=0.02)
        fp = ef.fair_price(REF)
        T = (date(2024, 7, 15) - REF).days / 365.0
        assert fp == pytest.approx(5000 * math.exp(0.03 * T), rel=1e-6)

    def test_convergence_at_expiry(self):
        """At expiry, futures = spot."""
        ef = EquityFuture(spot=5000, expiry=REF, rate=0.05)
        assert ef.fair_price(REF) == pytest.approx(5000)
        assert ef.convergence(REF) == pytest.approx(0.0)

    def test_basis_positive_no_div(self):
        """Positive rates, no dividends → futures > spot → positive basis."""
        ef = EquityFuture(spot=5000, expiry=date(2025, 1, 15), rate=0.05)
        assert ef.basis(REF) > 0

    def test_daily_settlement_pnl(self):
        ef = EquityFuture(spot=5000, expiry=date(2024, 7, 15), rate=0.05,
                         notional_per_point=50)
        pnl = ef.daily_settlement_pnl(5000, 5010, contracts=2)
        assert pnl == pytest.approx(10 * 50 * 2)

    def test_negative_pnl(self):
        ef = EquityFuture(spot=5000, expiry=date(2024, 7, 15), rate=0.05,
                         notional_per_point=50)
        pnl = ef.daily_settlement_pnl(5010, 5000, contracts=1)
        assert pnl == pytest.approx(-500)


# ---- Commodity futures ----

class TestCommodityFuture:
    def test_contango(self):
        near = CommodityFuture("WTI", date(2024, 3, 15), 75.0)
        far = CommodityFuture("WTI", date(2024, 6, 15), 78.0)
        assert contango_or_backwardation(near, far) == "contango"

    def test_backwardation(self):
        near = CommodityFuture("WTI", date(2024, 3, 15), 80.0)
        far = CommodityFuture("WTI", date(2024, 6, 15), 75.0)
        assert contango_or_backwardation(near, far) == "backwardation"

    def test_flat(self):
        near = CommodityFuture("WTI", date(2024, 3, 15), 75.0)
        far = CommodityFuture("WTI", date(2024, 6, 15), 75.0)
        assert contango_or_backwardation(near, far) == "flat"

    def test_wrong_order_raises(self):
        near = CommodityFuture("WTI", date(2024, 6, 15), 75.0)
        far = CommodityFuture("WTI", date(2024, 3, 15), 78.0)
        with pytest.raises(ValueError):
            contango_or_backwardation(near, far)

    def test_roll_yield_positive_backwardation(self):
        near = CommodityFuture("WTI", date(2024, 3, 15), 80.0)
        far = CommodityFuture("WTI", date(2024, 6, 15), 75.0)
        ry = roll_yield(near, far)
        assert ry > 0

    def test_roll_yield_negative_contango(self):
        near = CommodityFuture("WTI", date(2024, 3, 15), 75.0)
        far = CommodityFuture("WTI", date(2024, 6, 15), 80.0)
        ry = roll_yield(near, far)
        assert ry < 0


# ---- Calendar spread ----

class TestCalendarSpread:
    def test_backwardation_positive_spread(self):
        near = CommodityFuture("WTI", date(2024, 3, 15), 80.0)
        far = CommodityFuture("WTI", date(2024, 6, 15), 75.0)
        result = calendar_spread(near, far)
        assert result.spread == pytest.approx(5.0)
        assert result.structure == "backwardation"
        assert result.roll_yield > 0

    def test_contango_negative_spread(self):
        near = CommodityFuture("WTI", date(2024, 3, 15), 75.0)
        far = CommodityFuture("WTI", date(2024, 6, 15), 80.0)
        result = calendar_spread(near, far)
        assert result.spread == pytest.approx(-5.0)
        assert result.structure == "contango"

    def test_prices(self):
        near = CommodityFuture("WTI", date(2024, 3, 15), 75.0)
        far = CommodityFuture("WTI", date(2024, 6, 15), 78.0)
        result = calendar_spread(near, far)
        assert result.near_price == 75.0
        assert result.far_price == 78.0


# ---- Futures strip ----

class TestFuturesStrip:
    def test_sorted(self):
        futures = [
            CommodityFuture("WTI", date(2024, 6, 15), 78),
            CommodityFuture("WTI", date(2024, 3, 15), 75),
            CommodityFuture("WTI", date(2024, 9, 15), 80),
        ]
        strip = futures_strip_curve(futures)
        dates = [d for d, _ in strip]
        assert dates == sorted(dates)


# ---- Implied convenience yield ----

class TestConvenienceYield:
    def test_no_convenience_yield(self):
        """F = S × exp(r × T) → y = 0 (no storage cost)."""
        S, r, T = 100, 0.05, 1.0
        F = S * math.exp(r * T)
        y = implied_convenience_yield(S, F, r, 0.0, T)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_positive_convenience(self):
        """F < S × exp(r × T) → positive convenience yield."""
        S, r, T = 100, 0.05, 1.0
        F = S * math.exp(0.02 * T)  # as if y=0.03
        y = implied_convenience_yield(S, F, r, 0.0, T)
        assert y == pytest.approx(0.03, rel=1e-6)

    def test_with_storage(self):
        """Storage cost raises the implied convenience yield."""
        S, r, T, c = 100, 0.05, 1.0, 0.01
        F = S * math.exp((r + c) * T)
        y = implied_convenience_yield(S, F, r, c, T)
        assert y == pytest.approx(0.0, abs=1e-10)
