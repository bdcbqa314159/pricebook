"""Tests for rich/cheap analysis."""

import math
import pytest
from datetime import date

from pricebook.rich_cheap import (
    relative_value, rv_from_curve,
    spread_monitor, butterfly_monitor,
)
from pricebook.discount_curve import DiscountCurve
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


# ---- Relative value ----

class TestRelativeValue:
    def test_at_market_zero_spread(self):
        rv = relative_value(0.050, 0.050)
        assert rv.spread == pytest.approx(0.0)
        assert rv.signal == "fair"

    def test_off_market_nonzero_spread(self):
        rv = relative_value(0.051, 0.050)
        assert rv.spread == pytest.approx(0.001)

    def test_z_score_with_history(self):
        history = [0.001, 0.002, -0.001, 0.000, 0.001]
        rv = relative_value(0.053, 0.050, history=history)
        assert rv.z_score is not None
        assert rv.percentile is not None

    def test_rich_signal(self):
        history = [0.0001, 0.0002, -0.0001, 0.0000, 0.0001]
        rv = relative_value(0.060, 0.050, history=history, threshold=2.0)
        assert rv.signal == "rich"

    def test_cheap_signal(self):
        history = [0.0001, 0.0002, -0.0001, 0.0000, 0.0001]
        rv = relative_value(0.040, 0.050, history=history, threshold=2.0)
        assert rv.signal == "cheap"

    def test_fair_within_threshold(self):
        history = [0.001, -0.001, 0.002, -0.002, 0.000]
        rv = relative_value(0.0505, 0.050, history=history, threshold=2.0)
        assert rv.signal == "fair"

    def test_percentile(self):
        history = [0.001, 0.002, 0.003, 0.004, 0.005]
        rv = relative_value(0.050, 0.047, history=history)
        # spread = 0.003, percentile should be 60%
        assert rv.percentile == pytest.approx(60.0)

    def test_no_history(self):
        rv = relative_value(0.051, 0.050)
        assert rv.z_score is None
        assert rv.percentile is None


class TestRVFromCurve:
    def test_at_par_zero_spread(self):
        curve = _curve(0.05)
        swap = InterestRateSwap(
            REF, date(2034, 1, 15), fixed_rate=0.05,
            direction=SwapDirection.PAYER, notional=1_000_000,
        )
        par = swap.par_rate(curve)
        rv = rv_from_curve(curve, REF, 10, market_rate=par)
        assert abs(rv.spread) < 1e-10

    def test_off_par_nonzero(self):
        curve = _curve(0.05)
        rv = rv_from_curve(curve, REF, 10, market_rate=0.06)
        assert rv.spread > 0


# ---- Spread monitor ----

class TestSpreadMonitor:
    def test_flat_curve_zero_spread(self):
        curve = _curve(0.05)
        sl = spread_monitor(curve, REF, 2, 10)
        assert abs(sl.spread) < 1e-4  # near zero (tiny day-count noise)

    def test_upward_curve_positive_spread(self):
        curve = _upward_curve()
        sl = spread_monitor(curve, REF, 2, 10)
        assert sl.spread > 0

    def test_name(self):
        sl = spread_monitor(_curve(), REF, 2, 10)
        assert sl.name == "2s10s"

    def test_signal_with_history(self):
        curve = _upward_curve()
        sl = spread_monitor(curve, REF, 2, 10)
        # Use a tight history far from current spread
        history = [0.0001] * 20
        sl2 = spread_monitor(curve, REF, 2, 10, history=history, threshold=2.0)
        assert sl2.signal in ("wide", "tight", "fair")

    def test_different_tenors(self):
        curve = _upward_curve()
        s1 = spread_monitor(curve, REF, 2, 10)
        s2 = spread_monitor(curve, REF, 5, 30)
        assert s1.spread != s2.spread


# ---- Butterfly monitor ----

class TestButterflyMonitor:
    def test_flat_curve_zero_butterfly(self):
        curve = _curve(0.05)
        bf = butterfly_monitor(curve, REF, 2, 5, 10)
        assert abs(bf.butterfly) < 1e-4  # near zero (tiny day-count noise)

    def test_name(self):
        bf = butterfly_monitor(_curve(), REF, 2, 5, 10)
        assert bf.name == "2s5s10s"

    def test_upward_curve_nonzero(self):
        curve = _upward_curve()
        bf = butterfly_monitor(curve, REF, 2, 5, 10)
        # On a smooth upward curve, butterfly should be nonzero
        assert bf.butterfly != 0.0

    def test_signal_with_history(self):
        curve = _upward_curve()
        history = [0.00001] * 20
        bf = butterfly_monitor(curve, REF, 2, 5, 10, history=history, threshold=2.0)
        assert bf.signal in ("belly_cheap", "belly_rich", "fair")

    def test_rates_consistent(self):
        curve = _upward_curve()
        bf = butterfly_monitor(curve, REF, 2, 5, 10)
        expected = (bf.short_rate + bf.long_rate) / 2.0 - bf.belly_rate
        assert bf.butterfly == pytest.approx(expected)
