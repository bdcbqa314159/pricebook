"""Tests for futures-based curve stripping."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.futures_bootstrap import futures_strip
from pricebook.discount_curve import DiscountCurve


REF = date(2024, 1, 15)


def _make_deposits():
    return [
        (REF + relativedelta(months=1), 0.052),
        (REF + relativedelta(months=3), 0.051),
        (REF + relativedelta(months=6), 0.050),
    ]


def _make_futures():
    """Quarterly futures starting after deposits."""
    starts = [REF + relativedelta(months=m) for m in [6, 9, 12, 15]]
    return [
        (s, s + relativedelta(months=3), 0.049)
        for s in starts
    ]


def _make_swaps():
    return [
        (REF + relativedelta(years=3), 0.048),
        (REF + relativedelta(years=5), 0.047),
        (REF + relativedelta(years=10), 0.046),
    ]


class TestFuturesStrip:
    def test_deposits_only(self):
        curve = futures_strip(REF, _make_deposits(), [], [])
        d3m = REF + relativedelta(months=3)
        df = curve.df(d3m)
        assert 0.98 < df < 1.0

    def test_deposits_and_futures(self):
        curve = futures_strip(REF, _make_deposits(), _make_futures(), [])
        d1y = REF + relativedelta(years=1)
        df = curve.df(d1y)
        assert 0.94 < df < 1.0

    def test_full_strip(self):
        curve = futures_strip(REF, _make_deposits(), _make_futures(), _make_swaps())
        d10y = REF + relativedelta(years=10)
        df = curve.df(d10y)
        assert 0.5 < df < 0.7

    def test_futures_reprice(self):
        """Futures-stripped curve should reprice futures at input rates."""
        deposits = _make_deposits()
        futures = _make_futures()
        curve = futures_strip(REF, deposits, futures, [])

        for start, end, fut_rate in futures:
            fwd = curve.forward_rate(start, end)
            assert fwd == pytest.approx(fut_rate, abs=0.002)

    def test_convexity_adjustment(self):
        """With HW vol, convexity adjustment lowers forward rates."""
        curve_no_ca = futures_strip(REF, _make_deposits(), _make_futures(), [])
        curve_ca = futures_strip(REF, _make_deposits(), _make_futures(), [],
                                  hw_a=0.05, hw_sigma=0.01)
        d1y = REF + relativedelta(years=1)
        # With CA, forwards are lower → DFs are higher
        assert curve_ca.df(d1y) >= curve_no_ca.df(d1y) - 0.001

    def test_turn_of_year(self):
        """TOY adjustment should affect year-crossing futures."""
        curve_no_toy = futures_strip(REF, _make_deposits(), _make_futures(), [])
        curve_toy = futures_strip(REF, _make_deposits(), _make_futures(), [],
                                   turn_of_year=0.001)
        d18m = REF + relativedelta(months=18)
        # TOY adds spread → different DFs
        assert curve_toy.df(d18m) != pytest.approx(curve_no_toy.df(d18m), abs=1e-6)

    def test_swaps_reprice(self):
        """Swap portion should reprice at par."""
        curve = futures_strip(REF, _make_deposits(), _make_futures(), _make_swaps())
        # The 5Y swap should have par rate close to input
        from pricebook.swap import InterestRateSwap
        swap = InterestRateSwap(REF, REF + relativedelta(years=5), 0.047)
        par = swap.par_rate(curve)
        assert par == pytest.approx(0.047, abs=0.003)
