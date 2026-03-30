"""Tests for discrete dividend handling."""

import pytest
from datetime import date

from pricebook.dividend_model import (
    pv_dividends,
    dividend_adjusted_forward,
    piecewise_forward,
    equity_option_discrete_divs,
)
from pricebook.dividend_model import Dividend
from pricebook.equity_option import equity_option_price
from pricebook.black76 import OptionType
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
MATURITY = date(2025, 1, 15)
SPOT = 100.0
RATE = 0.05
VOL = 0.20


class TestPVDividends:
    def test_single_dividend(self):
        curve = make_flat_curve(REF, RATE)
        divs = [Dividend(date(2024, 7, 15), 2.0)]
        pv = pv_dividends(divs, curve, MATURITY)
        expected = 2.0 * curve.df(date(2024, 7, 15))
        assert pv == pytest.approx(expected)

    def test_no_dividends(self):
        curve = make_flat_curve(REF, RATE)
        assert pv_dividends([], curve, MATURITY) == 0.0

    def test_dividend_after_maturity_excluded(self):
        curve = make_flat_curve(REF, RATE)
        divs = [Dividend(date(2026, 1, 15), 5.0)]
        assert pv_dividends(divs, curve, MATURITY) == 0.0


class TestAdjustedForward:
    def test_no_dividends_equals_simple_forward(self):
        curve = make_flat_curve(REF, RATE)
        fwd = dividend_adjusted_forward(SPOT, [], curve, MATURITY)
        expected = SPOT / curve.df(MATURITY)
        assert fwd == pytest.approx(expected)

    def test_dividends_reduce_forward(self):
        curve = make_flat_curve(REF, RATE)
        fwd_no = dividend_adjusted_forward(SPOT, [], curve, MATURITY)
        divs = [Dividend(date(2024, 7, 15), 3.0)]
        fwd_div = dividend_adjusted_forward(SPOT, divs, curve, MATURITY)
        assert fwd_div < fwd_no


class TestPiecewiseForward:
    def test_forward_drops_at_ex_date(self):
        curve = make_flat_curve(REF, RATE)
        divs = [Dividend(date(2024, 7, 15), 2.0)]
        query_dates = [date(2024, 7, 14), date(2024, 7, 15), date(2024, 7, 16)]
        fwds = piecewise_forward(SPOT, divs, curve, query_dates)
        # Forward should drop after the ex-date
        assert fwds[0] > fwds[1]  # before vs on ex-date
        assert fwds[1] == pytest.approx(fwds[2], rel=0.01)  # on vs day after (similar)

    def test_no_dividends_smooth(self):
        curve = make_flat_curve(REF, RATE)
        query_dates = [date(2024, 4, 15), date(2024, 7, 15), date(2024, 10, 15)]
        fwds = piecewise_forward(SPOT, [], curve, query_dates)
        # Forward should be monotonically increasing (positive rates)
        assert fwds[0] < fwds[1] < fwds[2]

    def test_multiple_dividends(self):
        curve = make_flat_curve(REF, RATE)
        divs = [
            Dividend(date(2024, 4, 15), 1.0),
            Dividend(date(2024, 10, 15), 1.0),
        ]
        query_dates = [date(2024, 3, 15), date(2024, 6, 15), date(2024, 12, 15)]
        fwds = piecewise_forward(SPOT, divs, curve, query_dates)
        # After first div, forward should be lower than "no div" forward
        fwds_no = piecewise_forward(SPOT, [], curve, query_dates)
        assert fwds[0] == pytest.approx(fwds_no[0])  # before first div
        assert fwds[1] < fwds_no[1]  # after first div
        assert fwds[2] < fwds_no[2]  # after both divs


class TestOptionWithDiscreteDivs:
    def test_call_positive(self):
        curve = make_flat_curve(REF, RATE)
        divs = [Dividend(date(2024, 7, 15), 2.0)]
        p = equity_option_discrete_divs(SPOT, 100.0, divs, curve, VOL, MATURITY)
        assert p > 0

    def test_put_positive(self):
        curve = make_flat_curve(REF, RATE)
        divs = [Dividend(date(2024, 7, 15), 2.0)]
        p = equity_option_discrete_divs(SPOT, 100.0, divs, curve, VOL, MATURITY,
                                         OptionType.PUT)
        assert p > 0

    def test_put_call_parity_with_divs(self):
        """C - P = df(T) * (F - K) where F is dividend-adjusted forward."""
        curve = make_flat_curve(REF, RATE)
        divs = [Dividend(date(2024, 7, 15), 2.0)]
        K = 100.0
        c = equity_option_discrete_divs(SPOT, K, divs, curve, VOL, MATURITY, OptionType.CALL)
        p = equity_option_discrete_divs(SPOT, K, divs, curve, VOL, MATURITY, OptionType.PUT)
        fwd = dividend_adjusted_forward(SPOT, divs, curve, MATURITY)
        df = curve.df(MATURITY)
        assert c - p == pytest.approx(df * (fwd - K), abs=1e-10)

    def test_dividends_reduce_call_price(self):
        """Discrete dividends reduce call value (lower forward)."""
        curve = make_flat_curve(REF, RATE)
        c_no = equity_option_price(SPOT, 100.0, RATE, VOL, 1.0, OptionType.CALL)
        divs = [Dividend(date(2024, 7, 15), 3.0)]
        c_div = equity_option_discrete_divs(SPOT, 100.0, divs, curve, VOL, MATURITY)
        assert c_div < c_no

    def test_dividends_increase_put_price(self):
        """Discrete dividends increase put value (lower forward)."""
        curve = make_flat_curve(REF, RATE)
        p_no = equity_option_price(SPOT, 100.0, RATE, VOL, 1.0, OptionType.PUT)
        divs = [Dividend(date(2024, 7, 15), 3.0)]
        p_div = equity_option_discrete_divs(SPOT, 100.0, divs, curve, VOL, MATURITY,
                                             OptionType.PUT)
        assert p_div > p_no

    def test_no_dividends_matches_continuous(self):
        """No dividends + discrete model ≈ continuous model with q=0."""
        curve = make_flat_curve(REF, RATE)
        p_discrete = equity_option_discrete_divs(SPOT, 100.0, [], curve, VOL, MATURITY)
        p_continuous = equity_option_price(SPOT, 100.0, RATE, VOL, 1.0, OptionType.CALL)
        assert p_discrete == pytest.approx(p_continuous, rel=0.01)
