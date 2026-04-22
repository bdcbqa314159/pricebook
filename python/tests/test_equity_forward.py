"""Tests for equity forward pricing."""

import pytest
import math
from datetime import date

from pricebook.equity_forward import EquityForward
from pricebook.dividend_model import Dividend
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
MATURITY = date(2025, 1, 15)
SPOT = 100.0
RATE = 0.05


class TestConstruction:
    def test_basic(self):
        fwd = EquityForward(spot=100.0, maturity=MATURITY, valuation_date=REF)
        assert fwd.spot == 100.0
        assert fwd.div_yield == 0.0
        assert fwd.dividends == []

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            EquityForward(spot=-1, maturity=MATURITY, valuation_date=REF)


class TestForwardPrice:
    def test_no_dividend(self):
        curve = make_flat_curve(REF, RATE)
        fwd = EquityForward(spot=SPOT, maturity=MATURITY, valuation_date=REF)
        f = fwd.forward_price(curve)
        expected = SPOT / curve.df(MATURITY)
        assert f == pytest.approx(expected, rel=1e-6)

    def test_with_dividend_yield(self):
        q = 0.02
        curve = make_flat_curve(REF, RATE)
        fwd = EquityForward(spot=SPOT, maturity=MATURITY, valuation_date=REF, div_yield=q)
        f = fwd.forward_price(curve)
        T = (MATURITY - REF).days / 365.0
        expected = SPOT * math.exp((RATE - q) * T)
        assert f == pytest.approx(expected, rel=1e-4)


class TestDiscreteForward:
    def test_no_dividends(self):
        """With no dividends, forward = S / df(T)."""
        curve = make_flat_curve(REF, RATE)
        fwd = EquityForward(spot=SPOT, maturity=MATURITY, valuation_date=REF)
        f = fwd.forward_price(curve)
        expected = SPOT / curve.df(MATURITY)
        assert f == pytest.approx(expected, rel=1e-10)

    def test_single_dividend(self):
        curve = make_flat_curve(REF, RATE)
        divs = [Dividend(ex_date=date(2024, 7, 15), amount=2.0)]
        fwd = EquityForward(spot=SPOT, maturity=MATURITY, valuation_date=REF, dividends=divs)
        f = fwd.forward_price(curve)

        pv_div = 2.0 * curve.df(date(2024, 7, 15))
        expected = (SPOT - pv_div) / curve.df(MATURITY)
        assert f == pytest.approx(expected, rel=1e-10)

    def test_dividend_reduces_forward(self):
        curve = make_flat_curve(REF, RATE)
        fwd_no_div = EquityForward(spot=SPOT, maturity=MATURITY, valuation_date=REF)
        fwd_with_div = EquityForward(
            spot=SPOT, maturity=MATURITY, valuation_date=REF,
            dividends=[Dividend(date(2024, 7, 15), 3.0)],
        )
        assert fwd_with_div.forward_price(curve) < fwd_no_div.forward_price(curve)

    def test_dividend_after_maturity_ignored(self):
        curve = make_flat_curve(REF, RATE)
        divs = [Dividend(ex_date=date(2026, 1, 15), amount=5.0)]  # after maturity
        fwd = EquityForward(spot=SPOT, maturity=MATURITY, valuation_date=REF, dividends=divs)
        f = fwd.forward_price(curve)
        expected = SPOT / curve.df(MATURITY)
        assert f == pytest.approx(expected, rel=1e-10)

    def test_multiple_dividends(self):
        curve = make_flat_curve(REF, RATE)
        divs = [
            Dividend(date(2024, 4, 15), 1.0),
            Dividend(date(2024, 7, 15), 1.0),
            Dividend(date(2024, 10, 15), 1.0),
        ]
        fwd = EquityForward(spot=SPOT, maturity=MATURITY, valuation_date=REF, dividends=divs)
        f = fwd.forward_price(curve)

        pv_divs = sum(d.amount * curve.df(d.ex_date) for d in divs)
        expected = (SPOT - pv_divs) / curve.df(MATURITY)
        assert f == pytest.approx(expected, rel=1e-10)


class TestPV:
    def test_pv_at_forward_strike_is_zero(self):
        curve = make_flat_curve(REF, RATE)
        fwd = EquityForward(spot=SPOT, maturity=MATURITY, valuation_date=REF)
        strike = fwd.forward_price(curve)
        pv = fwd.pv(strike, curve)
        assert pv == pytest.approx(0.0, abs=0.01)

    def test_pv_with_dividends(self):
        curve = make_flat_curve(REF, RATE)
        divs = [Dividend(date(2024, 7, 15), 2.0)]
        fwd = EquityForward(spot=SPOT, maturity=MATURITY, valuation_date=REF, dividends=divs)
        f = fwd.forward_price(curve)
        pv = fwd.pv(f, curve)
        assert pv == pytest.approx(0.0, abs=0.01)

    def test_pv_positive_when_forward_above_strike(self):
        curve = make_flat_curve(REF, RATE)
        fwd = EquityForward(spot=SPOT, maturity=MATURITY, valuation_date=REF)
        pv = fwd.pv(90.0, curve)  # strike well below forward
        assert pv > 0
