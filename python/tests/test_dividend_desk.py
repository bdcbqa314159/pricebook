"""Tests for dividend modelling."""

import pytest
from datetime import date

from pricebook.dividend_desk import (
    implied_dividend_pv, implied_dividends_term_structure,
    strip_discrete_dividends,
    DividendSwap, dividend_forward, dividend_risk,
)
from pricebook.dividend_model import Dividend, dividend_adjusted_forward
from pricebook.discount_curve import DiscountCurve


REF = date(2024, 1, 15)


def _dc(rate=0.05):
    return DiscountCurve.flat(REF, rate)


def _divs():
    return [
        Dividend(date(2024, 6, 15), 2.0),
        Dividend(date(2024, 12, 15), 2.0),
        Dividend(date(2025, 6, 15), 2.0),
    ]


# ---- Implied dividends ----

class TestImpliedDividends:
    def test_put_call_parity(self):
        """With known PV(divs), implied should recover it."""
        dc = _dc()
        spot = 100.0
        K = 100.0
        divs = _divs()[:1]  # single div
        pv_div = divs[0].amount * dc.df(divs[0].ex_date)
        mat = date(2025, 1, 15)
        df = dc.df(mat)
        # Construct option prices from put-call parity
        # C - P = S - PV(divs) - K × df
        c_minus_p = spot - pv_div - K * df
        # Assume C = c_minus_p + 2, P = 2 (arbitrary split)
        C = c_minus_p + 2.0
        P = 2.0
        implied = implied_dividend_pv(spot, C, P, K, df)
        assert implied == pytest.approx(pv_div, rel=0.01)

    def test_term_structure(self):
        result = implied_dividends_term_structure(
            100.0,
            [
                (date(2024, 6, 15), 100, 5.0, 3.0, 0.98),
                (date(2025, 1, 15), 100, 7.0, 4.0, 0.95),
            ],
            REF, _dc(),
        )
        assert len(result) == 2
        # Cumulative PV should be non-decreasing
        assert result[1][1] >= result[0][1]

    def test_strip_dividends(self):
        cumulative = [
            (date(2024, 6, 15), 1.9),
            (date(2025, 1, 15), 3.7),
        ]
        divs = strip_discrete_dividends(cumulative, _dc())
        assert len(divs) == 2
        assert divs[0].amount > 0
        assert divs[1].amount > 0


# ---- Dividend swap ----

class TestDividendSwap:
    def test_at_fair_fixed_zero_pv(self):
        dc = _dc()
        divs = _divs()
        swap = DividendSwap(REF, date(2025, 12, 15), 0.0)
        fair = swap.fair_fixed(divs, dc)
        at_fair = DividendSwap(REF, date(2025, 12, 15), fair)
        assert at_fair.pv(divs, dc) == pytest.approx(0.0, abs=0.01)

    def test_above_fair_negative_pv(self):
        """Paying above fair fixed div → negative PV for receiver."""
        dc = _dc()
        divs = _divs()
        swap = DividendSwap(REF, date(2025, 12, 15), 0.0)
        fair = swap.fair_fixed(divs, dc)
        expensive = DividendSwap(REF, date(2025, 12, 15), fair + 1.0)
        assert expensive.pv(divs, dc) < 0

    def test_fair_fixed_positive(self):
        dc = _dc()
        divs = _divs()
        swap = DividendSwap(REF, date(2025, 12, 15), 0.0)
        fair = swap.fair_fixed(divs, dc)
        assert fair > 0


# ---- Dividend forward ----

class TestDividendForward:
    def test_forward_positive(self):
        dc = _dc()
        fwd = dividend_forward(_divs(), REF, date(2025, 12, 15), dc)
        assert fwd > 0

    def test_no_divs_zero(self):
        dc = _dc()
        fwd = dividend_forward([], REF, date(2025, 12, 15), dc)
        assert fwd == pytest.approx(0.0)

    def test_roundtrip_with_swap(self):
        """Forward should match fair fixed of a swap."""
        dc = _dc()
        divs = _divs()
        end = date(2025, 12, 15)
        fwd = dividend_forward(divs, REF, end, dc)
        swap = DividendSwap(REF, end, 0.0)
        fair = swap.fair_fixed(divs, dc)
        assert fwd == pytest.approx(fair, rel=0.01)


# ---- Dividend risk ----

class TestDividendRisk:
    def test_negative_div_delta(self):
        """Higher dividends → lower forward → div_delta < 0."""
        dc = _dc()
        result = dividend_risk(100.0, _divs(), dc, date(2025, 12, 15))
        assert result.div_delta < 0

    def test_forward_positive(self):
        dc = _dc()
        result = dividend_risk(100.0, _divs(), dc, date(2025, 12, 15))
        assert result.forward_price > 0

    def test_rho_negative(self):
        """Higher div yield → lower forward → rho < 0."""
        dc = _dc()
        result = dividend_risk(100.0, _divs(), dc, date(2025, 12, 15))
        assert result.div_rho < 0

    def test_no_divs_zero_delta(self):
        dc = _dc()
        result = dividend_risk(100.0, [], dc, date(2025, 12, 15))
        assert result.div_delta == pytest.approx(0.0)
