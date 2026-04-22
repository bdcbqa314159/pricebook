"""Tests for equity hardening (EQ1-EQ9)."""

from datetime import date

import pytest

from pricebook.dividend_model import Dividend, pv_dividends, implied_dividends_from_forwards
from pricebook.equity_forward import EquityForward
from pricebook.equity_trs import EquityTRS
from tests.conftest import make_flat_curve


# ---- EQ1+EQ2+EQ6+EQ9: EquityForward unified with curve, borrow, valuation_date ----

class TestEquityForwardUnified:
    def test_forward_continuous_from_curve(self):
        """Forward uses curve zero rate, not raw rate."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd = EquityForward(100.0, date(2027, 4, 21), ref, div_yield=0.02)
        f = fwd.forward_price(curve)
        # F ≈ 100 × exp((0.04 - 0.02) × 1) ≈ 102.02
        assert 101.5 < f < 103.0

    def test_forward_with_borrow_cost(self):
        """Borrow cost increases forward price."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd_no_borrow = EquityForward(100.0, date(2027, 4, 21), ref, div_yield=0.02)
        fwd_borrow = EquityForward(100.0, date(2027, 4, 21), ref, div_yield=0.02, borrow_cost=0.01)
        assert fwd_borrow.forward_price(curve) > fwd_no_borrow.forward_price(curve)

    def test_forward_with_discrete_dividends(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        divs = [Dividend(date(2026, 7, 15), 1.5), Dividend(date(2027, 1, 15), 1.5)]
        fwd = EquityForward(100.0, date(2027, 4, 21), ref, dividends=divs)
        f = fwd.forward_price(curve)
        # Forward should be reduced by PV of dividends
        fwd_no_div = EquityForward(100.0, date(2027, 4, 21), ref)
        assert f < fwd_no_div.forward_price(curve)

    def test_valuation_date_stored(self):
        fwd = EquityForward(100.0, date(2027, 4, 21), date(2026, 4, 21))
        assert fwd.valuation_date == date(2026, 4, 21)

    def test_maturity_before_valuation_raises(self):
        with pytest.raises(ValueError):
            EquityForward(100.0, date(2025, 1, 1), date(2026, 1, 1))


# ---- EQ3: Delta ----

class TestEquityForwardDelta:
    def test_delta_positive(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd = EquityForward(100.0, date(2027, 4, 21), ref, div_yield=0.02)
        assert fwd.delta(curve) > 0

    def test_delta_approximately_one_short_term(self):
        """Very short-term forward delta ≈ 1."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd = EquityForward(100.0, date(2026, 5, 21), ref)
        assert fwd.delta(curve) == pytest.approx(1.0, abs=0.01)

    def test_forward_dv01(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd = EquityForward(100.0, date(2031, 4, 21), ref)
        dv01 = fwd.forward_dv01(curve)
        assert dv01 > 0  # forward increases with rates


# ---- EQ5: Equity TRS ----

class TestEquityTRS:
    def test_positive_return(self):
        """Stock goes up → TR receiver gains."""
        ref = date(2026, 7, 21)
        curve = make_flat_curve(ref, rate=0.04)
        trs = EquityTRS("SPX", 1_000_000, 0.005,
                         date(2026, 4, 21), date(2026, 10, 21))
        result = trs.mark_to_market(100.0, 110.0, 2.0, curve)
        assert result.price_return > 0
        assert result.dividend_return > 0

    def test_negative_return(self):
        """Stock goes down → TR receiver loses."""
        ref = date(2026, 7, 21)
        curve = make_flat_curve(ref, rate=0.04)
        trs = EquityTRS("SPX", 1_000_000, 0.005,
                         date(2026, 4, 21), date(2026, 10, 21))
        result = trs.mark_to_market(100.0, 90.0, 1.0, curve)
        assert result.price_return < 0

    def test_breakeven(self):
        trs = EquityTRS("SPX", 1_000_000, 0.005,
                         date(2026, 4, 21), date(2027, 4, 21))
        be = trs.breakeven_return(0.04)
        assert be == pytest.approx(0.045)


# ---- EQ7: pv_dividends excludes maturity date ----

class TestDividendMaturityExclusion:
    def test_excludes_on_maturity(self):
        """Dividend on maturity date should be excluded."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        divs = [
            Dividend(date(2026, 7, 15), 1.5),
            Dividend(date(2027, 4, 21), 1.5),  # ON maturity
        ]
        pv = pv_dividends(divs, curve, date(2027, 4, 21))
        # Only the July dividend should be included
        pv_jul_only = 1.5 * curve.df(date(2026, 7, 15))
        assert pv == pytest.approx(pv_jul_only, rel=1e-10)

    def test_includes_before_maturity(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        divs = [Dividend(date(2026, 7, 15), 1.5)]
        pv = pv_dividends(divs, curve, date(2027, 4, 21))
        assert pv > 0


# ---- EQ8: Implied dividends from forward curve ----

class TestImpliedDividendsFromForwards:
    def test_detects_dividend(self):
        """Forward curve with drops should imply dividends."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        # Forward drops by ~2 between Jul and Oct (implied dividend)
        dates = [date(2026, 7, 21), date(2026, 10, 21), date(2027, 1, 21)]
        fwds = [102.0, 100.0, 102.0]  # drop at Oct implies dividend near Jul-Oct
        divs = implied_dividends_from_forwards(100.0, dates, fwds, curve)
        assert len(divs) > 0

    def test_no_div_on_rising_curve(self):
        """Steadily rising forwards → no implied dividends (or minimal)."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        dates = [date(2026, 7, 21), date(2026, 10, 21), date(2027, 1, 21)]
        fwds = [101.0, 102.0, 103.0]
        divs = implied_dividends_from_forwards(100.0, dates, fwds, curve)
        # Should be empty or near-zero (forward rises throughout)
        total = sum(d.amount for d in divs)
        assert total < 1.0
