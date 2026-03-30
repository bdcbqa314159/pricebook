"""
Slice 10 round-trip validation: equity options + dividends.

1. Put-call parity with dividends
2. European MC matches analytical (from Slice 9)
3. Dividend-adjusted forward recovers market forward
4. Greeks: analytical vs bump-and-reprice
"""

import pytest
import math
from datetime import date

from pricebook.equity_option import (
    equity_option_price,
    equity_delta,
    equity_gamma,
    equity_vega,
    equity_rho,
)
from pricebook.equity_forward import EquityForward, Dividend
from pricebook.dividend_model import (
    dividend_adjusted_forward,
    equity_option_discrete_divs,
)
from pricebook.mc_pricer import mc_european
from pricebook.discount_curve import DiscountCurve
from pricebook.black76 import OptionType


def _flat_curve(ref: date, rate: float) -> DiscountCurve:
    dates = [date(ref.year + i, ref.month, ref.day) for i in range(1, 11)]
    dfs = [math.exp(-rate * i) for i in range(1, 11)]
    return DiscountCurve(reference_date=ref, dates=dates, dfs=dfs)


REF = date(2024, 1, 15)
MATURITY = date(2025, 1, 15)
SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0


class TestPutCallParityWithDividends:
    @pytest.mark.parametrize("q", [0.0, 0.02, 0.05])
    def test_continuous_parity(self, q):
        """C - P = S*exp(-q*T) - K*exp(-r*T) for continuous dividends."""
        c = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        p = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT, q)
        expected = SPOT * math.exp(-q * T) - STRIKE * math.exp(-RATE * T)
        assert c - p == pytest.approx(expected, abs=1e-10)

    def test_discrete_parity(self):
        """C - P = df(T) * (F_adj - K) for discrete dividends."""
        curve = _flat_curve(REF, RATE)
        divs = [Dividend(date(2024, 7, 15), 2.0), Dividend(date(2024, 10, 15), 1.5)]
        K = 100.0
        c = equity_option_discrete_divs(SPOT, K, divs, curve, VOL, MATURITY, OptionType.CALL)
        p = equity_option_discrete_divs(SPOT, K, divs, curve, VOL, MATURITY, OptionType.PUT)
        fwd = dividend_adjusted_forward(SPOT, divs, curve, MATURITY)
        df = curve.df(MATURITY)
        assert c - p == pytest.approx(df * (fwd - K), abs=1e-10)


class TestMCvsAnalytical:
    def test_mc_matches_bs_call(self):
        """MC European matches Black-Scholes for call."""
        analytical = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        mc = mc_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, n_paths=200_000)
        assert abs(mc.price - analytical) < 3 * mc.std_error

    def test_mc_matches_bs_put(self):
        analytical = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        mc = mc_european(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT, n_paths=200_000)
        assert abs(mc.price - analytical) < 3 * mc.std_error

    def test_mc_with_dividend_yield(self):
        q = 0.03
        analytical = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        mc = mc_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                         div_yield=q, n_paths=200_000)
        assert abs(mc.price - analytical) < 3 * mc.std_error


class TestDividendAdjustedForwardRecovery:
    def test_forward_recovers_with_no_divs(self):
        """No dividends: adjusted forward = S / df(T)."""
        curve = _flat_curve(REF, RATE)
        fwd = dividend_adjusted_forward(SPOT, [], curve, MATURITY)
        expected = SPOT / curve.df(MATURITY)
        assert fwd == pytest.approx(expected, rel=1e-10)

    def test_forward_consistent_with_equity_forward(self):
        """dividend_model and equity_forward give the same answer."""
        curve = _flat_curve(REF, RATE)
        divs = [Dividend(date(2024, 7, 15), 2.0)]

        fwd_model = dividend_adjusted_forward(SPOT, divs, curve, MATURITY)

        ef = EquityForward(spot=SPOT, maturity=MATURITY, rate=RATE, dividends=divs)
        fwd_ef = ef.forward_discrete(curve)

        assert fwd_model == pytest.approx(fwd_ef, rel=1e-10)


class TestGreeksConsistency:
    """Analytical Greeks match bump-and-reprice."""

    BUMP = 0.01

    def test_delta_vs_bump(self):
        d = equity_delta(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        p_up = equity_option_price(SPOT + self.BUMP, STRIKE, RATE, VOL, T, OptionType.CALL)
        p_dn = equity_option_price(SPOT - self.BUMP, STRIKE, RATE, VOL, T, OptionType.CALL)
        bump_d = (p_up - p_dn) / (2 * self.BUMP)
        assert d == pytest.approx(bump_d, rel=1e-3)

    def test_gamma_vs_bump(self):
        g = equity_gamma(SPOT, STRIKE, RATE, VOL, T)
        d_up = equity_delta(SPOT + self.BUMP, STRIKE, RATE, VOL, T, OptionType.CALL)
        d_dn = equity_delta(SPOT - self.BUMP, STRIKE, RATE, VOL, T, OptionType.CALL)
        bump_g = (d_up - d_dn) / (2 * self.BUMP)
        assert g == pytest.approx(bump_g, rel=1e-3)

    def test_vega_vs_bump(self):
        v = equity_vega(SPOT, STRIKE, RATE, VOL, T)
        bump = 1e-4
        p_up = equity_option_price(SPOT, STRIKE, RATE, VOL + bump, T, OptionType.CALL)
        p_dn = equity_option_price(SPOT, STRIKE, RATE, VOL - bump, T, OptionType.CALL)
        bump_v = (p_up - p_dn) / (2 * bump)
        assert v == pytest.approx(bump_v, rel=1e-3)

    def test_rho_vs_bump(self):
        r = equity_rho(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        bump = 1e-4
        p_up = equity_option_price(SPOT, STRIKE, RATE + bump, VOL, T, OptionType.CALL)
        p_dn = equity_option_price(SPOT, STRIKE, RATE - bump, VOL, T, OptionType.CALL)
        bump_r = (p_up - p_dn) / (2 * bump)
        assert r == pytest.approx(bump_r, rel=0.01)

    def test_delta_with_dividend_vs_bump(self):
        q = 0.02
        d = equity_delta(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        bump = 0.01
        p_up = equity_option_price(SPOT + bump, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        p_dn = equity_option_price(SPOT - bump, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        bump_d = (p_up - p_dn) / (2 * bump)
        assert d == pytest.approx(bump_d, rel=1e-3)
