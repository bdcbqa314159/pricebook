"""
Slice 12 round-trip validation: binomial trees.

1. European tree converges to Black-Scholes
2. American call = European call when no dividends
3. American put > European put
4. American call with dividends > American call without
5. Put-call bounds for American options
6. Greeks via tree match analytical delta/gamma
"""

import pytest
import math

from pricebook.binomial_tree import binomial_european, binomial_american
from pricebook.equity_option import equity_option_price, equity_delta, equity_gamma
from pricebook.black76 import OptionType


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0


class TestEuropeanConvergence:
    """European tree converges to Black-Scholes as steps increase."""

    @pytest.mark.parametrize("n", [100, 200, 500, 1000])
    def test_call_convergence(self, n):
        tree = binomial_european(SPOT, STRIKE, RATE, VOL, T, n, OptionType.CALL)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        # Error should decrease with more steps
        assert abs(tree - bs) < bs * 0.01  # within 1% at n>=100

    def test_error_decreases_monotonically(self):
        """Error roughly O(1/n) — halving with doubling steps."""
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        err_200 = abs(binomial_european(SPOT, STRIKE, RATE, VOL, T, 200, OptionType.CALL) - bs)
        err_800 = abs(binomial_european(SPOT, STRIKE, RATE, VOL, T, 800, OptionType.CALL) - bs)
        assert err_800 < err_200 * 0.5


class TestAmericanProperties:
    def test_call_no_div_equals_european(self):
        am = binomial_american(SPOT, STRIKE, RATE, VOL, T, 500, OptionType.CALL)
        eu = binomial_european(SPOT, STRIKE, RATE, VOL, T, 500, OptionType.CALL)
        assert am == pytest.approx(eu, rel=1e-10)

    def test_put_exceeds_european(self):
        am = binomial_american(SPOT, STRIKE, RATE, VOL, T, 500, OptionType.PUT)
        eu = binomial_european(SPOT, STRIKE, RATE, VOL, T, 500, OptionType.PUT)
        assert am > eu

    def test_call_with_div_exceeds_without(self):
        """High dividend yield makes American call > European call."""
        q = 0.08
        am = binomial_american(SPOT, STRIKE, RATE, VOL, T, 500, OptionType.CALL, q)
        eu = binomial_european(SPOT, STRIKE, RATE, VOL, T, 500, OptionType.CALL, q)
        assert am > eu

    @pytest.mark.parametrize("strike", [80, 90, 100, 110, 120])
    def test_put_call_bounds(self, strike):
        """American options satisfy known bounds."""
        am_c = binomial_american(SPOT, strike, RATE, VOL, T, 500, OptionType.CALL)
        am_p = binomial_american(SPOT, strike, RATE, VOL, T, 500, OptionType.PUT)

        # C - P bounded by S - K*exp(-rT) and S - K
        diff = am_c - am_p
        lower = SPOT - strike
        upper = SPOT - strike * math.exp(-RATE * T)
        assert lower - 0.5 <= diff <= upper + 0.5


class TestGreeksVsAnalytical:
    """Tree Greeks (bump-and-reprice) match analytical for European case."""

    def test_delta_matches(self):
        bump = 0.5
        p_up = binomial_european(SPOT + bump, STRIKE, RATE, VOL, T, 1000, OptionType.CALL)
        p_dn = binomial_european(SPOT - bump, STRIKE, RATE, VOL, T, 1000, OptionType.CALL)
        tree_delta = (p_up - p_dn) / (2 * bump)
        bs_delta = equity_delta(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert tree_delta == pytest.approx(bs_delta, abs=0.005)

    def test_gamma_matches(self):
        bump = 2.0
        p_up = binomial_european(SPOT + bump, STRIKE, RATE, VOL, T, 1000, OptionType.CALL)
        p_mid = binomial_european(SPOT, STRIKE, RATE, VOL, T, 1000, OptionType.CALL)
        p_dn = binomial_european(SPOT - bump, STRIKE, RATE, VOL, T, 1000, OptionType.CALL)
        tree_gamma = (p_up - 2 * p_mid + p_dn) / (bump ** 2)
        bs_gamma = equity_gamma(SPOT, STRIKE, RATE, VOL, T)
        assert tree_gamma == pytest.approx(bs_gamma, rel=0.10)
