"""Tests for CRR binomial tree pricing."""

import pytest
import math

from pricebook.binomial_tree import binomial_european, binomial_american
from pricebook.equity_option import equity_option_price
from pricebook.black76 import OptionType


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
N = 500


class TestEuropeanTree:
    def test_call_converges_to_bs(self):
        tree = binomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert tree == pytest.approx(bs, rel=0.002)

    def test_put_converges_to_bs(self):
        tree = binomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        assert tree == pytest.approx(bs, rel=0.002)

    def test_otm_call(self):
        tree = binomial_european(SPOT, 120.0, RATE, VOL, T, N, OptionType.CALL)
        bs = equity_option_price(SPOT, 120.0, RATE, VOL, T, OptionType.CALL)
        assert tree == pytest.approx(bs, rel=0.005)

    def test_itm_put(self):
        tree = binomial_european(SPOT, 80.0, RATE, VOL, T, N, OptionType.PUT)
        bs = equity_option_price(SPOT, 80.0, RATE, VOL, T, OptionType.PUT)
        assert tree == pytest.approx(bs, rel=0.005)

    def test_with_dividend(self):
        q = 0.03
        tree = binomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL, q)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        assert tree == pytest.approx(bs, rel=0.002)

    def test_convergence_improves_with_steps(self):
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        err_100 = abs(binomial_european(SPOT, STRIKE, RATE, VOL, T, 100, OptionType.CALL) - bs)
        err_500 = abs(binomial_european(SPOT, STRIKE, RATE, VOL, T, 500, OptionType.CALL) - bs)
        assert err_500 < err_100

    def test_put_call_parity(self):
        c = binomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        p = binomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        expected = SPOT - STRIKE * math.exp(-RATE * T)
        assert c - p == pytest.approx(expected, rel=0.005)


class TestAmericanTree:
    def test_american_call_no_div_equals_european(self):
        """American call without dividends = European call (no early exercise)."""
        am = binomial_american(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        eu = binomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        assert am == pytest.approx(eu, rel=1e-10)

    def test_american_put_geq_european(self):
        """American put >= European put (early exercise premium)."""
        am = binomial_american(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        eu = binomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        assert am >= eu - 1e-10

    def test_american_put_premium_exists(self):
        """Deep ITM American put should have early exercise premium."""
        am = binomial_american(SPOT, 130.0, RATE, VOL, T, N, OptionType.PUT)
        eu = binomial_european(SPOT, 130.0, RATE, VOL, T, N, OptionType.PUT)
        assert am > eu + 0.01

    def test_american_call_with_dividend_geq_european(self):
        """American call with dividends >= European (early exercise may be optimal)."""
        q = 0.05
        am = binomial_american(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL, q)
        eu = binomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL, q)
        assert am >= eu - 1e-10

    def test_american_put_bounded(self):
        """American put is bounded: K - S <= P_am <= K."""
        am = binomial_american(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        assert am <= STRIKE
        assert am >= max(STRIKE - SPOT, 0)

    def test_american_call_bounded(self):
        """American call is bounded: S - K*exp(-rT) <= C_am <= S."""
        am = binomial_american(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        assert am <= SPOT
        assert am >= max(SPOT - STRIKE * math.exp(-RATE * T), 0)


class TestGreeksViaBump:
    def test_delta_positive_for_call(self):
        bump = 0.5
        p_up = binomial_american(SPOT + bump, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        p_dn = binomial_american(SPOT - bump, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        delta = (p_up - p_dn) / (2 * bump)
        assert 0 < delta < 1

    def test_delta_negative_for_put(self):
        bump = 0.5
        p_up = binomial_american(SPOT + bump, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        p_dn = binomial_american(SPOT - bump, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        delta = (p_up - p_dn) / (2 * bump)
        assert -1 < delta < 0

    def test_gamma_positive(self):
        bump = 0.5
        p_up = binomial_american(SPOT + bump, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        p_mid = binomial_american(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        p_dn = binomial_american(SPOT - bump, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        gamma = (p_up - 2 * p_mid + p_dn) / (bump ** 2)
        assert gamma > 0
