"""Tests for trinomial tree pricing."""

import pytest
import math

from pricebook.trinomial_tree import trinomial_european, trinomial_american
from pricebook.binomial_tree import binomial_european, binomial_american
from pricebook.equity_option import equity_option_price
from pricebook.black76 import OptionType


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
N = 300


class TestEuropeanTrinomial:
    def test_call_matches_bs(self):
        tri = trinomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert tri == pytest.approx(bs, rel=0.002)

    def test_put_matches_bs(self):
        tri = trinomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        assert tri == pytest.approx(bs, rel=0.002)

    def test_otm_call(self):
        tri = trinomial_european(SPOT, 120.0, RATE, VOL, T, N, OptionType.CALL)
        bs = equity_option_price(SPOT, 120.0, RATE, VOL, T, OptionType.CALL)
        assert tri == pytest.approx(bs, rel=0.005)

    def test_with_dividend(self):
        q = 0.03
        tri = trinomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL, q)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        assert tri == pytest.approx(bs, rel=0.002)

    def test_faster_convergence_than_binomial(self):
        """Trinomial should converge with fewer steps than CRR."""
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        tri_100 = trinomial_european(SPOT, STRIKE, RATE, VOL, T, 100, OptionType.CALL)
        bin_100 = binomial_european(SPOT, STRIKE, RATE, VOL, T, 100, OptionType.CALL)
        assert abs(tri_100 - bs) <= abs(bin_100 - bs) + 0.01

    def test_put_call_parity(self):
        c = trinomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        p = trinomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        expected = SPOT - STRIKE * math.exp(-RATE * T)
        assert c - p == pytest.approx(expected, rel=0.005)


class TestAmericanTrinomial:
    def test_call_no_div_equals_european(self):
        am = trinomial_american(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        eu = trinomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        assert am == pytest.approx(eu, rel=1e-6)

    def test_put_geq_european(self):
        am = trinomial_american(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        eu = trinomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        assert am >= eu - 0.01

    def test_put_matches_binomial(self):
        am_tri = trinomial_american(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        am_bin = binomial_american(SPOT, STRIKE, RATE, VOL, T, 500, OptionType.PUT)
        assert am_tri == pytest.approx(am_bin, rel=0.005)

    def test_deep_itm_put_early_exercise(self):
        am = trinomial_american(SPOT, 150.0, RATE, VOL, T, N, OptionType.PUT)
        eu = trinomial_european(SPOT, 150.0, RATE, VOL, T, N, OptionType.PUT)
        assert am > eu + 0.01

    def test_call_with_dividend(self):
        q = 0.08
        am = trinomial_american(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL, q)
        eu = trinomial_european(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL, q)
        assert am >= eu - 0.01


class TestGreeksViaBump:
    def test_delta_positive_call(self):
        bump = 0.5
        p_up = trinomial_european(SPOT + bump, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        p_dn = trinomial_european(SPOT - bump, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        delta = (p_up - p_dn) / (2 * bump)
        assert 0 < delta < 1

    def test_delta_negative_put(self):
        bump = 0.5
        p_up = trinomial_american(SPOT + bump, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        p_dn = trinomial_american(SPOT - bump, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        delta = (p_up - p_dn) / (2 * bump)
        assert -1 < delta < 0
