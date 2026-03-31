"""Tests for finite difference option pricing."""

import pytest
import math

from pricebook.finite_difference import fd_european, fd_american, FDScheme
from pricebook.equity_option import equity_option_price
from pricebook.binomial_tree import binomial_american
from pricebook.black76 import OptionType


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
N_S, N_T = 300, 300


class TestEuropeanFD:
    def test_cn_call_matches_bs(self):
        fd = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                         n_spot=N_S, n_time=N_T, scheme="cn")
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert fd == pytest.approx(bs, rel=0.005)

    def test_cn_put_matches_bs(self):
        fd = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                         n_spot=N_S, n_time=N_T, scheme="cn")
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        assert fd == pytest.approx(bs, rel=0.005)

    def test_implicit_matches_bs(self):
        fd = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                         n_spot=N_S, n_time=N_T, scheme="implicit")
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert fd == pytest.approx(bs, rel=0.01)

    def test_explicit_matches_bs(self):
        # Explicit needs finer grid for stability
        fd = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                         n_spot=100, n_time=5000, scheme="explicit")
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert fd == pytest.approx(bs, rel=0.02)

    def test_otm_call(self):
        fd = fd_european(SPOT, 120.0, RATE, VOL, T, OptionType.CALL,
                         n_spot=N_S, n_time=N_T, scheme="cn")
        bs = equity_option_price(SPOT, 120.0, RATE, VOL, T, OptionType.CALL)
        assert fd == pytest.approx(bs, rel=0.01)

    def test_with_dividend(self):
        q = 0.03
        fd = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                         div_yield=q, n_spot=N_S, n_time=N_T, scheme="cn")
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        assert fd == pytest.approx(bs, rel=0.005)

    def test_put_call_parity(self):
        c = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                        n_spot=N_S, n_time=N_T, scheme="cn")
        p = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                        n_spot=N_S, n_time=N_T, scheme="cn")
        expected = SPOT - STRIKE * math.exp(-RATE * T)
        assert c - p == pytest.approx(expected, rel=0.01)

    def test_cn_more_accurate_than_implicit(self):
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        cn = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                         n_spot=200, n_time=200, scheme="cn")
        imp = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                          n_spot=200, n_time=200, scheme="implicit")
        assert abs(cn - bs) < abs(imp - bs)


class TestAmericanFD:
    def test_american_call_no_div_equals_european(self):
        am = fd_american(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                         n_spot=N_S, n_time=N_T)
        eu = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                         n_spot=N_S, n_time=N_T, scheme="cn")
        assert am == pytest.approx(eu, rel=0.005)

    def test_american_put_geq_european(self):
        am = fd_american(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                         n_spot=N_S, n_time=N_T)
        eu = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                         n_spot=N_S, n_time=N_T, scheme="cn")
        assert am >= eu - 0.01

    def test_american_put_matches_binomial(self):
        am_fd = fd_american(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                            n_spot=N_S, n_time=N_T)
        am_tree = binomial_american(SPOT, STRIKE, RATE, VOL, T, 500, OptionType.PUT)
        assert am_fd == pytest.approx(am_tree, rel=0.01)

    def test_deep_itm_put_early_exercise(self):
        am = fd_american(SPOT, 150.0, RATE, VOL, T, OptionType.PUT,
                         n_spot=N_S, n_time=N_T)
        eu = fd_european(SPOT, 150.0, RATE, VOL, T, OptionType.PUT,
                         n_spot=N_S, n_time=N_T, scheme="cn")
        assert am > eu + 0.1


class TestGreeksViaBump:
    def test_delta_positive_call(self):
        bump = 0.5
        p_up = fd_european(SPOT + bump, STRIKE, RATE, VOL, T, OptionType.CALL,
                           n_spot=N_S, n_time=N_T, scheme="cn")
        p_dn = fd_european(SPOT - bump, STRIKE, RATE, VOL, T, OptionType.CALL,
                           n_spot=N_S, n_time=N_T, scheme="cn")
        delta = (p_up - p_dn) / (2 * bump)
        assert 0 < delta < 1

    def test_vega_positive(self):
        bump = 0.001
        p_up = fd_european(SPOT, STRIKE, RATE, VOL + bump, T, OptionType.CALL,
                           n_spot=N_S, n_time=N_T, scheme="cn")
        p_dn = fd_european(SPOT, STRIKE, RATE, VOL - bump, T, OptionType.CALL,
                           n_spot=N_S, n_time=N_T, scheme="cn")
        vega = (p_up - p_dn) / (2 * bump)
        assert vega > 0
