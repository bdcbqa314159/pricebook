"""Tests for trinomial tree pricing."""

import pytest
import math

from pricebook.numerical._trees import solve_tree, TreeMethod, ExerciseType
from pricebook.options.equity_option import equity_option_price
from pricebook.models.black76 import OptionType


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
N = 300


def _tri_eu(spot, strike, rate, vol, T, n, opt_type=OptionType.CALL, q=0.0):
    is_call = str(getattr(opt_type, 'value', opt_type)).lower() != "put"
    return solve_tree(spot, strike, rate, vol, T, TreeMethod.TRINOMIAL, n,
                      ExerciseType.EUROPEAN, is_call=is_call, div_yield=q).price


def _tri_am(spot, strike, rate, vol, T, n, opt_type=OptionType.CALL, q=0.0):
    is_call = str(getattr(opt_type, 'value', opt_type)).lower() != "put"
    return solve_tree(spot, strike, rate, vol, T, TreeMethod.TRINOMIAL, n,
                      ExerciseType.AMERICAN, is_call=is_call, div_yield=q).price


def _crr_eu(spot, strike, rate, vol, T, n, opt_type=OptionType.CALL, q=0.0):
    is_call = str(getattr(opt_type, 'value', opt_type)).lower() != "put"
    return solve_tree(spot, strike, rate, vol, T, TreeMethod.CRR, n,
                      ExerciseType.EUROPEAN, is_call=is_call, div_yield=q).price


def _crr_am(spot, strike, rate, vol, T, n, opt_type=OptionType.CALL, q=0.0):
    is_call = str(getattr(opt_type, 'value', opt_type)).lower() != "put"
    return solve_tree(spot, strike, rate, vol, T, TreeMethod.CRR, n,
                      ExerciseType.AMERICAN, is_call=is_call, div_yield=q).price


class TestEuropeanTrinomial:
    def test_call_matches_bs(self):
        tri = _tri_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert tri == pytest.approx(bs, rel=0.002)

    def test_put_matches_bs(self):
        tri = _tri_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        assert tri == pytest.approx(bs, rel=0.002)

    def test_otm_call(self):
        tri = _tri_eu(SPOT, 120.0, RATE, VOL, T, N, OptionType.CALL)
        bs = equity_option_price(SPOT, 120.0, RATE, VOL, T, OptionType.CALL)
        assert tri == pytest.approx(bs, rel=0.005)

    def test_with_dividend(self):
        q = 0.03
        tri = _tri_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL, q)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        assert tri == pytest.approx(bs, rel=0.002)

    def test_faster_convergence_than_binomial(self):
        """Trinomial should converge with fewer steps than CRR."""
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        tri_100 = _tri_eu(SPOT, STRIKE, RATE, VOL, T, 100, OptionType.CALL)
        bin_100 = _crr_eu(SPOT, STRIKE, RATE, VOL, T, 100, OptionType.CALL)
        assert abs(tri_100 - bs) <= abs(bin_100 - bs) + 0.01

    def test_put_call_parity(self):
        c = _tri_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        p = _tri_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        expected = SPOT - STRIKE * math.exp(-RATE * T)
        assert c - p == pytest.approx(expected, rel=0.005)


class TestAmericanTrinomial:
    def test_call_no_div_equals_european(self):
        am = _tri_am(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        eu = _tri_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        assert am == pytest.approx(eu, rel=1e-6)

    def test_put_geq_european(self):
        am = _tri_am(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        eu = _tri_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        assert am >= eu - 0.01

    def test_put_matches_binomial(self):
        am_tri = _tri_am(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        am_bin = _crr_am(SPOT, STRIKE, RATE, VOL, T, 500, OptionType.PUT)
        assert am_tri == pytest.approx(am_bin, rel=0.005)

    def test_deep_itm_put_early_exercise(self):
        am = _tri_am(SPOT, 150.0, RATE, VOL, T, N, OptionType.PUT)
        eu = _tri_eu(SPOT, 150.0, RATE, VOL, T, N, OptionType.PUT)
        assert am > eu + 0.01

    def test_call_with_dividend(self):
        q = 0.08
        am = _tri_am(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL, q)
        eu = _tri_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL, q)
        assert am >= eu - 0.01


class TestGreeksViaBump:
    def test_delta_positive_call(self):
        bump = 0.5
        p_up = _tri_eu(SPOT + bump, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        p_dn = _tri_eu(SPOT - bump, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        delta = (p_up - p_dn) / (2 * bump)
        assert 0 < delta < 1

    def test_delta_negative_put(self):
        bump = 0.5
        p_up = _tri_am(SPOT + bump, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        p_dn = _tri_am(SPOT - bump, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        delta = (p_up - p_dn) / (2 * bump)
        assert -1 < delta < 0
