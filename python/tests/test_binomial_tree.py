"""Tests for CRR binomial tree pricing."""

import pytest
import math

from pricebook.numerical._trees import solve_tree, TreeMethod, ExerciseType
from pricebook.options.equity_option import equity_option_price
from pricebook.models.black76 import OptionType


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
N = 500


def _crr_eu(spot, strike, rate, vol, T, n, opt_type=OptionType.CALL, q=0.0):
    is_call = str(getattr(opt_type, 'value', opt_type)).lower() != "put"
    return solve_tree(spot, strike, rate, vol, T, TreeMethod.CRR, n,
                      ExerciseType.EUROPEAN, is_call=is_call, div_yield=q).price


def _crr_am(spot, strike, rate, vol, T, n, opt_type=OptionType.CALL, q=0.0):
    is_call = str(getattr(opt_type, 'value', opt_type)).lower() != "put"
    return solve_tree(spot, strike, rate, vol, T, TreeMethod.CRR, n,
                      ExerciseType.AMERICAN, is_call=is_call, div_yield=q).price


class TestEuropeanTree:
    def test_call_converges_to_bs(self):
        tree = _crr_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert tree == pytest.approx(bs, rel=0.002)

    def test_put_converges_to_bs(self):
        tree = _crr_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        assert tree == pytest.approx(bs, rel=0.002)

    def test_otm_call(self):
        tree = _crr_eu(SPOT, 120.0, RATE, VOL, T, N, OptionType.CALL)
        bs = equity_option_price(SPOT, 120.0, RATE, VOL, T, OptionType.CALL)
        assert tree == pytest.approx(bs, rel=0.005)

    def test_itm_put(self):
        tree = _crr_eu(SPOT, 80.0, RATE, VOL, T, N, OptionType.PUT)
        bs = equity_option_price(SPOT, 80.0, RATE, VOL, T, OptionType.PUT)
        assert tree == pytest.approx(bs, rel=0.005)

    def test_with_dividend(self):
        q = 0.03
        tree = _crr_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL, q)
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        assert tree == pytest.approx(bs, rel=0.002)

    def test_convergence_improves_with_steps(self):
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        err_100 = abs(_crr_eu(SPOT, STRIKE, RATE, VOL, T, 100, OptionType.CALL) - bs)
        err_500 = abs(_crr_eu(SPOT, STRIKE, RATE, VOL, T, 500, OptionType.CALL) - bs)
        assert err_500 < err_100

    def test_put_call_parity(self):
        c = _crr_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        p = _crr_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        expected = SPOT - STRIKE * math.exp(-RATE * T)
        assert c - p == pytest.approx(expected, rel=0.005)


class TestAmericanTree:
    def test_american_call_no_div_equals_european(self):
        """American call without dividends = European call (no early exercise)."""
        am = _crr_am(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        eu = _crr_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        assert am == pytest.approx(eu, rel=1e-10)

    def test_american_put_geq_european(self):
        """American put >= European put (early exercise premium)."""
        am = _crr_am(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        eu = _crr_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        assert am >= eu - 1e-10

    def test_american_put_premium_exists(self):
        """Deep ITM American put should have early exercise premium."""
        am = _crr_am(SPOT, 130.0, RATE, VOL, T, N, OptionType.PUT)
        eu = _crr_eu(SPOT, 130.0, RATE, VOL, T, N, OptionType.PUT)
        assert am > eu + 0.01

    def test_american_call_with_dividend_geq_european(self):
        """American call with dividends >= European (early exercise may be optimal)."""
        q = 0.05
        am = _crr_am(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL, q)
        eu = _crr_eu(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL, q)
        assert am >= eu - 1e-10

    def test_american_put_bounded(self):
        """American put is bounded: K - S <= P_am <= K."""
        am = _crr_am(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        assert am <= STRIKE
        assert am >= max(STRIKE - SPOT, 0)

    def test_american_call_bounded(self):
        """American call is bounded: S - K*exp(-rT) <= C_am <= S."""
        am = _crr_am(SPOT, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        assert am <= SPOT
        assert am >= max(SPOT - STRIKE * math.exp(-RATE * T), 0)


class TestGreeksViaBump:
    def test_delta_positive_for_call(self):
        bump = 0.5
        p_up = _crr_am(SPOT + bump, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        p_dn = _crr_am(SPOT - bump, STRIKE, RATE, VOL, T, N, OptionType.CALL)
        delta = (p_up - p_dn) / (2 * bump)
        assert 0 < delta < 1

    def test_delta_negative_for_put(self):
        bump = 0.5
        p_up = _crr_am(SPOT + bump, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        p_dn = _crr_am(SPOT - bump, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        delta = (p_up - p_dn) / (2 * bump)
        assert -1 < delta < 0

    def test_gamma_positive(self):
        bump = 0.5
        p_up = _crr_am(SPOT + bump, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        p_mid = _crr_am(SPOT, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        p_dn = _crr_am(SPOT - bump, STRIKE, RATE, VOL, T, N, OptionType.PUT)
        gamma = (p_up - 2 * p_mid + p_dn) / (bump ** 2)
        assert gamma > 0
