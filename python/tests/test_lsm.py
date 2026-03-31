"""Tests for Longstaff-Schwartz Monte Carlo."""

import pytest
import math

from pricebook.lsm import lsm_american
from pricebook.binomial_tree import binomial_american, binomial_european
from pricebook.mc_pricer import mc_european, MCResult
from pricebook.black76 import OptionType


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0


class TestLSMPut:
    def test_positive(self):
        r = lsm_american(SPOT, STRIKE, RATE, VOL, T, n_paths=50_000)
        assert r.price > 0

    def test_returns_mc_result(self):
        r = lsm_american(SPOT, STRIKE, RATE, VOL, T, n_paths=10_000)
        assert isinstance(r, MCResult)
        assert r.std_error > 0

    def test_matches_binomial(self):
        """LSM American put ≈ binomial tree."""
        lsm = lsm_american(SPOT, STRIKE, RATE, VOL, T,
                            n_steps=50, n_paths=200_000)
        tree = binomial_american(SPOT, STRIKE, RATE, VOL, T, 500, OptionType.PUT)
        assert lsm.price == pytest.approx(tree, rel=0.03)

    def test_geq_european(self):
        """American put ≥ European put."""
        am = lsm_american(SPOT, STRIKE, RATE, VOL, T, n_paths=100_000)
        eu = mc_european(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT, n_paths=100_000)
        assert am.price >= eu.price - 3 * am.std_error

    def test_deep_itm_has_early_exercise(self):
        """Deep ITM put: American > European."""
        am = lsm_american(SPOT, 140.0, RATE, VOL, T, n_paths=100_000)
        eu = mc_european(SPOT, 140.0, RATE, VOL, T, OptionType.PUT, n_paths=100_000)
        assert am.price > eu.price


class TestLSMCall:
    def test_no_div_equals_european(self):
        """American call without dividends ≈ European call."""
        am = lsm_american(SPOT, STRIKE, RATE, VOL, T,
                          option_type=OptionType.CALL, n_paths=100_000)
        eu = mc_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, n_paths=100_000)
        assert am.price == pytest.approx(eu.price, rel=0.05)

    def test_with_dividend(self):
        """American call with high dividend should exceed European."""
        q = 0.08
        am = lsm_american(SPOT, STRIKE, RATE, VOL, T,
                          option_type=OptionType.CALL, div_yield=q, n_paths=100_000)
        eu = mc_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                         div_yield=q, n_paths=100_000)
        assert am.price >= eu.price - 3 * am.std_error


class TestConvergence:
    def test_more_paths_lower_error(self):
        r1 = lsm_american(SPOT, STRIKE, RATE, VOL, T, n_paths=10_000)
        r2 = lsm_american(SPOT, STRIKE, RATE, VOL, T, n_paths=40_000)
        assert r2.std_error < r1.std_error * 0.7

    def test_more_steps_stable(self):
        """More exercise dates shouldn't change price much."""
        r1 = lsm_american(SPOT, STRIKE, RATE, VOL, T, n_steps=20, n_paths=50_000)
        r2 = lsm_american(SPOT, STRIKE, RATE, VOL, T, n_steps=100, n_paths=50_000)
        assert r1.price == pytest.approx(r2.price, rel=0.05)
