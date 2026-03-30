"""Tests for Monte Carlo European option pricer."""

import pytest
import math

from pricebook.mc_pricer import mc_european, MCResult
from pricebook.black76 import OptionType, black76_price


# Analytical Black-Scholes price for cross-checking
def bs_price(spot, strike, rate, vol, T, option_type, div_yield=0.0):
    forward = spot * math.exp((rate - div_yield) * T)
    df = math.exp(-rate * T)
    return black76_price(forward, strike, vol, T, df, option_type)


SPOT, STRIKE, RATE, VOL, T = 100.0, 105.0, 0.05, 0.20, 1.0


class TestMCEuropeanCall:
    def test_call_matches_analytical(self):
        result = mc_european(SPOT, STRIKE, RATE, VOL, T,
                             OptionType.CALL, n_paths=200_000)
        analytical = bs_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert result.price == pytest.approx(analytical, rel=0.03)

    def test_call_within_confidence_interval(self):
        result = mc_european(SPOT, STRIKE, RATE, VOL, T,
                             OptionType.CALL, n_paths=200_000)
        analytical = bs_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        # Analytical should be within 3 standard errors
        assert abs(result.price - analytical) < 3 * result.std_error

    def test_result_has_std_error(self):
        result = mc_european(SPOT, STRIKE, RATE, VOL, T, n_paths=10_000)
        assert result.std_error > 0
        assert result.n_paths == 10_000


class TestMCEuropeanPut:
    def test_put_matches_analytical(self):
        result = mc_european(SPOT, STRIKE, RATE, VOL, T,
                             OptionType.PUT, n_paths=200_000)
        analytical = bs_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        assert result.price == pytest.approx(analytical, rel=0.03)

    def test_put_call_parity(self):
        call = mc_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                           n_paths=200_000, seed=42)
        put = mc_european(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                          n_paths=200_000, seed=42)
        df = math.exp(-RATE * T)
        parity = call.price - put.price - df * (SPOT * math.exp(RATE * T) - STRIKE)
        assert parity == pytest.approx(0.0, abs=0.5)


class TestAntithetic:
    def test_antithetic_doubles_paths(self):
        result = mc_european(SPOT, STRIKE, RATE, VOL, T,
                             n_paths=10_000, antithetic=True)
        assert result.n_paths == 20_000

    def test_antithetic_reduces_error(self):
        plain = mc_european(SPOT, STRIKE, RATE, VOL, T,
                            n_paths=50_000, seed=42, antithetic=False)
        anti = mc_european(SPOT, STRIKE, RATE, VOL, T,
                           n_paths=50_000, seed=42, antithetic=True)
        # Antithetic should have lower std error (for same base n_paths)
        # anti has 100K effective paths vs plain 50K, so error should be lower
        assert anti.std_error < plain.std_error

    def test_antithetic_matches_analytical(self):
        result = mc_european(SPOT, STRIKE, RATE, VOL, T,
                             OptionType.CALL, n_paths=100_000, antithetic=True)
        analytical = bs_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert abs(result.price - analytical) < 3 * result.std_error


class TestControlVariate:
    def test_control_variate_reduces_error(self):
        plain = mc_european(SPOT, STRIKE, RATE, VOL, T,
                            n_paths=50_000, seed=42, control_variate=False)
        cv = mc_european(SPOT, STRIKE, RATE, VOL, T,
                         n_paths=50_000, seed=42, control_variate=True)
        assert cv.std_error < plain.std_error

    def test_control_variate_matches_analytical(self):
        result = mc_european(SPOT, STRIKE, RATE, VOL, T,
                             OptionType.CALL, n_paths=50_000, control_variate=True)
        analytical = bs_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert abs(result.price - analytical) < 3 * result.std_error

    def test_both_reductions_combined(self):
        result = mc_european(SPOT, STRIKE, RATE, VOL, T,
                             OptionType.CALL, n_paths=50_000,
                             antithetic=True, control_variate=True)
        analytical = bs_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert abs(result.price - analytical) < 3 * result.std_error


class TestWithDividend:
    def test_dividend_call(self):
        q = 0.02
        result = mc_european(SPOT, STRIKE, RATE, VOL, T,
                             OptionType.CALL, div_yield=q, n_paths=200_000)
        analytical = bs_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        assert result.price == pytest.approx(analytical, rel=0.03)


class TestConvergence:
    def test_more_paths_lower_error(self):
        r1 = mc_european(SPOT, STRIKE, RATE, VOL, T, n_paths=10_000, seed=42)
        r2 = mc_european(SPOT, STRIKE, RATE, VOL, T, n_paths=40_000, seed=42)
        # Doubling paths should roughly halve the std error
        # 4x paths → 2x reduction
        assert r2.std_error < r1.std_error * 0.7
