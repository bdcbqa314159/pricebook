"""Tests for Asian option pricing."""

import pytest
import math

from pricebook.asian import (
    geometric_asian_analytical,
    mc_asian_arithmetic,
)
from pricebook.mc_pricer import MCResult
from pricebook.black76 import OptionType


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
N_STEPS = 50


class TestGeometricAnalytical:
    def test_positive_price(self):
        p = geometric_asian_analytical(SPOT, STRIKE, RATE, VOL, T, N_STEPS)
        assert p > 0

    def test_call_less_than_european(self):
        """Asian option is worth less than European (averaging reduces vol)."""
        from pricebook.black76 import black76_price
        forward = SPOT * math.exp(RATE * T)
        df = math.exp(-RATE * T)
        european = black76_price(forward, STRIKE, VOL, T, df, OptionType.CALL)
        asian = geometric_asian_analytical(SPOT, STRIKE, RATE, VOL, T, N_STEPS)
        assert asian < european

    def test_put_positive(self):
        p = geometric_asian_analytical(SPOT, 110.0, RATE, VOL, T, N_STEPS, OptionType.PUT)
        assert p > 0

    def test_higher_vol_higher_price(self):
        p_low = geometric_asian_analytical(SPOT, STRIKE, RATE, 0.10, T, N_STEPS)
        p_high = geometric_asian_analytical(SPOT, STRIKE, RATE, 0.30, T, N_STEPS)
        assert p_high > p_low


class TestArithmeticMC:
    def test_call_positive(self):
        result = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                     n_paths=50_000)
        assert result.price > 0
        assert result.std_error > 0

    def test_put_positive_otm(self):
        result = mc_asian_arithmetic(SPOT, 90.0, RATE, VOL, T, N_STEPS,
                                     OptionType.PUT, n_paths=50_000)
        assert result.price > 0

    def test_arithmetic_geq_geometric(self):
        """Arithmetic average >= geometric average, so arithmetic Asian >= geometric."""
        arith = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                     n_paths=100_000)
        geom = geometric_asian_analytical(SPOT, STRIKE, RATE, VOL, T, N_STEPS)
        # Arithmetic should be >= geometric (with some MC noise)
        assert arith.price > geom - 3 * arith.std_error

    def test_less_than_european(self):
        """Arithmetic Asian call is worth less than European call."""
        from pricebook.black76 import black76_price
        forward = SPOT * math.exp(RATE * T)
        df = math.exp(-RATE * T)
        european = black76_price(forward, STRIKE, VOL, T, df, OptionType.CALL)
        asian = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                     n_paths=100_000)
        assert asian.price < european + 3 * asian.std_error


class TestFloatingStrike:
    def test_floating_call_positive(self):
        result = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                     OptionType.CALL, n_paths=50_000,
                                     floating_strike=True)
        assert result.price > 0

    def test_floating_put_positive(self):
        result = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                     OptionType.PUT, n_paths=50_000,
                                     floating_strike=True)
        assert result.price > 0


class TestVarianceReduction:
    def test_antithetic_reduces_error(self):
        plain = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                     n_paths=20_000, seed=42)
        anti = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                    n_paths=20_000, seed=42, antithetic=True)
        # Antithetic doubles paths → should reduce error
        assert anti.std_error < plain.std_error

    def test_control_variate_reduces_error(self):
        plain = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                     n_paths=50_000, seed=42)
        cv = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                  n_paths=50_000, seed=42, control_variate=True)
        assert cv.std_error < plain.std_error

    def test_both_reductions(self):
        result = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                      n_paths=50_000, seed=42,
                                      antithetic=True, control_variate=True)
        analytical_geom = geometric_asian_analytical(SPOT, STRIKE, RATE, VOL, T, N_STEPS)
        # Price should be reasonable (within a few percent of geometric)
        assert abs(result.price - analytical_geom) / analytical_geom < 0.15


class TestGeometricMCMatchesAnalytical:
    def test_geometric_mc_vs_analytical(self):
        """MC geometric average should match the closed-form."""
        from pricebook.gbm import GBMGenerator
        from pricebook.rng import PseudoRandom
        import numpy as np

        gen = GBMGenerator(spot=SPOT, rate=RATE, vol=VOL)
        rng = PseudoRandom(seed=42)
        paths = gen.generate(T=T, n_steps=N_STEPS, n_paths=200_000, rng=rng)

        monitoring = paths[:, 1:]
        geom_avg = np.exp(np.log(monitoring).mean(axis=1))
        payoffs = np.maximum(geom_avg - STRIKE, 0.0)
        df = math.exp(-RATE * T)
        mc_price = float((df * payoffs).mean())

        analytical = geometric_asian_analytical(SPOT, STRIKE, RATE, VOL, T, N_STEPS)
        assert mc_price == pytest.approx(analytical, rel=0.03)
