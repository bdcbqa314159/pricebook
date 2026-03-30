"""
Slice 9 round-trip validation: Monte Carlo engine.

1. European MC matches Black-76 within confidence interval
2. Variance reduction reduces standard error
3. Asian geometric MC matches analytical formula
4. Convergence: doubling paths halves the standard error
"""

import pytest
import math

from pricebook.mc_pricer import mc_european
from pricebook.asian import geometric_asian_analytical, mc_asian_arithmetic
from pricebook.black76 import OptionType, black76_price


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0


def bs_price(spot, strike, rate, vol, T, opt, q=0.0):
    fwd = spot * math.exp((rate - q) * T)
    df = math.exp(-rate * T)
    return black76_price(fwd, strike, vol, T, df, opt)


class TestEuropeanMCvsAnalytical:
    """European MC matches Black-76 within confidence interval."""

    @pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
    def test_within_3_sigma(self, opt):
        result = mc_european(SPOT, STRIKE, RATE, VOL, T, opt, n_paths=200_000)
        analytical = bs_price(SPOT, STRIKE, RATE, VOL, T, opt)
        assert abs(result.price - analytical) < 3 * result.std_error

    @pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
    def test_antithetic_within_3_sigma(self, opt):
        result = mc_european(SPOT, STRIKE, RATE, VOL, T, opt,
                             n_paths=100_000, antithetic=True)
        analytical = bs_price(SPOT, STRIKE, RATE, VOL, T, opt)
        assert abs(result.price - analytical) < 3 * result.std_error

    @pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
    def test_control_variate_within_3_sigma(self, opt):
        result = mc_european(SPOT, STRIKE, RATE, VOL, T, opt,
                             n_paths=50_000, control_variate=True)
        analytical = bs_price(SPOT, STRIKE, RATE, VOL, T, opt)
        assert abs(result.price - analytical) < 3 * result.std_error

    def test_otm_call(self):
        """OTM call: strike well above spot."""
        result = mc_european(SPOT, 130.0, RATE, VOL, T, OptionType.CALL,
                             n_paths=200_000)
        analytical = bs_price(SPOT, 130.0, RATE, VOL, T, OptionType.CALL)
        assert abs(result.price - analytical) < 3 * result.std_error

    def test_itm_put(self):
        """ITM put: strike well above spot."""
        result = mc_european(SPOT, 130.0, RATE, VOL, T, OptionType.PUT,
                             n_paths=200_000)
        analytical = bs_price(SPOT, 130.0, RATE, VOL, T, OptionType.PUT)
        assert abs(result.price - analytical) < 3 * result.std_error


class TestVarianceReductionEffectiveness:
    """Variance reduction techniques reduce standard error."""

    def test_antithetic_vs_plain(self):
        plain = mc_european(SPOT, STRIKE, RATE, VOL, T, n_paths=50_000, seed=42)
        anti = mc_european(SPOT, STRIKE, RATE, VOL, T, n_paths=50_000, seed=42,
                           antithetic=True)
        assert anti.std_error < plain.std_error

    def test_control_variate_vs_plain(self):
        plain = mc_european(SPOT, STRIKE, RATE, VOL, T, n_paths=50_000, seed=42)
        cv = mc_european(SPOT, STRIKE, RATE, VOL, T, n_paths=50_000, seed=42,
                         control_variate=True)
        assert cv.std_error < plain.std_error

    def test_asian_cv_vs_plain(self):
        plain = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, 50,
                                     n_paths=50_000, seed=42)
        cv = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, 50,
                                  n_paths=50_000, seed=42, control_variate=True)
        assert cv.std_error < plain.std_error


class TestAsianGeometricConsistency:
    """MC geometric average matches analytical formula."""

    def test_geometric_mc_vs_analytical(self):
        from pricebook.gbm import GBMGenerator
        from pricebook.rng import PseudoRandom
        import numpy as np

        n_steps = 50
        gen = GBMGenerator(spot=SPOT, rate=RATE, vol=VOL)
        rng = PseudoRandom(seed=42)
        paths = gen.generate(T=T, n_steps=n_steps, n_paths=200_000, rng=rng)

        monitoring = paths[:, 1:]
        geom_avg = np.exp(np.log(monitoring).mean(axis=1))
        payoffs = np.maximum(geom_avg - STRIKE, 0.0)
        df = math.exp(-RATE * T)
        mc_price = float((df * payoffs).mean())
        mc_se = float((df * payoffs).std(ddof=1) / math.sqrt(200_000))

        analytical = geometric_asian_analytical(SPOT, STRIKE, RATE, VOL, T, n_steps)
        # Analytical is an approximation for discrete monitoring; allow ~2% relative
        assert mc_price == pytest.approx(analytical, rel=0.03)

    def test_geometric_put(self):
        from pricebook.gbm import GBMGenerator
        from pricebook.rng import PseudoRandom
        import numpy as np

        n_steps = 50
        gen = GBMGenerator(spot=SPOT, rate=RATE, vol=VOL)
        rng = PseudoRandom(seed=42)
        paths = gen.generate(T=T, n_steps=n_steps, n_paths=200_000, rng=rng)

        monitoring = paths[:, 1:]
        geom_avg = np.exp(np.log(monitoring).mean(axis=1))
        payoffs = np.maximum(STRIKE - geom_avg, 0.0)
        df = math.exp(-RATE * T)
        mc_price = float((df * payoffs).mean())
        mc_se = float((df * payoffs).std(ddof=1) / math.sqrt(200_000))

        analytical = geometric_asian_analytical(SPOT, STRIKE, RATE, VOL, T,
                                                 n_steps, OptionType.PUT)
        assert mc_price == pytest.approx(analytical, rel=0.03)


class TestConvergenceRate:
    """Doubling paths approximately halves the standard error."""

    def test_european_convergence(self):
        r1 = mc_european(SPOT, STRIKE, RATE, VOL, T, n_paths=10_000, seed=42)
        r2 = mc_european(SPOT, STRIKE, RATE, VOL, T, n_paths=40_000, seed=42)
        # 4x paths → ~2x error reduction (1/sqrt(4) = 0.5)
        ratio = r2.std_error / r1.std_error
        assert ratio < 0.65  # some slack for randomness

    def test_asian_convergence(self):
        r1 = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, 50,
                                  n_paths=10_000, seed=42)
        r2 = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, 50,
                                  n_paths=40_000, seed=42)
        ratio = r2.std_error / r1.std_error
        assert ratio < 0.65
