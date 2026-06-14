"""Regression for L2 T4 audit of `options.asian.geometric_asian_analytical`:

Pre-fix the discrete geometric-vol formula was

    σ_g² = σ² · (2n+1) / (6(n+1))

which is the variance of ln(G) when G is averaged over n+1 monitoring
points INCLUDING the deterministic t_0 = 0.  But the function's drift
formula and the MC counterpart (``mc_asian_arithmetic`` line 100,
``paths[:, 1:]``) use only n RANDOM points (t_1..t_n).  These two
conventions give different σ_g and biased the control-variate
adjustment by ~7.7% for n=12.

Correct formula for n random monitoring points (Kemna-Vorst):

    σ_g² = σ² · (n+1)(2n+1) / (6·n²)

We verify both directly (closed-form vs analytical derivation) and
indirectly (MC at high n_paths should match the analytical price for
the geometric case to within MC standard error).
"""

from __future__ import annotations

import math

import pytest

from pricebook.options.asian import (
    geometric_asian_analytical,
    mc_asian_arithmetic,
)
from pricebook.models.black76 import OptionType


class TestGeometricAsianFormula:
    def test_vol_g_matches_kemna_vorst(self):
        """For n=12 monitoring points, σ_g must satisfy
        σ_g² = σ² · (n+1)(2n+1) / (6n²)."""
        spot, K, r, vol, T, q, n = 100.0, 100.0, 0.04, 0.30, 1.0, 0.02, 12

        # Compute the analytical price and back out the implied vol_g
        # from the Black-76 result.  Easier: build the closed form
        # explicitly and compare prices.
        from pricebook.models.black76 import black76_price

        # Correct σ_g (Kemna-Vorst for n monitoring points).
        sigma_g = vol * math.sqrt((n + 1) * (2 * n + 1) / (6 * n * n))
        mu = r - q
        drift_g = (mu - 0.5 * vol ** 2) * (n + 1) / (2 * n) + 0.5 * sigma_g ** 2
        forward_g = spot * math.exp(drift_g * T)
        df = math.exp(-r * T)
        expected = black76_price(forward_g, K, sigma_g, T, df, OptionType.CALL)

        actual = geometric_asian_analytical(spot, K, r, vol, T, n, OptionType.CALL, q)
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_continuous_limit_matches(self):
        """As n → ∞, σ_g → σ / √3 (continuous monitoring)."""
        spot, K, r, vol, T, q = 100.0, 100.0, 0.05, 0.20, 1.0, 0.00
        n = 10_000
        price = geometric_asian_analytical(spot, K, r, vol, T, n, OptionType.CALL, q)
        # Continuous geometric Asian closed form: σ_g → σ/√3, drift adjusted.
        sigma_g_cont = vol / math.sqrt(3.0)
        drift_g_cont = (r - q - 0.5 * vol ** 2) * 0.5 + 0.5 * sigma_g_cont ** 2
        forward_cont = spot * math.exp(drift_g_cont * T)
        df = math.exp(-r * T)
        from pricebook.models.black76 import black76_price
        expected_cont = black76_price(forward_cont, K, sigma_g_cont, T, df, OptionType.CALL)
        assert price == pytest.approx(expected_cont, rel=1e-3)


class TestMCAgreesWithAnalytical:
    def test_mc_geometric_matches_analytical(self):
        """High-paths MC of the geometric Asian should match the
        analytical formula to within MC standard error.  Pre-fix the
        formula was 7.7% too low in σ_g, so the analytical price was
        below the MC geom mean by several %; post-fix they agree."""
        spot, K, r, vol, T, q, n = 100.0, 100.0, 0.04, 0.25, 1.0, 0.0, 12

        analytical = geometric_asian_analytical(
            spot, K, r, vol, T, n, OptionType.CALL, q,
        )

        # MC geom payoff at the same monitoring points.
        from pricebook.models.gbm import GBMGenerator
        from pricebook.statistics.rng import PseudoRandom
        import numpy as np

        gen = GBMGenerator(spot=spot, rate=r, vol=vol, div_yield=q)
        rng = PseudoRandom(seed=42)
        paths = gen.generate(T=T, n_steps=n, n_paths=200_000, rng=rng)
        monitoring = paths[:, 1:]  # t_1..t_n (matches MC convention)
        geom_avg = np.exp(np.log(monitoring).mean(axis=1))
        payoffs = np.maximum(geom_avg - K, 0.0)
        df = math.exp(-r * T)
        mc_price = float((df * payoffs).mean())
        mc_se = float((df * payoffs).std(ddof=1) / math.sqrt(len(payoffs)))

        # Must agree within 3 standard errors (~99.7% confidence).
        assert abs(analytical - mc_price) < 3.0 * mc_se
