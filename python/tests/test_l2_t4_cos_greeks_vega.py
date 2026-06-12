"""Regression for L2 Wave-2 audit — `cos_greeks` vega magnitude.

Pre-fix `_cos_vega` added `δσ²·T` (only the QUADRATIC term) to the
log-return variance, missing the dominant linear `2σΔσT` term that
comes from `(σ+Δσ)² − σ² = 2σΔσ + Δσ²`.

For σ=20%, Δσ=1%:
    pre-fix ΔVar = (0.01)² · T = 1e-4 · T
    correct ΔVar = (2 × 0.20 × 0.01 + 1e-4) · T ≈ 4.1e-3 · T

So the CF perturbation was ≈ 41× too small and the reported vega was
≈ 30× too small (≈ 0.012 vs analytical 0.376 for ATM call).

Also the drift correction was missing — bumping σ in a BS CF requires
adjusting BOTH the drift (μ → μ - σΔσT - 0.5Δσ²T) and the variance.

Post-fix infers σ_implied from the CF via the second cumulant and
applies both the drift and variance shifts.
"""

from __future__ import annotations

import math

import pytest

from pricebook.models.black76 import OptionType
from pricebook.models.cos_method import bs_char_func
from pricebook.models.fourier_greeks import cos_greeks


class TestCOSVegaMagnitude:
    @pytest.mark.parametrize("K,sigma", [
        (100.0, 0.20),    # ATM, normal vol
        (80.0, 0.20),     # ITM
        (120.0, 0.20),    # OTM
        (100.0, 0.10),    # low vol
        (100.0, 0.40),    # high vol
    ])
    def test_vega_matches_bs_analytical(self, K, sigma):
        """For a BS char func, cos_greeks vega should match analytical
        BS vega to <10% (across moneyness and vol regimes).

        Pre-fix the vega was ~30× too small everywhere; the missing
        drift correction would also flip the answer to ~35% too high
        for the variance-only fix.  Post-fix matches analytical."""
        S, r, T = 100.0, 0.05, 1.0
        cf = bs_char_func(r, 0.0, sigma, T)
        res = cos_greeks(cf, S, K, r, T,
                         option_type=OptionType.CALL, div_yield=0.0)

        # Analytical BS vega per 1% vol bump.
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        phi_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        bs_vega_per_pct = S * phi_d1 * math.sqrt(T) * 0.01

        rel = abs(res.vega - bs_vega_per_pct) / max(bs_vega_per_pct, 1e-10)
        assert rel < 0.10, (
            f"K={K}, σ={sigma}: cos_vega={res.vega:.4f}, "
            f"BS={bs_vega_per_pct:.4f}, rel={rel:.3%}"
        )

    def test_zero_vega_at_T_zero(self):
        """At T=0 there's no time to be wrong — vega should be ~0.
        Smoke test (avoids divide-by-zero in σ_implied extraction)."""
        cf = bs_char_func(0.05, 0.0, 0.20, 1e-9)
        res = cos_greeks(cf, 100.0, 100.0, 0.05, 1e-9,
                         option_type=OptionType.CALL, div_yield=0.0)
        # Just check it runs and returns finite.
        assert math.isfinite(res.vega)
