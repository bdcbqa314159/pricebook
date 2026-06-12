"""Regression for L2 Tier-3 T3.12 — COS method `c2` floor no longer destroys
low-variance pricing.

Pre-fix `cos_price` clamped the variance cumulant ``c2`` to a hard absolute
floor of 0.001 (`c2 = max(c2_estimated, 0.001)`).  For low-variance pricing
(short maturity or low vol — e.g. σ = 10 %, T = 0.01y → true c2 ≈ 1e-4), the
clamp inflated `L · √c2` by ≈ 3 ×, spreading the COS truncation interval far
past the actual density support and degrading convergence by orders of
magnitude.

Post-fix uses a tiny numerical-noise floor (1e-12) instead of a model-implied
minimum.
"""

from __future__ import annotations

import math

import pytest

from pricebook.models.black76 import OptionType, _norm_cdf
from pricebook.models.cos_method import bs_char_func, cos_price


def _bs_call(S, K, r, sigma, T):
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


class TestLowVarianceCOSConvergence:
    @pytest.mark.parametrize("sigma,T", [
        (0.10, 0.01),   # 10% vol, 1 week
        (0.05, 0.001),  # 5% vol, ~1 day
        (0.20, 0.05),   # 20% vol, 2-3 weeks
    ])
    def test_low_variance_matches_black_scholes(self, sigma, T):
        """For low-variance regimes, COS must converge to BS to <1e-6 (used
        to be off by orders of magnitude due to inflated truncation)."""
        S, K, r = 100.0, 100.0, 0.05
        bs = _bs_call(S, K, r, sigma, T)
        phi = bs_char_func(r, 0.0, sigma, T)
        cos_p = cos_price(phi, S, K, r, T, OptionType.CALL, N=128, L=10.0)
        rel = abs(cos_p - bs) / bs if bs > 0 else 0
        assert rel < 1e-6, (
            f"σ={sigma}, T={T}: COS={cos_p:.6f}, BS={bs:.6f}, rel={rel:.2e}"
        )

    def test_normal_variance_unchanged(self):
        """Sanity: typical variance (σ=20%, T=1) still works."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        bs = _bs_call(S, K, r, sigma, T)
        phi = bs_char_func(r, 0.0, sigma, T)
        cos_p = cos_price(phi, S, K, r, T, OptionType.CALL, N=128, L=10.0)
        assert abs(cos_p - bs) / bs < 1e-4
