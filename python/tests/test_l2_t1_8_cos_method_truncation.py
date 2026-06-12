"""Regression for L2 Tier-1 T1.8 — COS-method V_k bounds intersected with [a, b].

`cos_price` (Fang-Oosterlee 2008) builds a truncation interval [a, b] for the
log-moneyness y = log(S/K) under the risk-neutral density.  The V_k payoff
coefficients integrate the payoff against cosine basis functions over the
*intersection* of the payoff support with the truncation interval:

- Call payoff (e^y − 1)⁺ has support y ≥ 0, so V_k^call integrates over
  [max(0, a), b].
- Put payoff (1 − e^y)⁺ has support y ≤ 0, so V_k^put integrates over
  [a, min(0, b)].

Pre-fix the bounds were hardcoded as [0, b] (call) and [a, 0] (put).  This is
correct when a < 0 < b (the typical near-ATM case), but **wrong** when:

- a > 0 (deep-ITM call, low vol, short T) — pre-fix integrated over [0, b]
  which includes [0, a], a region outside the truncation that should not
  contribute to V_k.
- b < 0 (deep-ITM put, low vol, short T) — symmetric: pre-fix integrated over
  [a, 0] including [b, 0] which is outside truncation.

Symptom: deep-ITM options mispriced vs Black-Scholes by far more than the
typical 1e-6 tolerance of the COS method.
"""

from __future__ import annotations

import math

import pytest

from pricebook.models.black76 import OptionType
from pricebook.models.cos_method import bs_char_func, cos_price


def _bs_call(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    """Plain Black-Scholes call price (reference)."""
    from pricebook.models.black76 import _norm_cdf
    if T <= 0 or sigma <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _bs_put(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    return _bs_call(S, K, r, sigma, T, q) - S * math.exp(-q * T) + K * math.exp(-r * T)


class TestCOSDeepITMCall:
    """Deep-ITM call: x = log(S/K) is large positive.  When σ is low and T is
    short, the truncation interval [a, b] can lie entirely in y > 0, so the
    pre-fix [0, b] domain was wrong (it includes [0, a] which is outside
    truncation)."""

    def test_atm_call_unchanged(self):
        """Sanity: ATM (a < 0 < b regime) should still agree with Black-Scholes."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        ref = _bs_call(S, K, r, sigma, T)
        phi = bs_char_func(r, 0.0, sigma, T)
        cos_p = cos_price(phi, S, K, r, T, OptionType.CALL, N=128, L=10.0)
        assert abs(cos_p - ref) / ref < 1e-4, (
            f"ATM call: COS={cos_p:.6f}, BS={ref:.6f}, rel={abs(cos_p-ref)/ref:.2e}"
        )

    def test_deep_itm_call_low_vol(self):
        """Deep ITM with low vol pushes a > 0.  Pre-fix this mispriced
        by a noticeable amount; post-fix should match BS to <1e-3."""
        S, K, r, sigma, T = 200.0, 100.0, 0.05, 0.08, 0.25
        ref = _bs_call(S, K, r, sigma, T)
        phi = bs_char_func(r, 0.0, sigma, T)
        cos_p = cos_price(phi, S, K, r, T, OptionType.CALL, N=256, L=10.0)

        # Confirm the regime: a > 0
        x = math.log(S / K)
        L = 10.0
        sigma_tot = sigma * math.sqrt(T)
        # Approximate a, b
        a_approx = x + (r - 0.5 * sigma**2) * T - L * sigma_tot
        assert a_approx > 0, f"Test setup invariant: need a > 0; got a≈{a_approx:.3f}"

        rel = abs(cos_p - ref) / ref
        # Pre-fix this was off by ~1% (huge for a COS method).
        # Post-fix should be within 1e-3.
        assert rel < 1e-3, (
            f"Deep-ITM call (S=200, K=100, σ=8%, T=3m): "
            f"COS={cos_p:.6f}, BS={ref:.6f}, rel={rel:.2e}"
        )


class TestCOSDeepITMPut:
    """Symmetric deep-ITM put case: b < 0."""

    def test_atm_put_unchanged(self):
        """Sanity: ATM put should agree with Black-Scholes."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        ref = _bs_put(S, K, r, sigma, T)
        phi = bs_char_func(r, 0.0, sigma, T)
        cos_p = cos_price(phi, S, K, r, T, OptionType.PUT, N=128, L=10.0)
        assert abs(cos_p - ref) / ref < 1e-4, (
            f"ATM put: COS={cos_p:.6f}, BS={ref:.6f}, rel={abs(cos_p-ref)/ref:.2e}"
        )

    def test_deep_itm_put_low_vol(self):
        """Deep ITM put (S << K) with low vol pushes b < 0."""
        S, K, r, sigma, T = 50.0, 100.0, 0.05, 0.08, 0.25
        ref = _bs_put(S, K, r, sigma, T)
        phi = bs_char_func(r, 0.0, sigma, T)
        cos_p = cos_price(phi, S, K, r, T, OptionType.PUT, N=256, L=10.0)

        x = math.log(S / K)
        L = 10.0
        sigma_tot = sigma * math.sqrt(T)
        b_approx = x + (r - 0.5 * sigma**2) * T + L * sigma_tot
        assert b_approx < 0, f"Test setup invariant: need b < 0; got b≈{b_approx:.3f}"

        rel = abs(cos_p - ref) / ref
        assert rel < 1e-3, (
            f"Deep-ITM put (S=50, K=100, σ=8%, T=3m): "
            f"COS={cos_p:.6f}, BS={ref:.6f}, rel={rel:.2e}"
        )


class TestCOSPutCallParity:
    """A cleaner integrity test: put-call parity must hold for any (a, b)
    regime.  Pre-fix the deep-ITM regimes violated parity by the V_k bug."""

    @pytest.mark.parametrize("S,K", [(100.0, 100.0), (200.0, 100.0), (50.0, 100.0)])
    def test_parity(self, S, K):
        r, sigma, T = 0.05, 0.08, 0.25
        phi = bs_char_func(r, 0.0, sigma, T)
        c = cos_price(phi, S, K, r, T, OptionType.CALL, N=256, L=10.0)
        p = cos_price(phi, S, K, r, T, OptionType.PUT, N=256, L=10.0)
        parity_lhs = c - p
        parity_rhs = S - K * math.exp(-r * T)
        diff = abs(parity_lhs - parity_rhs)
        # COS method intrinsic accuracy is ~1e-5 for N=256.
        assert diff < 1e-3, (
            f"Put-call parity (S={S}, K={K}): C−P={parity_lhs:.6f}, "
            f"S−Ke^{-r * T:.3f}={parity_rhs:.6f}, diff={diff:.2e}"
        )
