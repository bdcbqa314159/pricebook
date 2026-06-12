"""Regression for L2 deferred-module audit — `cos_bermudan` Bermudan/American
COS recursion now includes the Im(φ)·sin term.

Pre-fix the backward step computed `c_new[k] = df · Re(φ(u_k) · c[k])` with
real c[k], collapsing to `df · Re(φ) · c[k]`, then evaluated continuation as
`Σ c_new[k] · cos(u_k(x−a))` — DROPPING the Im(φ)·sin contribution entirely.

Fang-Oosterlee (2009) eq 2.10 actually evaluates
    Re[φ(u_k) · exp(i·u_k·(x − a))] · V_k
which expands to
    [Re(φ_k) cos(u_k(x − a)) − Im(φ_k) sin(u_k(x − a))] · V_k.

For drifted processes (BS with r ≠ 0, jump models with non-zero mean jump),
Im(φ) ≠ 0 and the sin terms carry the drift contribution.  Pre-fix the
recursion silently treated all processes as zero-drift, biasing the price.

Empirically on a vanilla BS American put (S=K=100, r=5%, σ=20%, T=1y) the
pre-fix COS-Bermudan with n_ex=100 returned 6.99, while the PDE American
benchmark gives 6.09 — a 15% over-statement.  Post-fix the agreement is
within 0.3% at n_ex=100 and 0.01% at n_ex=50.
"""

from __future__ import annotations

import cmath
import math

import pytest

from pricebook.models.black76 import OptionType
from pricebook.models.cos_bermudan import cos_bermudan
from pricebook.numerical._pde import GridType, PDEMethod, PDESolver1D


def _bs_cf(rate: float, sigma: float, dt: float):
    def phi(u):
        mu = (rate - 0.5 * sigma**2) * dt
        var = sigma**2 * dt
        return cmath.exp(1j * u * mu - 0.5 * u**2 * var)
    return phi


class TestBermudanMatchesPDE:
    def test_bs_american_put_via_cos_matches_pde(self):
        """COS Bermudan with many exercise dates should match PDE American
        within a small tolerance.  Pre-fix was 15% high."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        pde = PDESolver1D(
            method=PDEMethod.CRANK_NICOLSON,
            n_space=400, n_time=400, grid_type=GridType.UNIFORM,
        ).solve(S, K, T, sigma, r, is_call=False, is_american=True)

        cf = _bs_cf(r, sigma, T / 50)
        cos_p = cos_bermudan(
            cf, S, K, r, T, n_exercise=50,
            option_type=OptionType.PUT, N=128, L=10.0,
        )
        rel = abs(cos_p - pde.price) / pde.price
        assert rel < 0.01, (
            f"COS Bermudan {cos_p:.4f} vs PDE {pde.price:.4f}, rel={rel:.3%}"
        )

    def test_bermudan_increases_with_n_exercise(self):
        """A genuine Bermudan with more exercise points must be WORTH MORE.
        Pre-fix the price was nearly insensitive to n_ex (the missing sin
        terms made the recursion incompatible with the drift, producing
        n_ex-independent garbage)."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        prices = []
        for n_ex in [5, 20, 100]:
            cf = _bs_cf(r, sigma, T / n_ex)
            p = cos_bermudan(
                cf, S, K, r, T, n_exercise=n_ex,
                option_type=OptionType.PUT, N=128, L=10.0,
            )
            prices.append(p)
        # Monotone increasing in n_ex (Bermudan converging to American).
        assert prices[0] < prices[1] < prices[2], (
            f"Prices not monotone in n_ex: {prices}"
        )
        # Gap between coarse and fine Bermudan should be significant.
        assert prices[2] - prices[0] > 0.05

    def test_bs_european_call_at_n_ex_1_matches_european(self):
        """A '1-exercise Bermudan' (only exercisable at expiry) is a European.
        Compare to Black-Scholes."""
        S, K, r, sigma, T = 100.0, 110.0, 0.05, 0.20, 1.0
        from pricebook.models.black76 import _norm_cdf
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        bs_call = S * _norm_cdf(d1) - K * math.exp(-r*T) * _norm_cdf(d2)

        cf = _bs_cf(r, sigma, T)
        cos_eu = cos_bermudan(
            cf, S, K, r, T, n_exercise=1,
            option_type=OptionType.CALL, N=128, L=10.0,
        )
        rel = abs(cos_eu - bs_call) / bs_call
        # n_ex=1 → only exercise at T, so equivalent to European.
        # Pre-fix dropped drift → significant bias; post-fix matches BS.
        assert rel < 0.01, (
            f"COS n_ex=1 call {cos_eu:.4f} vs BS {bs_call:.4f}, rel={rel:.3%}"
        )
