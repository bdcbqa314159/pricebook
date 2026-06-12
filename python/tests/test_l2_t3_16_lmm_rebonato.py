"""Regression for L2 Tier-3 T3.16 — `LMM.rebonato_swaption_vol` uses ρ=1
correlation (standard Rebonato simplification), not ρ=δ_ij (uncorrelated).

Pre-fix: `var = Σ w_i² σ_i² × T` — the diagonal (uncorrelated) sum.  By
Cauchy–Schwarz this is ≤ (Σ w_i σ_i)² with equality only at N=1.  For an
N-period swap with uniform vols, the pre-fix vol was ≈ √(Σ w²)/Σ w of the
post-fix vol — an under-estimate of order 1/√N for equal weights.

Post-fix: `var = (Σ w_i σ_i)² × T` — the standard Rebonato ρ=1 result, and
the docstring's claimed formula.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.models.lmm import LMM


class TestRebonatoCorrelation:
    def test_single_period_unchanged(self):
        """N=1: ρ=1 and ρ=δ_ij give the same answer; this is a sanity check."""
        vols = np.array([0.20])
        L0 = np.array([0.04])
        sigma = LMM.rebonato_swaption_vol(vols, L0, tau=1.0, T_expiry=1.0)
        # Single forward, single period: σ_swap ≈ σ_forward = 0.20.
        assert math.isclose(sigma, 0.20, rel_tol=1e-6)

    def test_multi_period_uses_rho_one(self):
        """N=5 equal vols, equal forwards.  ρ=1 gives σ_swap = Σw·σ = σ ·
        Σw = σ · 1 = σ.  ρ=0 gives σ_swap = √(N) / N · σ ≈ σ/√N which is
        much smaller (a factor of √5 ≈ 2.24 too small)."""
        vols = np.full(5, 0.20)
        L0 = np.full(5, 0.04)
        sigma = LMM.rebonato_swaption_vol(vols, L0, tau=1.0, T_expiry=1.0)
        # Post-fix: σ_swap ≈ σ (since Σ weights = 1 by construction for
        # equal weights).
        # Pre-fix would have given σ_swap ≈ σ / √5 ≈ 0.0894.
        assert sigma > 0.15, (
            f"σ_swap = {sigma:.4f}; pre-fix would have been ~ σ/√5 ≈ 0.09"
        )
        # And not larger than the underlying σ (correlation ≤ 1).
        assert sigma <= 0.21

    def test_corr_matrix_argument(self):
        """User can pass a correlation matrix.  ρ=identity gives the
        diagonal (uncorrelated) answer — which equals the pre-fix value."""
        vols = np.full(3, 0.20)
        L0 = np.full(3, 0.04)
        sigma_rho_1 = LMM.rebonato_swaption_vol(vols, L0, 1.0, 1.0)
        sigma_diag = LMM.rebonato_swaption_vol(vols, L0, 1.0, 1.0,
                                                 corr=np.eye(3))
        # ρ=1 strictly greater than ρ=I for N > 1.
        assert sigma_rho_1 > sigma_diag

    def test_corr_matrix_intermediate(self):
        """A partially-correlated matrix (e.g. 0.5 off-diagonal) gives
        an answer between ρ=0 and ρ=1."""
        vols = np.full(3, 0.20)
        L0 = np.full(3, 0.04)
        sigma_rho_1 = LMM.rebonato_swaption_vol(vols, L0, 1.0, 1.0)
        sigma_diag = LMM.rebonato_swaption_vol(vols, L0, 1.0, 1.0,
                                                 corr=np.eye(3))
        corr_half = 0.5 * np.eye(3) + 0.5 * np.ones((3, 3))
        sigma_half = LMM.rebonato_swaption_vol(vols, L0, 1.0, 1.0,
                                                 corr=corr_half)
        assert sigma_diag < sigma_half < sigma_rho_1
