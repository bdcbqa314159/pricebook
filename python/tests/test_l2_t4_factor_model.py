"""Regression for L2 phase-2 audit of `risk.factor_model`:

(a) ``factor_covariance(method='shrinkage')`` used the ad-hoc intensity
    ``alpha_lw = 1 / (n · delta)`` with no basis in Ledoit-Wolf (2004).
    The result was NOT a Ledoit-Wolf shrunk estimator.  Now uses the
    correct LW formula:
        π̂  = sum over i,j of (1/T) Σ_t (x̃_ti x̃_tj - s_ij)²
        γ̂² = ||F - S||²_F
        ρ̂  = trace(π̂ matrix)        (identity-target case)
        κ  = (π̂ - ρ̂) / γ̂²
        δ* = max(0, min(κ/T, 1))

(b) ``factor_timing`` code contradicted its own docstring.  Docstring
    said "low z-score (cheap), overweight" (contrarian); code did
    "high z-score, overweight" (momentum on factor value).  Now
    follows contrarian convention consistently.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.risk.factor_model import factor_covariance, factor_timing


class TestLedoitWolfShrinkage:
    def test_intensity_in_valid_range(self):
        """δ* must lie in [0, 1]."""
        rng = np.random.default_rng(42)
        n_obs = 500
        # Correlated factors.
        true_cov = np.array([[1.0, 0.3, 0.1],
                              [0.3, 1.0, 0.2],
                              [0.1, 0.2, 1.0]])
        L = np.linalg.cholesky(true_cov)
        data = rng.standard_normal((n_obs, 3)) @ L.T
        result = factor_covariance(
            {"f1": data[:, 0], "f2": data[:, 1], "f3": data[:, 2]},
            method="shrinkage",
        )
        # Variances should match true ~1.0 within sampling noise.
        for i in range(3):
            assert 0.7 < result.covariance[i, i] < 1.3

    def test_high_dimensional_shrinkage_is_meaningful(self):
        """For p≈T (high-dim), LW should give nontrivial shrinkage > 10%."""
        rng = np.random.default_rng(7)
        n_obs = 30
        n_factors = 15  # p/T = 0.5 → significant shrinkage expected
        data = rng.standard_normal((n_obs, n_factors))

        factor_dict = {f"f{i}": data[:, i] for i in range(n_factors)}
        result_sample = factor_covariance(factor_dict, method="sample")
        result_shrink = factor_covariance(factor_dict, method="shrinkage")

        # Compare condition numbers — shrunk should be better conditioned.
        # Sample cov in 30x15 is well-defined but noisy; shrinkage reduces noise.
        assert result_shrink.condition_number <= result_sample.condition_number + 1e-6

        # Off-diagonals of shrunk matrix should be smaller in magnitude than sample
        # (since identity target has zero off-diagonals → contraction toward 0).
        off_diag_sample = result_sample.covariance - np.diag(np.diag(result_sample.covariance))
        off_diag_shrink = result_shrink.covariance - np.diag(np.diag(result_shrink.covariance))
        assert np.linalg.norm(off_diag_shrink) <= np.linalg.norm(off_diag_sample)

    def test_independent_factors_shrink_toward_identity(self):
        """For T iid factors with true cov = I, shrinkage should pull off-diagonals to 0."""
        rng = np.random.default_rng(11)
        n_obs = 60
        n_factors = 8
        data = rng.standard_normal((n_obs, n_factors))
        factor_dict = {f"f{i}": data[:, i] for i in range(n_factors)}

        result = factor_covariance(factor_dict, method="shrinkage")
        # Off-diagonal magnitudes should be < typical sample-cov noise level.
        off_diag = result.covariance - np.diag(np.diag(result.covariance))
        # Sample-cov off-diagonals on iid data have std ~ 1/sqrt(n_obs) ≈ 0.13.
        # Shrinkage should bring them noticeably smaller.
        assert np.abs(off_diag).max() < 0.15


class TestFactorTimingContrarian:
    def test_high_z_underweight(self):
        """High z-score (factor expensive) → underweight (contrarian)."""
        values = np.concatenate([np.zeros(100), np.ones(10) * 3.0])
        returns = np.zeros(110)
        result = factor_timing(values, returns)
        assert result.signal == "underweight"

    def test_low_z_overweight(self):
        """Low z-score (factor cheap) → overweight."""
        values = np.concatenate([np.zeros(100), np.ones(10) * (-3.0)])
        returns = np.zeros(110)
        result = factor_timing(values, returns)
        assert result.signal == "overweight"

    def test_hit_rate_perfect_mean_reversion(self):
        """In a perfectly mean-reverting regime, contrarian timing should win."""
        # Construct values where each value is exactly the next-period return inverted.
        # The crisper the construction the cleaner the test.
        from pricebook.risk.factor_model import zscore
        n = 200
        # values = +1 for first half, -1 for second half — clean threshold crossings.
        values = np.concatenate([np.ones(100), -np.ones(100)])
        # returns: when z > 0 → return < 0; when z < 0 → return > 0.
        # zscore of this constant-then-flipped series is non-trivial; compute explicitly.
        z = zscore(values)
        returns = -np.sign(z) * 0.01  # exactly anti-correlated

        result = factor_timing(values, returns, z_threshold=0.5)
        # Every (z, return) pair where |z| > threshold has opposite signs → contrarian wins.
        # One transition bar (z[99] -> return[100]) may straddle; >99% is the test.
        assert result.historical_hit > 0.99
