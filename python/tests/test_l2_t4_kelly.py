"""Regression for L2 phase-2 audit of `risk.kelly`:

(a) ``kelly_fraction`` silently returned f=0 when volatility=0, masking
    two genuinely different degenerate cases (positive excess = +∞,
    negative excess = −∞).  Now raises ValueError.

(b) ``multi_asset_kelly`` validated nothing and fell back to
    diagonal-only inverse on singular Σ.  Diagonal fallback drops
    correlation structure; for highly-correlated assets the true
    Kelly concentrates weight on the best risk-adjusted asset but
    diagonal-only spreads it.  Now uses Moore-Penrose pseudoinverse
    and validates cov shape/symmetry/mu compatibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.risk.kelly import kelly_fraction, multi_asset_kelly


class TestKellyFractionRequiresVol:
    def test_zero_vol_raises(self):
        with pytest.raises(ValueError, match="volatility"):
            kelly_fraction(expected_return=0.10, volatility=0.0)

    def test_negative_vol_raises(self):
        with pytest.raises(ValueError, match="volatility"):
            kelly_fraction(expected_return=0.05, volatility=-0.20)

    def test_positive_vol_unchanged(self):
        result = kelly_fraction(0.10, 0.20)
        assert result.kelly_fraction == pytest.approx(0.10 / 0.04, rel=1e-12)


class TestMultiAssetKellyValidation:
    def test_non_square_cov_raises(self):
        with pytest.raises(ValueError, match="square"):
            multi_asset_kelly(
                mu=np.array([0.1, 0.08, 0.05]),
                cov=np.eye(3)[:, :2],  # (3, 2) - non-square
            )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="incompatible"):
            multi_asset_kelly(
                mu=np.array([0.1, 0.08]),
                cov=np.eye(3),
            )

    def test_non_symmetric_raises(self):
        non_sym = np.array([[0.04, 0.01], [0.02, 0.04]])  # rho_12 != rho_21
        with pytest.raises(ValueError, match="symmetric"):
            multi_asset_kelly(mu=np.array([0.1, 0.08]), cov=non_sym)


class TestMultiAssetKellyAnalytical:
    def test_two_assets_matches_closed_form(self):
        """For diagonal Σ, Kelly = excess / σ²_i per-asset."""
        mu = np.array([0.10, 0.08])
        cov = np.diag([0.04, 0.0225])  # σ = (0.2, 0.15)
        result = multi_asset_kelly(mu, cov, max_leverage=100.0)
        expected = mu / np.diag(cov)
        np.testing.assert_allclose(result["weights"], expected, atol=1e-10)

    def test_correlated_kelly_concentrates_via_inverse(self):
        """For correlated Σ, Kelly = Σ⁻¹·μ̄ (not diagonal)."""
        rho = 0.5
        sig = np.array([0.2, 0.15])
        cov = np.array([
            [sig[0]**2,  rho * sig[0] * sig[1]],
            [rho * sig[0] * sig[1], sig[1]**2]
        ])
        mu = np.array([0.10, 0.08])

        result = multi_asset_kelly(mu, cov, max_leverage=100.0)
        # Analytical: Σ⁻¹·μ
        expected = np.linalg.solve(cov, mu)
        np.testing.assert_allclose(result["weights"], expected, atol=1e-10)


class TestMultiAssetKellySingularPseudoInverse:
    def test_singular_cov_uses_pinv_not_diag(self):
        """Two perfectly correlated assets → Σ is singular.

        Pre-fix diagonal fallback: f = excess / σ² per asset.
        Post-fix pseudoinverse: gives the min-norm solution.
        """
        # Perfectly correlated: rho = 1.0 → cov is singular.
        sig = np.array([0.2, 0.2])
        rho = 1.0 - 1e-12  # near-singular but invertible numerically
        cov_near_sing = np.array([
            [sig[0]**2, rho * sig[0] * sig[1]],
            [rho * sig[0] * sig[1], sig[1]**2]
        ])
        mu = np.array([0.10, 0.08])
        # Should not error.  And total leverage should be finite.
        result = multi_asset_kelly(mu, cov_near_sing, max_leverage=100.0)
        assert np.isfinite(result["leverage"])
        assert np.all(np.isfinite(result["weights"]))


class TestMultiAssetKellyGrowthFormula:
    def test_growth_matches_analytical(self):
        """g = rf + f·μ̄ − 0.5·f'Σf."""
        mu = np.array([0.10, 0.08])
        cov = np.diag([0.04, 0.0225])
        rf = 0.02
        result = multi_asset_kelly(mu, cov, risk_free_rate=rf,
                                   fraction=1.0, max_leverage=1000.0)
        f = np.array(result["weights"])
        expected_growth = rf + float(f @ (mu - rf)) - 0.5 * float(f @ cov @ f)
        assert result["expected_growth"] == pytest.approx(expected_growth, abs=1e-12)
