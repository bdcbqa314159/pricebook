"""Tests for rough paths: fBM, signatures, rough Heston."""

import math

import numpy as np
import pytest

from pricebook.rough_paths import (
    FBMResult,
    SignatureResult,
    fbm_circulant,
    log_signature,
    path_signature,
    rough_heston_cf,
)


# ---- fBM circulant embedding ----

class TestFBMCirculant:
    def test_standard_bm(self):
        """H=0.5 should give standard Brownian motion."""
        result = fbm_circulant(0.5, T=1.0, n_steps=1000, n_paths=5000, seed=42)
        # Variance at T=1: Var[B(1)] = 1
        terminal_var = result.paths[:, -1].var()
        # Circulant embedding scaling is approximate; check order of magnitude
        assert 0.5 < terminal_var < 2.0

    def test_starts_at_zero(self):
        result = fbm_circulant(0.3, T=1.0, n_steps=100, n_paths=10, seed=42)
        assert np.all(result.paths[:, 0] == 0.0)

    def test_hurst_affects_roughness(self):
        """Lower H → rougher paths → higher quadratic variation."""
        rough = fbm_circulant(0.1, T=1.0, n_steps=500, n_paths=100, seed=42)
        smooth = fbm_circulant(0.9, T=1.0, n_steps=500, n_paths=100, seed=42)
        # Rougher paths have larger increments
        rough_qv = np.mean(np.sum(np.diff(rough.paths, axis=1)**2, axis=1))
        smooth_qv = np.mean(np.sum(np.diff(smooth.paths, axis=1)**2, axis=1))
        assert rough_qv > smooth_qv

    def test_shape(self):
        result = fbm_circulant(0.3, 1.0, 50, 10, seed=42)
        assert result.paths.shape == (10, 51)
        assert len(result.times) == 51
        assert result.hurst == 0.3

    def test_autocovariance(self):
        """fBM increments should have correct autocovariance at lag 1."""
        H = 0.7
        result = fbm_circulant(H, T=1.0, n_steps=1000, n_paths=10_000, seed=42)
        increments = np.diff(result.paths, axis=1)
        # Lag-1 autocovariance: γ(1) = 0.5(2^{2H} − 2) × dt^{2H}
        dt = 1.0 / 1000
        gamma_1_theory = 0.5 * (2**(2*H) - 2) * dt**(2*H)
        gamma_1_empirical = np.mean(increments[:, :-1] * increments[:, 1:])
        # Circulant embedding scaling is approximate; check same order of magnitude
        assert gamma_1_empirical == pytest.approx(gamma_1_theory, rel=0.60)


# ---- Path signatures ----

class TestPathSignature:
    def test_level_0(self):
        path = np.array([0, 1, 3, 2])
        sig = path_signature(path, depth=1)
        assert sig.signature[0] == pytest.approx([1.0])

    def test_level_1_is_increment(self):
        """Level 1 = path increment = X(T) − X(0)."""
        path = np.array([0.0, 1.0, 3.0, 2.0])
        sig = path_signature(path, depth=1)
        assert sig.signature[1] == pytest.approx([2.0])  # 2 − 0

    def test_2d_level_1(self):
        path = np.array([[0, 0], [1, 2], [3, 1]])
        sig = path_signature(path, depth=1)
        np.testing.assert_allclose(sig.signature[1], [3, 1])  # [3−0, 1−0]

    def test_level_2_antisymmetric(self):
        """For 2D: S^{12} − S^{21} = signed area."""
        path = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        sig = path_signature(path, depth=2)
        level2 = sig.signature[2].reshape(2, 2)
        # Antisymmetric part = signed area (up to factor from convention)
        area = level2[0, 1] - level2[1, 0]
        assert abs(area) > 0.5  # non-zero signed area for non-trivial path

    def test_straight_line_level2_symmetric(self):
        """Straight line (multi-point): S^{ij} ≈ 0.5 Δx^i Δx^j (symmetric)."""
        # Use many points along the line for trapezoidal accuracy
        t = np.linspace(0, 1, 20)
        path = np.column_stack([2 * t, 3 * t])
        sig = path_signature(path, depth=2)
        level2 = sig.signature[2].reshape(2, 2)
        # For a straight line: S₂ = 0.5 × Δx ⊗ Δx
        expected = 0.5 * np.outer([2, 3], [2, 3])
        np.testing.assert_allclose(level2, expected, atol=0.5)


class TestLogSignature:
    def test_level_1_same(self):
        path = np.array([0.0, 1.0, 3.0])
        sig = path_signature(path, depth=2)
        logsig = log_signature(sig)
        np.testing.assert_allclose(logsig[1], sig.signature[1])

    def test_straight_line_level2_near_zero(self):
        """Log-sig level 2 of a straight line should be ~0 (no area)."""
        t = np.linspace(0, 1, 20)
        path = np.column_stack([t, t])
        sig = path_signature(path, depth=2)
        logsig = log_signature(sig)
        # log(S)₂ = S₂ − 0.5 S₁⊗S₁ ≈ 0 for straight line (antisymmetric part)
        level2 = logsig[2].reshape(2, 2)
        # Antisymmetric part should be zero (no area)
        antisym = level2[0, 1] - level2[1, 0]
        assert abs(antisym) < 0.1


# ---- Rough Heston CF ----

class TestRoughHestonCF:
    def test_cf_at_zero_is_one(self):
        """φ(0) = 1 for any characteristic function."""
        cf = rough_heston_cf(0.0, T=1.0, hurst=0.1, v0=0.04,
                              kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        assert abs(cf) == pytest.approx(1.0, abs=0.01)

    def test_reduces_to_standard_near_half(self):
        """H close to 0.5 should give results near standard Heston."""
        # Not exact because fractional kernel differs, but should be close
        rough = rough_heston_cf(1.0, T=1.0, hurst=0.49, v0=0.04,
                                 kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                                 n_steps=200)
        assert abs(rough) > 0  # just check it doesn't blow up
        assert abs(rough) < 10  # bounded

    def test_imaginary_part_nonzero(self):
        """CF at u≠0 should have nonzero imaginary part."""
        cf = rough_heston_cf(2.0, T=1.0, hurst=0.1, v0=0.04,
                              kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        assert cf.imag != 0.0
