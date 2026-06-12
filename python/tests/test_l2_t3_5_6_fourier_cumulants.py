"""Regression for L2 Tier-3 T3.5 / T3.6 — `CharacteristicFunction.cumulants`
sign + stability.

* T3.5 — skewness had the WRONG SIGN.  The cumulant relation is
  `kappa_n = (−i)^n · log_phi^(n)(0)`.  For n=3, `(−i)³ = i`, and combining
  with Im(·) of the 3rd-derivative stencil flips sign: `kappa_3 =
  −Im(log_phi'''(0))`.  Pre-fix `c3 = +Im(stencil)/2h³` → wrong sign.

* T3.6 — kurtosis stencil used `h = 1e-4` for the 4th derivative.  `h⁴ =
  1e-16 ≈ machine ε`, so the 5-point stencil's numerator (subject to
  catastrophic cancellation) was dominated by round-off noise.  Use a
  wider `h` (~1e-2) for the 4th-derivative stencil, balancing truncation
  error and round-off.

The standard verification: a χ²(k=2) distribution has known cumulants
mean = 2, variance = 4, skewness = √2 ≈ 1.414 (positive), excess kurtosis
= 6.  Its CF is `φ(u) = 1 / (1 − 2iu)`.

Note: for k=2 chi-squared, the EXACT cumulants are:
    κ_n = 2^n · (n−1)! · k/2 = 2^n · (n−1)!
giving κ_1=2, κ_2=4, κ_3=16 (so skewness = κ_3/κ_2^{1.5} = 16/8 = 2.0),
κ_4=96 (so excess kurtosis = κ_4/κ_2² = 96/16 = 6).
"""

from __future__ import annotations

import cmath
import math

import pytest

from pricebook.numerical._fourier import CharacteristicFunction


def _gaussian_cf(mu: float, sigma: float):
    return lambda u: cmath.exp(1j * u * mu - 0.5 * u * u * sigma * sigma)


def _chi2_k2_cf():
    """Chi-squared(k=2) characteristic function: φ(u) = 1 / (1 − 2iu)."""
    return lambda u: 1.0 / (1.0 - 2j * u)


def _gamma_cf(shape: float, scale: float):
    """Gamma(k, θ) CF: φ(u) = (1 − iuθ)^(−k)."""
    return lambda u: (1.0 - 1j * u * scale) ** (-shape)


class TestGaussianCumulants:
    """Sanity: Gaussian has skew=0, ek=0."""

    def test_gaussian_zero_skew_kurtosis(self):
        cf = CharacteristicFunction(_gaussian_cf(mu=0.1, sigma=0.2), T=1.0)
        c = cf.cumulants(max_order=4)
        assert math.isclose(c["mean"], 0.1, abs_tol=1e-6)
        assert math.isclose(c["variance"], 0.04, abs_tol=1e-6)
        assert abs(c["skewness"]) < 1e-3
        assert abs(c["excess_kurtosis"]) < 1e-2


class TestChi2Cumulants:
    """χ²(k=2) — known positive skew = 2 and ek = 6."""

    def test_skew_positive_and_correct(self):
        """Fix T3.5: pre-fix this returned skewness ≈ −2 (wrong sign)."""
        cf = CharacteristicFunction(_chi2_k2_cf(), T=1.0)
        c = cf.cumulants(max_order=3)
        assert math.isclose(c["mean"], 2.0, abs_tol=1e-4)
        assert math.isclose(c["variance"], 4.0, abs_tol=1e-3)
        # Skewness should be +2 (positive).  Pre-fix was −2.
        assert c["skewness"] > 0, f"skewness = {c['skewness']:.4f} (must be positive)"
        assert math.isclose(c["skewness"], 2.0, abs_tol=0.05)

    def test_excess_kurtosis_stable(self):
        """Fix T3.6: pre-fix the h=1e-4 stencil for the 4th derivative gave
        round-off-dominated noise.  Post-fix the wider h gives ≈ 6."""
        cf = CharacteristicFunction(_chi2_k2_cf(), T=1.0)
        c = cf.cumulants(max_order=4)
        assert math.isclose(c["excess_kurtosis"], 6.0, abs_tol=0.1), (
            f"excess kurtosis = {c['excess_kurtosis']:.4f}, expected 6.0"
        )


class TestGammaSkewSign:
    """Gamma(k=3, θ=1): positive skewness 2/√k = 2/√3 ≈ 1.155, ek = 6/k = 2."""

    def test_gamma_skew_sign(self):
        cf = CharacteristicFunction(_gamma_cf(shape=3.0, scale=1.0), T=1.0)
        c = cf.cumulants(max_order=4)
        # Mean = kθ = 3, variance = kθ² = 3.
        assert math.isclose(c["mean"], 3.0, abs_tol=1e-3)
        assert math.isclose(c["variance"], 3.0, abs_tol=1e-3)
        # Skewness = 2/√k = 2/√3 ≈ 1.1547 (positive).
        expected_skew = 2.0 / math.sqrt(3.0)
        assert c["skewness"] > 0
        assert math.isclose(c["skewness"], expected_skew, abs_tol=0.05)
        # Excess kurtosis = 6/k = 2.
        assert math.isclose(c["excess_kurtosis"], 2.0, abs_tol=0.1)
