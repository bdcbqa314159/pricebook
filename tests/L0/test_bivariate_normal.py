"""The bivariate normal CDF `Φ₂` (L0) — tranche 36b's new primitive. **Pre-registered.**

Wrong-way risk needs `Φ₂(h,k;ρ) = P(X ≤ h, Y ≤ k)` for standard normals with correlation `ρ`, and
`src/` has no such function — nor does it use scipy (the `quadrature` module says why). So this is a
genuine new L0 primitive, and §4's "a reference is code too" applies to it **before** anything
consumes it: it lands green on its own first.

**It is derived, not transcribed**, which §4 rates higher. Differentiating the bivariate density in
the correlation (Plackett's identity) gives

    ∂Φ₂/∂ρ = φ₂(h,k;ρ) = exp(−(h² − 2ρhk + k²) / (2(1−ρ²))) / (2π√(1−ρ²))

so `Φ₂(h,k;ρ) = Φ(h)Φ(k) + ∫₀^ρ φ₂(h,k;t) dt`. That form is chosen for a reason the oracles below
exercise: at `ρ = 0` the integral is empty, so **independence is exact by construction** rather than
a numerical limit — which is what lets tranche 36b's `ρ → 0` reduction be bit-identical.

**The accuracy argument is owed in the regime this is USED, not on average (T20a).** The wrong-way
integrand lives in the **joint tail** — small survival probability *and* deep exposure — which is
exactly where `norm_cdf`'s own lower-tail defect hid for fifteen tranches. So the tail is asserted
at **relative** tolerance against `scipy.stats.multivariate_normal`, kept off the production path,
and the committed node count's resolution is stated rather than assumed.
"""

import math

import pytest
from scipy.stats import multivariate_normal, norm

from pricebook.foundation.bivariate_normal import bivariate_normal_cdf

RHOS = (-0.95, -0.6, -0.2, 0.0, 0.3, 0.7, 0.95)


def _scipy_phi2(h: float, k: float, rho: float) -> float:
    """An independent high-precision reference, off the production path (the T20a arrangement)."""
    return float(multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]).cdf([h, k]))


class TestTheReductionsThatNeedNoReference:
    """Identities true of any correct `Φ₂`, so they survive a transcription or quadrature error."""

    @pytest.mark.parametrize("h", [-2.5, -0.4, 0.0, 1.3])
    @pytest.mark.parametrize("k", [-1.8, 0.0, 2.2])
    def test_INDEPENDENCE_at_zero_correlation_is_EXACT(self, h: float, k: float) -> None:
        """`Φ₂(h,k;0) = Φ(h)Φ(k)` — bit-identical, because the correlation integral is empty.

        Not an approximation that happens to be close: the chosen form makes it a structural
        property, which is what tranche 36b's `ρ → 0` reduction rests on.
        """
        assert bivariate_normal_cdf(h, k, 0.0) == norm.cdf(h) * norm.cdf(k)

    @pytest.mark.parametrize("h", [-2.0, -0.3, 1.1])
    @pytest.mark.parametrize("rho", RHOS)
    def test_the_MARGINALS_come_back(self, h: float, rho: float) -> None:
        """`Φ₂(h,+∞;ρ) = Φ(h)` for every `ρ` — the cheapest check that the integral is normalised."""
        assert bivariate_normal_cdf(h, 12.0, rho) == pytest.approx(norm.cdf(h), abs=1e-12)
        assert bivariate_normal_cdf(12.0, h, rho) == pytest.approx(norm.cdf(h), abs=1e-12)

    @pytest.mark.parametrize("rho", RHOS)
    def test_an_IMPOSSIBLE_corner_is_zero(self, rho: float) -> None:
        assert bivariate_normal_cdf(-12.0, 0.5, rho) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("h", [-1.5, 0.2, 1.7])
    def test_PERFECT_correlation_is_the_minimum_marginal(self, h: float) -> None:
        """`Φ₂(h,k;1) = Φ(min(h,k))` — the comonotone limit, approached as `ρ → 1`."""
        for k in (-0.9, h, 2.0):
            assert bivariate_normal_cdf(h, k, 0.999) == pytest.approx(
                norm.cdf(min(h, k)), abs=5e-3
            )

    @pytest.mark.parametrize("rho", RHOS)
    def test_the_REFLECTION_identity_holds(self, rho: float) -> None:
        """`Φ₂(h,k;ρ) + Φ₂(−h,k;−ρ) = Φ(k)` — the half-plane decomposition, a requirement the
        quadrature must satisfy for every argument and which no single value oracle implies."""
        for h, k in ((-1.2, 0.8), (0.3, -1.6), (2.0, 1.1)):
            total = bivariate_normal_cdf(h, k, rho) + bivariate_normal_cdf(-h, k, -rho)
            assert total == pytest.approx(norm.cdf(k), abs=1e-10)

    @pytest.mark.parametrize("rho", RHOS)
    def test_it_is_SYMMETRIC_in_its_arguments(self, rho: float) -> None:
        for h, k in ((-1.2, 0.8), (0.3, -1.6), (2.4, -0.1)):
            assert bivariate_normal_cdf(h, k, rho) == pytest.approx(
                bivariate_normal_cdf(k, h, rho), rel=1e-14
            )

    @pytest.mark.parametrize("rho", RHOS)
    def test_it_is_a_PROBABILITY_and_monotone_in_each_argument(self, rho: float) -> None:
        previous = -1.0
        for h in (-3.0, -1.0, 0.0, 1.0, 3.0):
            value = bivariate_normal_cdf(h, 0.5, rho)
            assert 0.0 <= value <= 1.0
            assert value >= previous
            previous = value


class TestAgainstAnIndependentHighPrecisionReference:
    """scipy off the production path — the tranche-20a arrangement, which exists for this."""

    @pytest.mark.parametrize("rho", RHOS)
    @pytest.mark.parametrize("h,k", [(-1.0, 0.5), (0.0, 0.0), (1.2, -0.7), (2.0, 2.0)])
    def test_the_BODY_agrees_to_high_absolute_accuracy(self, h: float, k: float, rho: float) -> None:
        assert bivariate_normal_cdf(h, k, rho) == pytest.approx(_scipy_phi2(h, k, rho), abs=1e-12)

    @pytest.mark.parametrize("rho", [-0.8, -0.3, 0.3, 0.8])
    @pytest.mark.parametrize("h,k", [(-3.0, -3.0), (-4.0, -2.0), (-5.0, -4.0), (-6.0, -6.0)])
    def test_the_JOINT_TAIL_agrees_at_RELATIVE_tolerance(self, h: float, k: float, rho: float) -> None:
        """**The regime wrong-way risk actually uses**, and the one an absolute tolerance cannot see.

        At `Φ₂ ≈ 1e-9` an absolute `1e-12` check passes however bad the relative error gets — the
        exact shape that let `norm_cdf`'s lower-tail cancellation survive fifteen tranches (T20a).
        """
        reference = _scipy_phi2(h, k, rho)
        assert reference > 0.0, "the reference must itself be meaningful here"
        assert bivariate_normal_cdf(h, k, rho) == pytest.approx(reference, rel=1e-7)


class TestTheGuardsAreReachable:
    @pytest.mark.parametrize("rho", [-1.5, 1.5, math.nan])
    def test_a_CORRELATION_outside_the_unit_interval_is_refused(self, rho: float) -> None:
        with pytest.raises(ValueError, match="correlation"):
            bivariate_normal_cdf(0.0, 0.0, rho)

    def test_PERFECT_correlation_is_handled_exactly_not_by_quadrature(self) -> None:
        """`|ρ| = 1` makes the integrand singular, so it is the closed form or nothing."""
        assert bivariate_normal_cdf(0.7, 1.4, 1.0) == pytest.approx(norm.cdf(0.7), rel=1e-15)
        assert bivariate_normal_cdf(0.7, 1.4, -1.0) == pytest.approx(
            max(norm.cdf(0.7) + norm.cdf(1.4) - 1.0, 0.0), rel=1e-15
        )
