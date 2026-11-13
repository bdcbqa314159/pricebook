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
        from pricebook.foundation.distributions import norm_cdf

        # Against OUR `norm_cdf`, not scipy's: the claim is that this function performs exactly the
        # product of the library's own marginals, and scipy's differs in the last ULP. The first
        # draft compared against scipy and failed for that reason alone — a test asserting
        # bit-identity against a *different* implementation is asserting the wrong thing.
        assert bivariate_normal_cdf(h, k, 0.0) == norm_cdf(h) * norm_cdf(k)

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
        """`Φ₂(h,k;1) = Φ(min(h,k))` **exactly** at `ρ = 1` (closed form, not quadrature)."""
        for k in (-0.9, h, 2.0):
            assert bivariate_normal_cdf(h, k, 1.0) == pytest.approx(
                norm.cdf(min(h, k)), rel=1e-15
            )

    def test_the_comonotone_limit_is_APPROACHED_SLOWLY_which_is_the_mathematics(self) -> None:
        """The approach to `Φ(min)` is slow in `ρ`, and that is a fact about `Φ₂`, not a defect.

        Measured at `h = k = 0.2`: the gap to `Φ(0.2)` is 2.2e-02 at `ρ = 0.99`, 7.0e-03 at 0.999
        and 2.3e-03 at 0.9999 — roughly `√(1−ρ)`. A first draft asserted the limit to 5e-3 *at*
        `ρ = 0.999` and failed for that reason; the honest assertion is the **monotone approach**,
        with the exact value taken at `ρ = 1` where the closed form applies.
        """
        target = norm.cdf(0.2)
        gaps = [abs(bivariate_normal_cdf(0.2, 0.2, r) - target) for r in (0.9, 0.99, 0.999)]
        assert all(b < a for a, b in zip(gaps, gaps[1:])), gaps
        assert gaps[-1] < 1e-2

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

    @pytest.mark.parametrize(
        "h,k,rho", [(-2.0, -2.0, -0.5), (-2.5, -2.5, -0.5), (-4.0, 2.0, -0.7),
                    (-4.0, -4.0, 0.5), (-3.0, -3.0, 0.5), (-1.0, -1.0, -0.5)]
    )
    def test_the_TAIL_agrees_at_RELATIVE_tolerance_INSIDE_the_validity_regime(
        self, h: float, k: float, rho: float
    ) -> None:
        """**The regime wrong-way risk uses**, and the one an absolute tolerance cannot see.

        At `Φ₂ ≈ 1e-9` an absolute `1e-12` check passes however bad the relative error gets — the
        shape that let `norm_cdf`'s lower-tail cancellation survive fifteen tranches (T20a). Every
        case here has a true value above the measured validity floor asserted below.
        """
        reference = _scipy_phi2(h, k, rho)
        assert reference > 1e-8, "this case must sit inside the regime the next test characterises"
        assert bivariate_normal_cdf(h, k, rho) == pytest.approx(reference, rel=1e-7)

    def test_the_VALIDITY_REGIME_is_measured_and_stated_not_assumed(self) -> None:
        """**Where relative accuracy stops, and why — stated because a consumer must respect it.**

        `Φ₂ = Φ(h)Φ(k) + ∫₀^ρ φ₂`, and in the deep joint tail with `ρ < 0` the integral very nearly
        cancels the product: at `h = k = −3.5, ρ = −0.5` the true value is 8e-14 while the
        intermediate `Φ(h)Φ(k)` is 5e-8, so the answer is the difference of two numbers agreeing to
        six digits. Measured relative error against scipy:

        | true `Φ₂` | relative error |
        |---|---|
        | 3.8e-03 | 5.6e-15 |
        | 3.2e-06 | 4.5e-12 |
        | 2.4e-08 | 1.2e-09 |
        | 7.1e-11 | 3.7e-07 |
        | 8.1e-14 | 1.4e-04 |

        So: **relative accuracy to ~1e-7 while `Φ₂ ≳ 1e-8`; absolute accuracy (~1e-16) always.**
        Below ~1e-14 the question is moot — `scipy` itself returns **exactly 0.0** there, so the
        high-precision reference has failed before we have (at `h=k=−4, ρ=−0.5` ours gives 3.5e-17
        and scipy gives 0). That is the T20a lesson in its sharpest form: *the reference is code
        too*, and here it is the weaker side.

        **A second boundary, in the correlation rather than the value.** The integrand is peaked as
        `|ρ| → 1`, and a fixed rule cannot resolve it: measured against scipy at `h = k = 0.2`, the
        relative error is 4e-16 at `ρ = 0.95`, 8e-16 at 0.99, **4.3e-08 at 0.999** and **9.9e-05 at
        0.9999**. So the regime is `|ρ| ≲ 0.99`; `|ρ| = 1` itself is exact (closed form, no
        quadrature), and the gap between is where a consumer must not live.

        The consumer's obligation is to assert it stays inside both bounds, which tranche 36b's WWR
        oracle does rather than trusting this silently.
        """
        import math as _math

        # Inside the regime: relative accuracy holds.
        inside = _scipy_phi2(-2.5, -2.5, -0.5)
        assert inside > 1e-8
        assert abs(bivariate_normal_cdf(-2.5, -2.5, -0.5) / inside - 1.0) < 1e-7

        # Outside it: the value is still ABSOLUTELY tiny and correct in that sense, but relative
        # accuracy is gone — asserted so the boundary is a fact of the suite, not a footnote.
        outside = bivariate_normal_cdf(-3.5, -3.5, -0.5)
        assert abs(outside - _scipy_phi2(-3.5, -3.5, -0.5)) < 1e-15
        assert abs(outside / _scipy_phi2(-3.5, -3.5, -0.5) - 1.0) > 1e-6

        # And past it the reference itself fails, which is why the floor is not simply "go deeper".
        assert _scipy_phi2(-4.0, -4.0, -0.5) == 0.0
        assert 0.0 < bivariate_normal_cdf(-4.0, -4.0, -0.5) < 1e-15
        assert _math.isfinite(bivariate_normal_cdf(-4.0, -4.0, -0.5))

        # The correlation boundary, asserted the same way: accurate at 0.99, degraded at 0.9999.
        assert abs(bivariate_normal_cdf(0.2, 0.2, 0.99) / _scipy_phi2(0.2, 0.2, 0.99) - 1.0) < 1e-12
        assert abs(bivariate_normal_cdf(0.2, 0.2, 0.9999) / _scipy_phi2(0.2, 0.2, 0.9999) - 1.0) > 1e-6


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
