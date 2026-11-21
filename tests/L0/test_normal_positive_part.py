"""`normal_positive_part` — `E[(m + s*Z)+]` for a standard normal `Z` (L0).

**This is tranche 25's `expected_positive_exposure` body, promoted at its second consumer.** The
rule of two (§6) fires on a shared SHAPE, and the shape here is exact: a CVA's expected positive
exposure and a Bachelier caplet are the *same closed form*, `m*Phi(m/s) + s*phi(m/s)`. Tranche 71's
Bachelier caplet is an L4 engine and the exposure is L5, so §2 forbids the engine importing it —
the choice was a third copy or a promotion, and §6 answers that.

The oracles here are the POINTWISE ones, because the CVA module's own tests are a regression guard
and not an independent check of the formula: they telescope through the same code. What is asserted
independently is (a) the closed form against a quadrature of `(m + s*z)+` against the normal
density, which shares no line with the implementation, and (b) the two limits the formula must
satisfy for ALL parameters.
"""

import math

import pytest

from pricebook.foundation.distributions import norm_cdf, norm_pdf, normal_positive_part


def _quadrature(mean: float, stdev: float, half_width: float = 12.0, n: int = 40_001) -> float:
    """`E[(m + s*Z)+]` by Simpson over the density — no `Phi`, no `phi`, no closed form.

    Independent of the implementation in the way §4 demands: it integrates the DEFINITION of the
    expectation rather than restating the analytic answer. 12 standard deviations is well past
    where the integrand contributes at double precision."""
    step = 2.0 * half_width / (n - 1)
    total = 0.0
    for i in range(n):
        z = -half_width + i * step
        weight = 1.0 if i in (0, n - 1) else (4.0 if i % 2 else 2.0)
        density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        total += weight * max(mean + stdev * z, 0.0) * density
    return total * step / 3.0


# `m/s` spans deep out of the money, at the money and deep in: the ratio is the only argument the
# formula has, so a fixture clustered near zero would leave both tails untested.
CASES = ((-0.04, 0.01), (-0.006, 0.0085), (0.0, 0.012), (0.003, 0.0085), (0.05, 0.004))


@pytest.mark.parametrize("mean,stdev", CASES)
def test_the_closed_form_matches_an_INDEPENDENT_quadrature_of_the_definition(mean, stdev):
    got, want = normal_positive_part(mean, stdev), _quadrature(mean, stdev)
    assert abs(got - want) < 1e-12 + 1e-9 * abs(want), (got, want)


@pytest.mark.parametrize("mean,stdev", CASES)
def test_the_two_TERMS_are_both_present_and_correctly_weighted(mean, stdev):
    # The pointwise transcription. A properties-only oracle set tolerates a dropped `s*phi` term
    # wherever the quadrature's band is loose, and §4's charter names that exact survivor.
    ratio = mean / stdev
    assert normal_positive_part(mean, stdev) == pytest.approx(
        mean * norm_cdf(ratio) + stdev * norm_pdf(ratio), rel=0.0, abs=0.0
    )


def test_the_premise_that_the_phi_TERM_is_material_at_every_case():
    # Non-degeneracy, asserted rather than assumed (§4): if `s*phi` were negligible anywhere in the
    # fixture, the quadrature oracle above would pass with it dropped at that point.
    for mean, stdev in CASES:
        ratio = mean / stdev
        whole, first = normal_positive_part(mean, stdev), mean * norm_cdf(ratio)
        assert abs(whole - first) > 0.05 * abs(whole), (mean, stdev, whole, first)


def test_a_zero_deviation_is_the_plain_POSITIVE_PART_and_that_is_its_limit():
    assert normal_positive_part(0.03, 0.0) == 0.03
    assert normal_positive_part(-0.03, 0.0) == 0.0
    # and the limit is approached, not merely defined at the boundary
    assert normal_positive_part(-0.03, 1e-6) == pytest.approx(0.0, abs=1e-20)
    assert normal_positive_part(0.03, 1e-6) == pytest.approx(0.03, abs=1e-12)


def test_it_is_a_positive_part_so_it_DOMINATES_the_mean_and_stays_non_negative():
    for mean, stdev in CASES:
        value = normal_positive_part(mean, stdev)
        assert value >= max(mean, 0.0) - 1e-18, (mean, stdev, value)


def test_a_negative_deviation_is_REFUSED_rather_than_silently_squared():
    with pytest.raises(ValueError, match="standard deviation"):
        normal_positive_part(0.01, -0.001)
