"""The non-central chi-squared CDF and survival function (L0).

The primitive tranche 72's CEV caplet is built on, and the reason it is two functions rather than
one. **`scipy` is the oracle and is deliberately off the production path** — the tranche-20a
arrangement: `foundation/` ships no scipy dependency, and the test may use what the library may not,
which is what makes this an independent cross-check rather than a round trip.

**Why a native survival function exists, measured rather than assumed.** Cox's CEV call formula is
`F·[1 − Q(2w; ·)] − K·Q(2z; ·)`, so the first term needs the *upper* tail. Forming it as `1 − cdf`
loses that tail exactly the way `(1 + erf)/2` lost the normal one (OI-3, tranche 20c): measured here,
`1 − cdf` carries **1.2% relative error at x = 100** and returns **exactly 0.0 beyond x ≈ 200**, where
the true survival is 8.7e-32. A deep out-of-the-money caplet is precisely where a CEV pricer is asked
for a number. So `noncentral_chisquare_sf` sums the *upper* incomplete gammas directly — a sum of
positive terms, no cancellation anywhere — and holds **relative** accuracy to 1.1e-14 down to 3e-148.
The oracle below asserts both halves: that `sf` is accurate there, **and** that `1 - cdf` is not. The
second half is what proves the design necessary rather than merely plausible (the tranche-15 pattern).
"""

import math

import numpy
import pytest

from pricebook.foundation.noncentral_chisquare import (
    noncentral_chisquare_cdf,
    noncentral_chisquare_sf,
)

scipy_stats = pytest.importorskip("scipy.stats")

# The grid spans what the CEV caplet actually asks for and what the series finds hard: the centre,
# the deep upper tail (where the Poisson-weighted sum's terms underflow), HIGH NONCENTRALITY (where
# the sum needs ~lambda/2 terms before it converges), a fractional and a large degrees of freedom,
# and `x` below the mode. A low-noncentrality-only sweep makes a truncated series look exact (§4).
CASES = (
    (5.0, 2.0, 5.0),
    (1.0, 3.0, 1.0),
    (20.0, 10.0, 20.0),
    (0.5, 1.0, 0.1),
    (100.0, 2.0, 80.0),
    (2000.0, 4.0, 1500.0),          # high noncentrality — the slow-convergence regime
    (1.0e-3, 3.0, 2.0),             # far below the mode, CDF ~ 3e-06
    (300.0, 100.0, 200.0),          # large dof
    (60.0, 4.0, 5.0),               # upper tail, CDF ~ 1 - 1.2e-07
    (400.0, 4.0, 5.0),              # deep upper tail, survival ~ 1.8e-69
)


@pytest.mark.parametrize("x,dof,nc", CASES)
def test_the_CDF_matches_scipy_RELATIVELY(x, dof, nc):
    want = scipy_stats.ncx2.cdf(x, dof, nc)
    got = noncentral_chisquare_cdf(x, dof, nc)
    assert got == pytest.approx(want, rel=1e-11, abs=0.0), (got, want)


@pytest.mark.parametrize("x,dof,nc", CASES)
def test_the_SURVIVAL_matches_scipy_RELATIVELY_including_the_deep_tail(x, dof, nc):
    # RELATIVE, not absolute — tranche 20a's rule. An absolute band passes however bad the relative
    # error gets where the true value is ~0, which is the whole upper tail.
    want = scipy_stats.ncx2.sf(x, dof, nc)
    got = noncentral_chisquare_sf(x, dof, nc)
    assert got == pytest.approx(want, rel=1e-11, abs=0.0), (got, want)


@pytest.mark.parametrize("x", (100.0, 200.0, 400.0, 800.0))
def test_the_survival_is_NATIVE_because_one_minus_the_cdf_is_destroyed_there(x):
    # Both halves, so the design is proved necessary and not assumed (tranche 15).
    want = scipy_stats.ncx2.sf(x, 4.0, 5.0)
    assert noncentral_chisquare_sf(x, 4.0, 5.0) == pytest.approx(want, rel=1e-11, abs=0.0)
    complement = 1.0 - noncentral_chisquare_cdf(x, 4.0, 5.0)
    assert abs(complement - want) > 0.01 * want, (
        f"at x={x} the complement is supposed to be ruined; if it is not, this fixture no longer "
        f"demonstrates why `sf` exists"
    )


@pytest.mark.parametrize("x,dof,nc", ((5.0, 2.0, 5.0), (20.0, 10.0, 20.0), (300.0, 100.0, 200.0)))
def test_the_two_SUM_to_one_where_neither_is_near_a_boundary(x, dof, nc):
    # A requirement, true for all parameters, and it does NOT telescope: the two are summed from
    # different incomplete-gamma branches, so agreement is evidence about both.
    total = noncentral_chisquare_cdf(x, dof, nc) + noncentral_chisquare_sf(x, dof, nc)
    assert total == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("x,dof", ((5.0, 2.0), (50.0, 0.5), (1.0, 7.0), (120.0, 100.0)))
def test_a_ZERO_noncentrality_is_the_CENTRAL_chi_squared(x, dof):
    # The reduction to an already-known distribution, and the companion for it is every nc > 0 case
    # above: a reduction a no-op also satisfies would be vacuous (§4/T39).
    assert noncentral_chisquare_cdf(x, dof, 0.0) == pytest.approx(
        scipy_stats.chi2.cdf(x, dof), rel=1e-12
    )
    assert noncentral_chisquare_sf(x, dof, 0.0) == pytest.approx(
        scipy_stats.chi2.sf(x, dof), rel=1e-12, abs=0.0
    )


def test_the_noncentrality_actually_MOVES_the_answer_so_the_reduction_is_not_vacuous():
    # The non-degeneracy premise for the test above.
    flat = noncentral_chisquare_cdf(20.0, 10.0, 0.0)
    shifted = noncentral_chisquare_cdf(20.0, 10.0, 20.0)
    assert abs(shifted - flat) > 0.3, (flat, shifted)


def test_the_cdf_is_MONOTONE_in_x_and_the_survival_is_its_mirror():
    previous_cdf, previous_sf = -1.0, 2.0
    for x in (0.1, 1.0, 5.0, 10.0, 25.0, 60.0):
        cdf, sf = (
            noncentral_chisquare_cdf(x, 4.0, 5.0),
            noncentral_chisquare_sf(x, 4.0, 5.0),
        )
        assert cdf > previous_cdf and sf < previous_sf, x
        previous_cdf, previous_sf = cdf, sf


def test_the_boundaries_are_exact_not_approached():
    assert noncentral_chisquare_cdf(0.0, 4.0, 5.0) == 0.0
    assert noncentral_chisquare_sf(0.0, 4.0, 5.0) == 1.0
    assert noncentral_chisquare_cdf(-1.0, 4.0, 5.0) == 0.0
    assert noncentral_chisquare_sf(-1.0, 4.0, 5.0) == 1.0
    assert noncentral_chisquare_cdf(1e12, 4.0, 5.0) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("dof,nc", ((0.0, 5.0), (-1.0, 5.0), (4.0, -1.0)))
def test_a_non_positive_dof_or_a_negative_noncentrality_is_REFUSED(dof, nc):
    with pytest.raises(ValueError):
        noncentral_chisquare_cdf(5.0, dof, nc)
    with pytest.raises(ValueError):
        noncentral_chisquare_sf(5.0, dof, nc)


def test_a_non_finite_argument_is_REFUSED_rather_than_returning_nan():
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError):
            noncentral_chisquare_cdf(bad, 4.0, 5.0)


# ============================================================================================
# The shared Poisson ladder, extracted at tranche 77's second consumer (§6's rule of two).
# ============================================================================================

def test_the_ladder_sums_POISSON_WEIGHTS_to_one_when_the_term_is_one():
    # The weights are a probability mass function, so a constant summand returns that constant —
    # which is the cheapest complete check that the walk visits every term with the right weight.
    from pricebook.foundation.noncentral_chisquare import poisson_weighted_sum

    for intensity in (0.1, 1.0, 5.0, 40.0, 300.0, 1500.0):
        assert poisson_weighted_sum(intensity, lambda index: 1.0) == pytest.approx(
            1.0, rel=1e-12, abs=0.0
        )


def test_the_ladder_reproduces_the_POISSON_MEAN_which_the_constant_test_cannot_see():
    # `sum_j P(j)*j = intensity/2`. A walk that visited every term with equal weight would pass the
    # test above and fail this one, so the pair pins the weights and not merely their total.
    from pricebook.foundation.noncentral_chisquare import poisson_weighted_sum

    for intensity in (0.5, 4.0, 60.0, 900.0):
        assert poisson_weighted_sum(intensity, float) == pytest.approx(
            0.5 * intensity, rel=1e-10, abs=0.0
        )


def test_the_ladder_matches_the_INVERSE_central_moment_tranche_77_prices_on():
    # `E[1/chi'^2_d(lambda)] = sum_j P(j)/(d + 2j - 2)`, against scipy's own non-central chi-squared
    # by quadrature — a different route to the same number, and the one the 3/2 model needs.
    from pricebook.foundation.noncentral_chisquare import poisson_weighted_sum

    for dof, intensity in ((54.0, 200.0), (10.0, 3.0), (6.0, 0.5)):
        ladder = poisson_weighted_sum(intensity, lambda j: 1.0 / (dof + 2.0 * j - 2.0))
        grid = numpy.linspace(1e-9, dof + intensity + 40.0 * math.sqrt(dof + 2.0 * intensity), 400_001)
        density = scipy_stats.ncx2.pdf(grid, dof, intensity)
        quadrature = float(numpy.trapezoid(density / grid, grid))
        assert ladder == pytest.approx(quadrature, rel=1e-6, abs=0.0)


@pytest.mark.parametrize("intensity", [0.0, -1.0, math.inf, math.nan])
def test_a_NON_POSITIVE_or_non_finite_intensity_is_REFUSED(intensity):
    # The T33b input-population rule: every present caller guards before calling, so this branch is
    # unreachable from the engines and a mutation dropping it survived the whole suite — §4's third
    # survivor kind. Reached here by supplying the input the guard exists to reject.
    from pricebook.foundation.noncentral_chisquare import poisson_weighted_sum

    with pytest.raises(ValueError, match="positive intensity"):
        poisson_weighted_sum(intensity, lambda index: 1.0)
