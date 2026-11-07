"""The normal CDF adapter -- testing OUR wrapper, not scipy's numerics.

The distinction matters: scipy is a pinned dependency whose correctness is not this library's
claim. What is this library's claim is that the adapter exposes it faithfully, and in particular
that the symmetry the cap/floor's put-call-parity oracle leans on holds exactly.
"""

import math

import pytest

from pricebook.foundation.distributions import (
    _acklam_rational,
    inverse_norm_cdf,
    norm_cdf,
)


def test_phi_of_zero_is_one_half() -> None:
    assert norm_cdf(0.0) == 0.5


def test_the_symmetry_the_parity_ORACLE_depends_on() -> None:
    """`Phi(x) + Phi(-x) = 1`. The cap/floor parity `cap - floor = N*psi*Pn*(m - K)` follows from
    exactly this, so it is asserted here rather than assumed there."""
    for x in (0.001, 0.1, 0.5, 1.0, 2.5, 6.0):
        assert norm_cdf(x) + norm_cdf(-x) == pytest.approx(1.0, abs=1e-15)


def test_it_is_monotone_and_its_tails_saturate() -> None:
    xs = [-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0]
    values = [norm_cdf(x) for x in xs]
    assert values == sorted(values)
    assert norm_cdf(-40.0) == pytest.approx(0.0, abs=1e-15)
    assert norm_cdf(40.0) == pytest.approx(1.0, abs=1e-15)


def test_the_familiar_quantiles() -> None:
    """Textbook values, as a sanity check that the adapter is not silently shifted or scaled."""
    assert norm_cdf(1.0) == pytest.approx(0.8413447461, abs=1e-9)
    assert norm_cdf(1.959963985) == pytest.approx(0.975, abs=1e-9)
    assert norm_cdf(-2.326347874) == pytest.approx(0.01, abs=1e-9)


# ---------------------------------------------------------------------------------------------
# Tranche 20 — the inverse normal CDF. ACKLAM-2000, recommended by JAECKEL-2002 §9.3 and printed
# in neither. This is the piece tranche 15 deliberately deferred: *"the spec said to transcribe
# `Φ⁻¹` from Jäckel ch 9. Jäckel does not print it."* The acquisition closed that.
#
# The governing idea is that **Acklam's coefficients are untrusted until they agree with an
# inversion of our own exact `norm_cdf`.** A rational approximation is 21 transcribed magic numbers;
# a single mistyped digit gives a function that still looks like `Φ⁻¹` — monotone, odd, roughly
# right — and would pass every structural check. So the oracle's anchor is an independent root-find,
# and the published accuracy is MEASURED rather than taken on the author's word.
# ---------------------------------------------------------------------------------------------


def _inverse_by_root_find(p: float) -> float:
    """`Φ⁻¹(p)` by bracketing the root of `norm_cdf(x) − p` — the independent reference.

    Deliberately a *different method*, not a second transcription: it uses only `norm_cdf` (exact,
    via `math.erf`) and Brent, so it shares no coefficient, no branch structure and no break-point
    with Acklam. That is what makes it an independent check on 21 magic numbers — §4's independence
    rule, where the usual move (read the formula twice) is powerless because both readings would
    copy the same digits."""
    from pricebook.foundation.solvers import bracketed_root

    return bracketed_root(lambda x: norm_cdf(x) - p, -40.0, 40.0)


@pytest.mark.parametrize(
    "p", [1e-12, 1e-6, 0.001, 0.02424, 0.02425, 0.02426, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97575, 0.999, 1 - 1e-12]
)
def test_T20_the_inverse_agrees_with_an_INDEPENDENT_ROOT_FIND_of_our_own_norm_cdf(p):
    """The anchor: two methods sharing nothing but the CDF itself.

    The sweep straddles **both break-points** (`p_low = 0.02425` and `p_high = 1 − p_low`) at
    `±1e-5`, because a rational approximation's regions join there and a wrong branch condition is
    invisible anywhere else — the degenerate-fixture rule applied to a piecewise formula: sample
    *where the pieces meet*, not only in their interiors."""
    assert inverse_norm_cdf(p) == pytest.approx(_inverse_by_root_find(p), rel=1e-11, abs=1e-11)


def test_T20_the_ROUND_TRIP_holds_to_machine_precision():
    """`Φ(Φ⁻¹(p)) = p` — tranche 15's standing deferred oracle, now collectable.

    A requirement oracle: it is the definition of an inverse, so it survives any transcription error
    in the coefficients, and it is the one check that would still mean something if Acklam's page
    itself were wrong."""
    for i in range(1, 400):
        p = i / 400.0
        assert norm_cdf(inverse_norm_cdf(p)) == pytest.approx(p, rel=1e-14, abs=1e-15)


def test_T20_the_inverse_reproduces_the_STANDARD_QUANTILES():
    """Known values, to the digits they are known to — the textbook quantiles every reader can check.

    Independent of both implementations: these are not read back from either, and a systematic error
    in the coefficients that somehow survived the root-find comparison would still have to reproduce
    1.959963984540054 to fifteen digits."""
    assert inverse_norm_cdf(0.5) == 0.0
    assert inverse_norm_cdf(0.975) == pytest.approx(1.959963984540054, rel=1e-13)
    assert inverse_norm_cdf(0.95) == pytest.approx(1.6448536269514722, rel=1e-13)
    assert inverse_norm_cdf(0.99) == pytest.approx(2.3263478740408408, rel=1e-13)
    assert inverse_norm_cdf(0.025) == pytest.approx(-1.959963984540054, rel=1e-13)


def test_T20_the_inverse_is_ODD_about_one_half():
    """`Φ⁻¹(1−p) = −Φ⁻¹(p)` — a requirement from `Φ`'s own symmetry, so it survives a transcription
    error. It also pins the two tail branches against each other: Acklam's upper region is the lower
    region negated, so an asymmetry here is a wrong sign in one of the two branches."""
    for p in (1e-10, 1e-4, 0.01, 0.02425, 0.1, 0.3, 0.49):
        assert inverse_norm_cdf(1.0 - p) == pytest.approx(-inverse_norm_cdf(p), rel=1e-11, abs=1e-12)


def test_T20_the_inverse_is_STRICTLY_MONOTONE_which_is_what_SOBOL_requires():
    """Not a nicety — it is the reason this tranche exists before Sobol.

    A low-discrepancy sequence fills the unit cube evenly; that evenness survives into the variates
    only under a **monotone** transform. Box-Muller is not one (it consumes two uniforms through a
    sine and a square root and scrambles the ordering), which is why the inversion had to be built
    first. A non-monotone `Φ⁻¹` would silently destroy the property Sobol is adopted for, and no
    price-level oracle would show it."""
    previous = -math.inf
    for i in range(1, 2_000):
        value = inverse_norm_cdf(i / 2_000.0)
        assert value > previous
        previous = value


def test_T20_ACKLAMS_PUBLISHED_ACCURACY_is_MEASURED_not_taken_on_trust():
    """*"An algorithm with a relative error less than 1.15 × 10⁻⁹ in the entire region."*

    Measured against the root-find on the **unrefined** rational approximation, which is the only
    way this claim can be checked — and checking it is what proves the 21 coefficients were
    transcribed correctly. A single wrong digit would not break monotonicity or oddness, but it
    would blow this bound by orders of magnitude.

    The companion half: **one Halley step must then reach machine precision**, as Acklam states. So
    the pair is 'the approximation is as good as advertised' and 'the refinement does what it is
    for' — neither alone would catch a refinement that silently did nothing."""
    worst_raw = 0.0
    worst_refined = 0.0
    for i in range(1, 2_000):
        p = i / 2_000.0
        exact = _inverse_by_root_find(p)
        raw = _acklam_rational(p)
        worst_raw = max(worst_raw, abs(raw - exact) / abs(exact) if exact != 0.0 else abs(raw))
        refined = inverse_norm_cdf(p)
        worst_refined = max(
            worst_refined, abs(refined - exact) / abs(exact) if exact != 0.0 else abs(refined)
        )

    assert worst_raw < 1.15e-9, f"the rational approximation is worse than published: {worst_raw:.3e}"
    assert worst_raw > 1e-12, (
        f"the unrefined approximation is {worst_raw:.3e} — too good to be unrefined, so this is "
        f"measuring the refined value and the oracle has no power over the coefficients"
    )
    assert worst_refined < 1e-13, f"one Halley step did not reach machine precision: {worst_refined:.3e}"


def test_T20_the_REFINEMENT_is_necessary_and_the_wrong_variant_is_caught():
    """§4's 'assert the wrong variant fails' pattern, on the refinement rather than a formula.

    Acklam's Halley step is `x ← x − u/(1 + x·u/2)`. The **Newton** step is `x ← x − u`, which is
    the same expression with the third-order correction dropped — a plausible simplification that
    still converges, just more slowly. It must leave a measurably worse answer, or the Halley
    denominator is unoracled surface."""
    worst_halley, worst_newton = 0.0, 0.0
    for i in range(1, 500):
        p = i / 500.0
        exact = _inverse_by_root_find(p)
        start = _acklam_rational(p)
        error = norm_cdf(start) - p
        u = error * math.sqrt(2.0 * math.pi) * math.exp(start * start / 2.0)
        halley = start - u / (1.0 + start * u / 2.0)
        newton = start - u
        worst_halley = max(worst_halley, abs(halley - exact))
        worst_newton = max(worst_newton, abs(newton - exact))
    assert worst_halley < worst_newton
    assert worst_halley < 1e-14


def test_T20_a_PROBABILITY_OUTSIDE_the_open_unit_interval_is_refused():
    """`Φ⁻¹` is defined on `(0,1)`; the endpoints are `±∞`, not large numbers.

    Refused rather than clamped, and this is the guard tranche 15's §4 ruling moved here: a redraw
    loop in `UniformSource` guarding against an exact `0.0` was removed as unreachable dead code
    precisely *because* the domain check belongs at `Φ⁻¹`, which a test can call with `0.0` directly.
    This is that test."""
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="must lie strictly between 0 and 1"):
            inverse_norm_cdf(bad)
    with pytest.raises(ValueError, match="not finite"):
        inverse_norm_cdf(float("nan"))


def test_T20_the_FAR_TAILS_are_computed_rather_than_overflowing():
    """Acklam's lower region uses `q = √(−2 ln p)`, which stays finite for any representable `p`, so
    the deep tail is a real domain rather than a place the function gives up. Checked against the
    round-trip, since a root-find on `[-40, 40]` cannot reach `Φ⁻¹(1e-300) ≈ -37.0` reliably enough
    to be the reference out there."""
    for p in (1e-20, 1e-50, 1e-200, 1e-300):
        x = inverse_norm_cdf(p)
        assert math.isfinite(x) and x < 0.0
        assert norm_cdf(x) == pytest.approx(p, rel=1e-9)
