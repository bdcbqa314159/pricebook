"""The normal CDF adapter -- testing OUR wrapper, not scipy's numerics.

The distinction matters: scipy is a pinned dependency whose correctness is not this library's
claim. What is this library's claim is that the adapter exposes it faithfully, and in particular
that the symmetry the cap/floor's put-call-parity oracle leans on holds exactly.
"""

import math

import pytest

from pricebook.foundation.distributions import (
    _A,
    _B,
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
# Tranche 20c — the RELATIVE-accuracy oracle `norm_cdf` never had (OI-3).
#
# `norm_cdf` shipped at tranche 5 as `0.5·(1 + erf(x/√2))` and was green for fifteen tranches, because
# its oracle asserted the tails "saturate" — an ABSOLUTE claim, which passes however bad the relative
# error becomes where the true value is ~0. In the lower tail `1 + erf` cancels catastrophically as
# `erf → −1`, destroying the significant digits of a small probability.
#
# This is the oracle that would have caught it, written before the fix.
# ---------------------------------------------------------------------------------------------


def test_T20c_norm_cdf_is_accurate_in_RELATIVE_terms_across_the_whole_range():
    """The point of the tranche: a tail probability must be right *as a probability*, not merely small.

    Checked against `scipy.special.ndtr` — an independent high-precision implementation, kept off the
    production path by tranche 11's arrangement (`distributions.py` moved off scipy *"so the oracle and
    the implementation no longer share a provider"*). §4's "a reference is code too" applies and is
    satisfied: scipy has the tail accuracy our own root-find did not.

    **`x = −7.03` is the pre-registered failure.** The old form carries 6.1e-06 relative error there
    against 7.4e-15 for the `erfc` form — nine orders of magnitude. The bound below is 1e-13, loose
    enough to be about the *form* rather than about the last bits, and tight enough that the
    cancelling form cannot pass it anywhere below about `x = −6`."""
    from scipy.special import ndtr

    worst = 0.0
    worst_at = None
    for tenths in range(-90, 91):
        x = tenths / 10.0
        exact = float(ndtr(x))
        relative = abs(norm_cdf(x) - exact) / exact
        if relative > worst:
            worst, worst_at = relative, x
    assert worst < 1e-13, f"worst RELATIVE error {worst:.3e} at x={worst_at}"

    # The single point that motivated the tranche, asserted on its own so the regression is named.
    exact = float(ndtr(-7.03))
    assert abs(norm_cdf(-7.03) - exact) / exact < 1e-13


def test_T20c_the_DEEP_tail_keeps_its_significant_digits():
    """Further out than any pricer needs, because the question is whether the *form* is right.

    At `x = -30` the probability is ~5e-198 — still a normal `float`, and still a number with sixteen
    significant digits that the computation either keeps or throws away. The superseded form returns
    exactly `0.0` here: not a small probability but a lost one.

    **The bound loosens with depth, and that is the REFERENCE's accuracy rather than ours** — §4's
    "a reference is code too", which this tranche's own charter note states. `math.erfc` and scipy's
    `ndtr` are different implementations computing a value spanning three hundred orders of magnitude,
    and their *relative* agreement decays slowly: 1.6e-14 at `x = -10`, 6.4e-14 at `-20`, 2.1e-13 at
    `-37`. Asserting 1e-13 everywhere (as a first version did) is asserting the reference is better
    than it is — it failed at `x = -30` with 1.74e-13."""
    from scipy.special import ndtr

    for x, bound in ((-10.0, 1e-13), (-15.0, 1e-13), (-20.0, 1e-13), (-30.0, 1e-12), (-37.0, 1e-12)):
        exact = float(ndtr(x))
        assert exact > 0.0, "premise: scipy must resolve this, or the comparison is vacuous"
        assert norm_cdf(x) > 0.0, f"norm_cdf({x}) collapsed to zero — the digits are gone"
        assert abs(norm_cdf(x) - exact) / exact < bound


def test_T20c_we_resolve_FURTHER_than_the_reference_does():
    """Where the oracle runs out before the code does — worth asserting so it is not mistaken later
    for a limit of ours.

    At `x = -38` scipy's `ndtr` underflows to **exactly 0.0** while `erfc(-x/sqrt2)/2` still returns a
    subnormal (~2.9e-316). So beyond about `x = -37` there is no reference to compare against, and the
    honest statement is the structural one: the value stays positive and monotone down to where the
    `float` itself gives out. A test that simply stopped at `-37` would leave the impression that is
    where *we* stop."""
    from scipy.special import ndtr

    assert float(ndtr(-38.0)) == 0.0, "if scipy resolves this now, use it as the reference here too"
    assert norm_cdf(-38.0) > 0.0
    assert norm_cdf(-38.0) < norm_cdf(-37.0)
    # The floating-point floor, not an error: below about -39 even a subnormal cannot hold it.
    assert norm_cdf(-50.0) == 0.0


def test_T20c_the_change_AGREES_where_the_old_form_was_SOUND_and_the_rest_is_ITS_error():
    """The §4 pure-refactor proof, as a measured **partition** rather than a single tolerance.

    `erfc(-y) ≡ 1 + erf(y)` in ℝ, so the two forms are the same quantity and only their
    floating-point behaviour differs. Measured, not asserted:

    * **`x ≥ 0`: agreement to 1 ULP** (bit-identical at 88% of a fine grid over `[0, 6]`, never worse
      than one). Both forms are accurate there — 2.19e-16 versus 2.50e-16 against `scipy.special.ndtr`
      — because `erf > 0` means `1 + erf` never cancels. A first version of this oracle claimed
      *bit*-identity here and failed at `x = 1.2`: both routes produce a value near 1 and round
      differently in the last bit, which is rounding, not error. Worth recording, because "identical"
      was the claim I wanted and one ULP is the claim that is true.
    * down to `x = -4`: tens of ULP at worst (`-3` is the outlier at 36, a relative 5.8e-15).
    * **`x ≲ -5`: the forms part company — 3.9e-11 relative at `-5`, 1.3e-10 at `-6` — and that is the
      OLD form's error, not this change's noise.** Attributed rather than merely noted: both are
      compared against the independent reference below, and only one of them tracks it.

    So the usual "agreement bounded to a few ULP" framing would be **false** here, and claiming it
    would be the §4 violation this rule exists to prevent. The partition also bounds the blast radius
    for the nineteen tranches of oracles that read `norm_cdf`: anything evaluating `Φ` above about
    `x = -4` moves by at most tens of ULP, which no tolerance in the suite can see."""
    from scipy.special import ndtr

    def superseded(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # The non-negative half: one ULP, and both accurate.
    for t in range(0, 1_201):
        x = t / 200.0
        assert norm_cdf(x) == pytest.approx(superseded(x), rel=3e-16), f"moved at x={x}"
    assert norm_cdf(0.0) == 0.5

    # Down to -4: rounding only.
    for x in (-1.0, -2.0, -3.0, -4.0):
        assert norm_cdf(x) == pytest.approx(superseded(x), rel=1e-14)

    # Below that they diverge, and the divergence is the OLD form's error — which is the claim that
    # makes this a fix rather than a drift.
    for x in (-5.0, -6.0, -7.03):
        exact = float(ndtr(x))
        assert abs(norm_cdf(x) - exact) / exact < 1e-14, "the new form must track the reference"
        assert abs(superseded(x) - exact) / exact > 1e-11, "the old form must not"


def test_T20c_the_OLD_form_FAILS_this_oracle_which_is_why_the_tranche_exists():
    """§4's 'assert the wrong variant fails' pattern, on the form being replaced.

    Without this, the oracle above could be passing for reasons unrelated to the change — a bound that
    both forms satisfy would prove nothing. So the superseded expression is evaluated here directly and
    required to breach it, which also fixes the measurement in the record: **6.1e-06 at `x = −7.03`**,
    and exactly `0.0` far enough out."""
    from scipy.special import ndtr

    def cancelling_form(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    exact = float(ndtr(-7.03))
    old_relative = abs(cancelling_form(-7.03) - exact) / exact
    assert old_relative > 1e-7, (
        f"the superseded form is only {old_relative:.3e} wrong at x=-7.03 — then this tranche is "
        f"not fixing what it claims to"
    )
    # Far enough out the cancelling form does not merely lose precision, it returns zero.
    assert cancelling_form(-20.0) == 0.0
    assert norm_cdf(-20.0) > 0.0

    # And the two agree in the centre, which is the other half: the change is a tail fix, not a
    # different function.
    for x in (-3.0, -1.0, 0.0, 1.0, 3.0):
        assert norm_cdf(x) == pytest.approx(cancelling_form(x), rel=1e-15)


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

    # Built on `norm_cdf`, which **since tranche 20c is** the `erfc` form. At tranche 20a this line
    # had to call a private `erfc` sibling instead, because `norm_cdf`'s `(1+erf)/2` form cancelled as
    # `erf -> -1` and the deep tail disagreed by 3.9e-07 — with the REFERENCE as the wrong side. Fixing
    # the cause (OI-3) removed the workaround: there is one CDF again, and it is the accurate one.
    # Still independent of Acklam — no coefficient, branch or break-point in common.
    return bracketed_root(lambda x: norm_cdf(x) - p, -40.0, 40.0)


@pytest.mark.parametrize(
    "p",
    [1e-12, 1e-6, 0.001, 0.02424, 0.02425, 0.02426, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97575, 0.999],
)
def test_T20_the_inverse_agrees_with_an_INDEPENDENT_ROOT_FIND_of_our_own_norm_cdf(p):
    """The anchor: two methods sharing nothing but the CDF itself.

    The sweep straddles **both break-points** (`p_low = 0.02425` and `p_high = 1 − p_low`) at
    `±1e-5`, because a rational approximation's regions join there and a wrong branch condition is
    invisible anywhere else — the degenerate-fixture rule applied to a piecewise formula: sample
    *where the pieces meet*, not only in their interiors.

    **The sweep stops at 0.999 at the top, and that is a limit of the REFERENCE, not of the code.**
    For `p` near 1 the root-find must solve `Phi(x) = 1 - eps`, and `Phi` is flat at 1 in double
    precision, so the bracket cannot resolve a root — the reference becomes unresolvable long before
    the implementation becomes wrong. The far upper tail is covered instead by the round-trip and by
    the symmetry reduction, which maps it onto the lower tail exactly."""
    # 1e-13, not tighter: the ROOT-FIND is the limiting side (it misses an independent
    # implementation by up to 3.74e-15 where ours is sub-ULP), so a tighter bound here would be
    # asserting Brent's tolerance rather than our accuracy. The sharp comparison is the scipy one.
    assert inverse_norm_cdf(p) == pytest.approx(_inverse_by_root_find(p), rel=1e-13, abs=1e-13)


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


def test_T20_the_inverse_is_ODD_about_one_half_EXACTLY_where_the_inputs_allow_it():
    """`Φ⁻¹(1−p) = −Φ⁻¹(p)` — and the interesting part is *where this can be exact at all*.

    The implementation reduces `p > 1/2` to `1 − p` and negates, so oddness is exact **in
    arithmetic**. But testing it requires `1 − (1 − p)` to return `p`, and in binary floating point
    that holds only when `1 − p` is representable without loss — i.e. for **dyadic** `p`. At
    `p = 0.1`, `1 − (1 − 0.1)` is `0.09999999999999998`, so the two calls are given genuinely
    different inputs and must give different answers.

    A first version of this test asserted oddness at `p = 1e-10` and failed: at that end `1 − p`
    retains only about four digits of `p`, so the asymmetry it measured was in the **inputs**, not
    in the function. Split accordingly — exact on dyadics, and elsewhere bounded by the input's own
    representation loss rather than by a tolerance picked to pass."""
    for p in (0.5, 0.25, 0.125, 0.0625, 0.03125, 1.0 / 1024.0):
        assert 1.0 - (1.0 - p) == p, "premise: this p must round-trip through 1-p exactly"
        assert inverse_norm_cdf(1.0 - p) == -inverse_norm_cdf(p)

    for p in (1e-4, 0.01, 0.02425, 0.1, 0.3, 0.49):
        loss = abs((1.0 - (1.0 - p)) - p) / p
        assert inverse_norm_cdf(1.0 - p) == pytest.approx(
            -inverse_norm_cdf(p), rel=max(1e-14, 20.0 * loss)
        )


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


def test_T20_the_inverse_matches_an_INDEPENDENT_IMPLEMENTATION_to_sub_ULP():
    """The strongest cross-check available, and it is stronger than the root-find above.

    `scipy.special.ndtri` is an independent *implementation* of `Φ⁻¹` — a different algorithm, a
    different provider, no shared coefficient, break-point or refinement. The charter already makes
    this the right shape: `distributions.py` moved off scipy at tranche 11 precisely so *"the oracle
    and the implementation no longer share a provider"*, and this is that arrangement paying off.

    **Measured worst miss across 2,000 points: 7.66e-16 — sub-ULP — and bit-exact at many points.**
    Which matters because it settles a disagreement the other direction: the Brent root-find
    reference misses scipy by up to 3.74e-15, so where the two references differ, *ours tracks
    scipy and the root-find is the weaker side*. That happened three times in this tranche (the
    cancelling `norm_cdf`, the unresolvable upper tail, and here), which is its own lesson: a
    reference is not automatically the authority just because a test calls it one."""
    from scipy.special import ndtri

    def miss(got: float, want: float) -> float:
        return abs(got - want) / max(abs(want), 1.0)

    worst = 0.0
    for i in range(1, 2_000):
        p = i / 2_000.0
        worst = max(worst, miss(inverse_norm_cdf(p), float(ndtri(p))))
    assert worst < 2e-15, f"worst miss against an independent implementation: {worst:.3e}"

    # And in the deep tails, where the root-find reference cannot follow.
    for p in (1e-20, 1e-50, 1e-200, 1e-300):
        assert miss(inverse_norm_cdf(p), float(ndtri(p))) < 1e-14


def test_T20_ACKLAMS_PUBLISHED_ACCURACY_is_MEASURED_not_taken_on_trust():
    """*"An algorithm with a relative error less than 1.15 × 10⁻⁹ in the entire region."*

    Measured against the root-find on the **unrefined** rational approximation, which is the only
    way this claim can be checked — and checking it is what proves the 21 coefficients were
    transcribed correctly. A single wrong digit would not break monotonicity or oddness, but it
    would blow this bound by orders of magnitude.

    The companion half: **one Halley step must then reach machine precision**, as Acklam states. So
    the pair is 'the approximation is as good as advertised' and 'the refinement does what it is
    for' — neither alone would catch a refinement that silently did nothing."""
    # **Relative-where-relative-means-something.** `Phi^-1` passes through zero at `p = 1/2`, so a
    # plain relative error explodes near the middle for reasons that have nothing to do with the
    # coefficients — a first version of this oracle measured 1.4e-13 and blamed the refinement. The
    # honest metric is relative in the tails and absolute near zero, which is what dividing by
    # `max(|x|, 1)` gives.
    def miss(got: float, want: float) -> float:
        return abs(got - want) / max(abs(want), 1.0)

    from scipy.special import ndtri

    worst_raw = 0.0
    worst_refined = 0.0
    for i in range(1, 2_000):
        p = i / 2_000.0
        exact = float(ndtri(p))      # the independent implementation, not the root-find
        worst_raw = max(worst_raw, miss(_acklam_rational(p), exact))
        worst_refined = max(worst_refined, miss(inverse_norm_cdf(p), exact))

    # **1.1288e-09 measured against a published 1.15e-09** — 98% of the bound, and that tightness
    # is the oracle's grip on the 21 coefficients. Measured resolution: a coefficient perturbation
    # of ~20 ULP (4e-15 relative) moves this into the remaining headroom, and 4.5e-9 relative sends
    # it to 2.4e-05. Below ~20 ULP the Halley refinement simply repairs the error — two such
    # mutations were proved to leave the output BIT-IDENTICAL — so that is the resolution, not a
    # hole: there is nothing finer for an oracle to see.
    assert worst_raw < 1.15e-9, f"the rational approximation is worse than published: {worst_raw:.3e}"
    assert worst_raw > 1e-10, (
        f"the unrefined approximation is {worst_raw:.3e}, far better than published — that is not "
        f"Acklam's rational form, so the oracle has lost its grip on the coefficients"
    )
    assert worst_refined < 2e-15, f"one Halley step did not reach machine precision: {worst_refined:.3e}"


def _one_step(start: float, p: float, third_order: bool) -> float:
    """One refinement step from `start`, with or without Halley's third-order denominator."""
    error = norm_cdf(start) - p
    u = error * math.sqrt(2.0 * math.pi) * math.exp(start * start / 2.0)
    return start - u / (1.0 + start * u / 2.0) if third_order else start - u


def test_T20_the_published_accuracy_bound_CAN_FAIL_which_is_what_gives_it_grip():
    """§4's prove-the-band-can-fail rule, applied to the thing that guards 21 transcribed numbers.

    `worst_raw < 1.15e-9` is the only oracle with any purchase on the coefficients — the refined
    output is repaired by one Halley step and is therefore nearly indifferent to them, which is good
    engineering and bad for verification. So the bound's power is demonstrated rather than assumed:
    a coefficient perturbed by 4.5e-9 *relative* (one part in 220 million) must push the measured
    error clear of the bound the oracle enforces. Measured: **4.40e-08, which is 38x the 1.15e-9
    bound** — so a transcription error of that size cannot pass. (A first version of this assertion
    demanded 1e-06, extrapolated from a full-sweep measurement rather than the central region this
    coefficient governs; the prediction was wrong, the oracle was not.)

    This also states the oracle's limit honestly. Perturbations below ~20 ULP vanish into the
    refinement — measured, with two of them leaving the output bit-identical — so the transcription's
    low-order digits are genuinely not load-bearing. The claim worth making is "the coefficients are
    right to the precision that affects the answer", not "every digit is verified"."""
    from scipy.special import ndtri

    perturbed = (_A[0], _A[1] + 1e-6, _A[2], _A[3], _A[4], _A[5])

    def rational_with(a_coefficients, p: float) -> float:
        """Acklam's central branch only — the region the perturbed coefficient governs."""
        q = p - 0.5
        r = q * q
        numerator = (
            (((((a_coefficients[0] * r + a_coefficients[1]) * r + a_coefficients[2]) * r
               + a_coefficients[3]) * r + a_coefficients[4]) * r + a_coefficients[5]) * q
        )
        return numerator / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)

    worst_true, worst_perturbed = 0.0, 0.0
    for i in range(100, 1_900):                  # the central region
        p = i / 2_000.0
        exact = float(ndtri(p))
        scale = max(abs(exact), 1.0)
        worst_true = max(worst_true, abs(_acklam_rational(p) - exact) / scale)
        worst_perturbed = max(worst_perturbed, abs(rational_with(perturbed, p) - exact) / scale)

    assert worst_true < 1.15e-9
    assert worst_perturbed > 10.0 * 1.15e-9, (
        f"a 4.5e-9 relative coefficient error only moved the measured accuracy to "
        f"{worst_perturbed:.3e} — the published bound has no grip on the transcription"
    )


def test_T20_HALLEYS_THIRD_ORDER_TERM_is_NOT_measurable_from_Acklams_start():
    """An honest negative about the source's own choice, and it needed measuring to see.

    Acklam specifies **Halley's** rational method, `x ← x − u/(1 + x·u/2)`. Dropping the denominator
    gives **Newton**, `x ← x − u`. From a start already accurate to 1.15e-9, Newton squares the error
    to ~1e-18 — below double precision — so **both land on the same answer**: worst absolute miss
    4.44e-16 either way, across the whole lower half. The third-order term cannot be falsified at
    this starting accuracy.

    So the oracle asserts what is true rather than what would be flattering: the two agree to
    machine precision here. A first version of this test asserted Halley was *measurably better* and
    failed with `4.44e-16 < 4.44e-16` — the assertion was wishful, not the code wrong.

    **The denominator is still transcribed, not dropped**, because the source specifies it and the
    cost is one multiply-add; and its necessity is demonstrated in the companion below, from a start
    where third order *does* bite. Without that companion it would be unfalsifiable surface — §4's
    unreachable-branch rule, which asks that the check be relocated somewhere a test can reach."""
    worst_gap = 0.0
    for i in range(1, 250):          # p <= 1/2, where the implementation does its refining
        p = i / 500.0
        start = _acklam_rational(p)
        worst_gap = max(worst_gap, abs(_one_step(start, p, True) - _one_step(start, p, False)))
    assert worst_gap < 1e-15, f"the two steps differ by {worst_gap:.3e} — then order IS measurable"


def test_T20_HALLEYS_THIRD_ORDER_TERM_DOES_bite_from_a_DEGRADED_start():
    """The companion that makes the Halley denominator reachable from a test.

    Third order only shows when the starting error is large enough that `error²` is still above
    machine precision. Acklam's start is too good for that, so the step is exercised from a
    deliberately degraded one — a 1% perturbation — where Newton's quadratic convergence and
    Halley's cubic separate by orders of magnitude. That is what proves `1 + x·u/2` is doing
    arithmetic rather than sitting unverified."""
    worst_halley, worst_newton = 0.0, 0.0
    for i in range(1, 200):
        p = i / 400.0
        exact = _inverse_by_root_find(p)
        degraded = _acklam_rational(p) * 1.01
        worst_halley = max(worst_halley, abs(_one_step(degraded, p, True) - exact))
        worst_newton = max(worst_newton, abs(_one_step(degraded, p, False) - exact))
    assert worst_halley < worst_newton / 10.0, (
        f"Halley {worst_halley:.3e} vs Newton {worst_newton:.3e} — the third-order term is not "
        f"separating even from a degraded start, so it is unverified"
    )


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
