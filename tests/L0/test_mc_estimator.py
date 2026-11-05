"""The Monte Carlo estimator: sample mean and standard error (L0). T15.

`estimate(samples)` returns the sample mean and `sigma_hat / sqrt(N)`. Finance-free statistics, and
the piece that lets an MC oracle assert `|MC - closed form| < k*SE` with the SE **measured** rather
than a tolerance invented to make a test pass.

**The oracles are closed-form, not statistical.** For a sample whose mean and variance are known by
hand, both outputs are known exactly — so a wrong divisor (`N` vs `N-1`), a missing square root, or
a forgotten `sqrt(N)` shows immediately. Those three are what an estimator transcription gets wrong.
"""

import math

import pytest

from pricebook.foundation.mc_estimator import (
    MonteCarloEstimate,
    antithetic_estimate,
    control_variate_estimate,
    estimate,
)


def test_the_mean_of_a_hand_computable_sample():
    out = estimate([1.0, 2.0, 3.0, 4.0])
    assert out.mean == pytest.approx(2.5, abs=1e-15)


def test_the_standard_error_uses_the_SAMPLE_variance_and_sqrt_N():
    """`[1,2,3,4]`: sample variance (N-1) is 5/3, so `SE = sqrt(5/3)/2 = 0.6454972...`. Using the
    population divisor `N` would give 0.559, and omitting `sqrt(N)` would give 1.29 — both caught."""
    out = estimate([1.0, 2.0, 3.0, 4.0])
    assert out.standard_error == pytest.approx(math.sqrt(5.0 / 3.0) / 2.0, rel=1e-15)
    assert out.standard_error != pytest.approx(math.sqrt(1.25) / 2.0, rel=1e-6)
    assert out.standard_error != pytest.approx(math.sqrt(5.0 / 3.0), rel=1e-6)


def test_a_CONSTANT_sample_has_zero_standard_error():
    out = estimate([3.5] * 100)
    assert out.mean == pytest.approx(3.5, abs=1e-15)
    assert out.standard_error == pytest.approx(0.0, abs=1e-15)


def test_the_standard_error_has_the_EXACT_closed_form_one_over_root_N_minus_one():
    """**My first version of this asserted `SE(4N) = SE(N)/2` and was arithmetically wrong.** On the
    sample `[1,-1]` repeated, every deviation is +/-1 so the squared deviations sum to `N`, and with
    the `N-1` divisor the sample variance is `N/(N-1)` — itself still moving with `N`. So

        SE = sqrt( (N/(N-1)) / N ) = 1 / sqrt(N - 1)

    exactly, not `1/sqrt(N)`: the clean inverse-root-N law is asymptotic. Asserting the closed form
    is the stronger oracle anyway — it pins the `N-1` divisor and the `sqrt(N)` *together*, where the
    ratio test pinned only their combination."""
    import itertools
    import math

    for reps in (1, 4, 16, 64):
        sample = list(itertools.chain.from_iterable([[1.0, -1.0]] * reps))
        out = estimate(sample)
        n = out.n_samples
        assert out.standard_error == pytest.approx(1.0 / math.sqrt(n - 1), rel=1e-14), n

    # And the asymptotic law, asserted EXACTLY rather than as "about 2". Quadrupling 256 samples
    # to 1024 gives a ratio of `sqrt(1023/255) = 2.00294...`, not 2 -- so a `rel=1e-3` band around
    # 2.0 fails by a hair, as my first attempt did. The exact closed form needs no band at all, and
    # the residual gap from 2 is the `N-1` divisor's visible footprint.
    small = estimate(list(itertools.chain.from_iterable([[1.0, -1.0]] * 128)))
    large = estimate(list(itertools.chain.from_iterable([[1.0, -1.0]] * 512)))
    ratio = small.standard_error / large.standard_error
    assert ratio == pytest.approx(math.sqrt(1023.0 / 255.0), rel=1e-12)
    assert abs(ratio - 2.0) < 0.01, "the asymptotic law should hold to ~1% by N = 256"


def test_a_SINGLE_sample_has_no_standard_error_and_says_so():
    """One draw cannot estimate a spread. Returning 0.0 would claim certainty; a `nan` would
    propagate silently — so it refuses."""
    with pytest.raises(ValueError, match="at least two"):
        estimate([1.0])


def test_an_EMPTY_sample_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        estimate([])


def test_a_NON_FINITE_sample_is_refused_not_averaged():
    """A `nan` payoff would make the mean `nan` and the SE `nan` — a price-shaped non-answer. The
    stock's migration-hardening lesson (non-finite-fitness poisoning), as a pre-registered oracle."""
    with pytest.raises(ValueError, match="not finite"):
        estimate([1.0, 2.0, float("nan")])
    with pytest.raises(ValueError, match="not finite"):
        estimate([1.0, float("inf")])


# ---------------------------------------------------------------------------------------------
# Tranche 19 — variance reduction. JAECKEL-2002 §10.1 (antithetic) and §10.3 (control variates).
#
# Every oracle here is a PAIR, which is the tranche's governing discipline: the reduced estimator
# must be UNBIASED (same expectation) and must REDUCE variance (measured). Neither half alone is
# the technique — a reduction that moves the price is worse than no reduction at all.
# ---------------------------------------------------------------------------------------------


def _call(x: float) -> float:
    """Monotone and convex in the draw — so antithetic reduces, but not to zero."""
    return max(x - 0.5, 0.0)


def _square(x: float) -> float:
    """SYMMETRIC in the draw, so the mirror is identical: `Cov[v, ṽ] > 0` and eq 10.1 fails. This is
    the case that proves the reduction condition is doing work rather than always holding."""
    return x * x


def _normals(count: int, seed: int = 7) -> list[float]:
    from pricebook.foundation.rng import UniformSource
    from pricebook.foundation.variates import standard_normals

    draws = UniformSource(seed=seed).uniforms(2 * count + 2 * count % 2)
    return list(standard_normals(draws)[:count])


# ============================================================================================
# ANTITHETIC SAMPLING — §10.1
# ============================================================================================


def test_T19_antithetic_is_EXACT_on_a_payoff_LINEAR_in_the_draw():
    """The sharpest antithetic check, and it is exact rather than statistical.

    For `v(z) = a·z + b`, the pairwise average `½(v(z) + v(−z))` is `b` for **every** draw, so the
    estimator has zero variance. No path count is involved — this is algebra, and it fails
    immediately if the mirror is not `−z` or the `½` is dropped.

    **Zero in exact arithmetic, ~1e-18 in floating point, and the oracle says the second.** A first
    version of this test asserted `standard_error == 0.0` and failed at `1.5e-18`: `3z + 1.5` and
    `−3z + 1.5` each round, so their sum is not bit-exactly `3.0` for every draw. The residual is
    machine noise on the *summands*, not variance in the estimator, so the assertion is a
    relative-to-the-mean bound at the floating-point floor — the charter's tolerance rule applied to
    a quantity that is exact in ℝ and cannot be in `float`."""
    z = _normals(2_000)
    values = [3.0 * x + 1.5 for x in z]
    mirrored = [3.0 * (-x) + 1.5 for x in z]

    reduced = antithetic_estimate(values, mirrored)
    assert reduced.mean == pytest.approx(1.5, rel=1e-15)
    assert reduced.standard_error < 1e-15 * abs(reduced.mean)
    assert reduced.n_samples == len(z)


def test_T19_antithetic_REDUCES_but_not_to_zero_on_a_CONVEX_payoff():
    """The mandatory companion: a reduction that is exact on a linear payoff might be a no-op in
    general. A call is convex, so the mirror does not cancel — the variance falls but stays positive,
    and the reduction is real rather than an artefact of linearity."""
    z = _normals(20_000)
    plain = estimate([_call(x) for x in z])
    reduced = antithetic_estimate([_call(x) for x in z], [_call(-x) for x in z])

    assert reduced.standard_error > 0.0
    assert reduced.standard_error < plain.standard_error
    # Unbiased: both estimate the same expectation, within the plain estimator's own error bar.
    assert abs(reduced.mean - plain.mean) < 4.0 * plain.standard_error


def test_T19_the_antithetic_STANDARD_ERROR_comes_from_the_PAIRWISE_averages():
    """Jäckel's §10.1 subtlety, and the one an implementation gets wrong by being helpful.

    *"If instead of viewing both `v_i = v(z_i)` and `ṽ_i = v(−z_i)` as individual samples, we only
    count the pairwise average `v̄_i` as an individual sample, all is well again, because the pairwise
    averages are independent."* Pooling all `2N` values and calling `estimate` on them would report a
    **smaller** standard error from samples that are not independent — a confident wrong error bar,
    which is worse than a large one. Asserted by showing the two differ and that `n_samples` is `N`,
    not `2N`."""
    z = _normals(4_000)
    values, mirrored = [_call(x) for x in z], [_call(-x) for x in z]

    reduced = antithetic_estimate(values, mirrored)
    pooled = estimate(values + mirrored)

    assert reduced.n_samples == len(z)
    assert pooled.n_samples == 2 * len(z)
    assert reduced.standard_error != pytest.approx(pooled.standard_error, rel=1e-6)


def test_T19_antithetic_reduces_variance_EXACTLY_WHEN_the_pair_covariance_is_negative():
    """Eqs 10.1/10.2, transcribed: the reduction condition is `V[½(v+ṽ)] < ½V[v]`, **equivalent** to
    `Cov[v, ṽ] < 0` — and the `½` on the right is not decoration.

    It is there because an antithetic sample costs *two* evaluations, so the fair comparison is
    variance per unit of work, not per sample. A monotone payoff always satisfies it (Jäckel);
    a symmetric one (`v(z) = z²`, where the mirror is identical) has `Cov > 0` and antithetic makes
    things worse per unit of work — the case that proves the condition is doing something."""
    z = _normals(20_000)
    for payoff, monotone in ((_call, True), (_square, False)):
        values = [payoff(x) for x in z]
        mirrored = [payoff(-x) for x in z]
        pairs = [0.5 * (a + b) for a, b in zip(values, mirrored, strict=True)]

        pair_variance = _variance(pairs)
        half_plain = 0.5 * _variance(values)
        covariance = _covariance(values, mirrored)

        # The equivalence of 10.1 and 10.2, asserted as an equivalence rather than assumed.
        assert (pair_variance < half_plain) is (covariance < 0.0)
        assert (covariance < 0.0) is monotone


def _variance(samples) -> float:
    mean = math.fsum(samples) / len(samples)
    return math.fsum((s - mean) ** 2 for s in samples) / (len(samples) - 1)


def _covariance(left, right) -> float:
    n = len(left)
    ml, mr = math.fsum(left) / n, math.fsum(right) / n
    return math.fsum((a - ml) * (b - mr) for a, b in zip(left, right, strict=True)) / (n - 1)


# ============================================================================================
# CONTROL VARIATES — §10.3, eqs 10.8 and 10.9
# ============================================================================================


def _correlated(count: int, rho: float, seed: int = 11):
    """`(X, Y)` with `Y` standard normal and `corr(X, Y) = rho` by construction, `E[Y] = 0` exactly.

    Built from two independent normal streams so the target correlation is a *premise* of the
    fixture rather than something measured off it — the degenerate-fixture guard the spec names:
    `rho = 1` would zero the variance trivially and hide a centring error, `rho = 0` reduces
    nothing, so the oracles below use a realistic intermediate value."""
    a = _normals(count, seed)
    b = _normals(count, seed + 1)
    y = a
    x = [rho * p + math.sqrt(1.0 - rho * rho) * q for p, q in zip(a, b, strict=True)]
    return x, y


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.9])
def test_T19_the_control_variate_is_UNBIASED_and_reduces_variance_by_ONE_MINUS_RHO_SQUARED(rho):
    """Eq 10.8's estimator, eq 10.9's `β*`, and the variance factor — which Jäckel does NOT print.

    `v̂_CV = (1/n) Σ [v(u_i) + β(g* − g(u_i))]` (10.8) with `β* = Cov[v,g]/V[g]` (10.9). Both
    renderings agree on those. The resulting variance `V[v](1 − ρ²)` is **absent from the source**,
    so it is not transcribed — it is *derived* from 10.9 in two lines
    (`V[v] + β²V[g] − 2βCov` at `β = β*` collapses to `V[v] − Cov²/V[g]`) and asserted as a theorem.
    §4 blesses exactly this: an earlier step of a derivation is an independent check on the result,
    and a theorem is stronger than a second reading of one line.

    Unbiasedness is the anchor and comes first: `E[g] = g*` is known exactly, so eq 10.8 adds a
    mean-zero term and cannot move the expectation."""
    x, y = _correlated(40_000, rho)
    reduced = control_variate_estimate(x, y, control_mean=0.0)

    # (10.9), pointwise.
    assert reduced.beta == pytest.approx(_covariance(x, y) / _variance(y), rel=1e-14)
    assert reduced.correlation == pytest.approx(rho, abs=0.02)

    # The DERIVED variance factor, as a theorem rather than a transcription.
    plain = estimate(x)
    expected = plain.standard_error * math.sqrt(1.0 - reduced.correlation**2)
    assert reduced.standard_error == pytest.approx(expected, rel=1e-12)

    # Unbiased: the adjusted mean agrees with the plain mean inside the plain error bar.
    assert abs(reduced.mean - plain.mean) < 4.0 * plain.standard_error
    assert reduced.standard_error < plain.standard_error


def test_T19_the_control_must_be_centred_on_its_EXACT_mean_or_the_estimator_is_BIASED():
    """The failure mode that makes the technique dangerous, asserted rather than warned about.

    Eq 10.8 is unbiased **only** because `g*` is the exact `E[g]`. Centring on an *approximate* mean
    shifts the estimate by `β·(g* − E[g])` — a bias that no amount of sampling removes. So a control
    whose mean is merely a good approximation is not a valid control, however well it correlates."""
    x, y = _correlated(40_000, 0.9)
    honest = control_variate_estimate(x, y, control_mean=0.0)
    wrong = control_variate_estimate(x, y, control_mean=0.05)

    # Eq 10.8 ADDS `β(g* − g)`, so raising `g*` RAISES the estimate. (A first version of this test
    # had the sign backwards; the direction is part of what is being pinned.)
    shift = honest.beta * 0.05
    assert wrong.mean == pytest.approx(honest.mean + shift, rel=1e-12)
    # The bias is many standard errors wide — it would not be mistaken for noise, it would be
    # mistaken for a price.
    assert abs(wrong.mean - honest.mean) > 10.0 * honest.standard_error


def test_T19_a_SUPPLIED_beta_is_used_and_beta_ONE_is_still_unbiased():
    """`β = 1` is the provably-unbiased simple choice: it needs no estimate from the samples, so it
    carries none of the `O(1/N)` bias Jäckel warns of, at the cost of being optimal only when
    `V[g] = Cov[v,g]`. The estimator must honour a supplied `β` rather than silently re-fitting."""
    x, y = _correlated(40_000, 0.9)
    unit = control_variate_estimate(x, y, control_mean=0.0, beta=1.0)
    assert unit.beta == 1.0
    plain = estimate(x)
    assert abs(unit.mean - plain.mean) < 4.0 * plain.standard_error
    # Still a reduction at this correlation, but not as good as the optimal beta.
    optimal = control_variate_estimate(x, y, control_mean=0.0)
    assert optimal.standard_error <= unit.standard_error


def test_T19_an_UNCORRELATED_control_reduces_nothing_and_a_CONSTANT_one_is_refused():
    """Non-vacuity at both ends. At `ρ ≈ 0` the factor `1 − ρ²` is ≈ 1, so the technique is a no-op —
    worth asserting so that a passing reduction test cannot be explained by luck. And a control with
    zero variance makes `β* = Cov/0` undefined: refused, not returned as an infinity."""
    x, y = _correlated(40_000, 0.0)
    plain, reduced = estimate(x), control_variate_estimate(x, y, control_mean=0.0)
    assert reduced.standard_error == pytest.approx(plain.standard_error, rel=0.05)

    with pytest.raises(ValueError, match="no variance"):
        control_variate_estimate([1.0, 2.0, 3.0], [5.0, 5.0, 5.0], control_mean=5.0)


def test_T19_the_reducers_REFUSE_ragged_or_non_finite_input():
    with pytest.raises(ValueError, match="same length"):
        antithetic_estimate([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="same length"):
        control_variate_estimate([1.0, 2.0], [1.0], control_mean=0.0)
    with pytest.raises(ValueError, match="not finite"):
        antithetic_estimate([1.0, float("nan")], [1.0, 2.0])
    with pytest.raises(ValueError, match="not finite"):
        control_variate_estimate([1.0, 2.0], [1.0, float("inf")], control_mean=0.0)
