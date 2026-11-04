"""Standard normal variates by Box-Muller (L0). JAECKEL-2002 eq 9.6.

    x = sqrt(-2 ln u) * sin(2*pi*v)
    y = sqrt(-2 ln u) * cos(2*pi*v)        for u, v independent uniform on (0,1)^2

**Transcribed in both renderings and they agree exactly** — `-raw` and `-layout` both give eq 9.6 in
this form, and Jaeckel states the joint density of `(x,y)` is *"that of two independent standard
normal variates!"*.

**Ruled over inverse-CDF inversion, and the reason matters.** Jaeckel prefers inversion and
recommends Acklam's or Moro's formula — but **does not print either**; the coefficients are in
`[Ack00]`/`[Mor95]`, `referenced-by` only, and the implementation is "on the accompanying CD" we do
not have. Box-Muller eq 9.6 **is** printed in full, is **exact** rather than an approximation, and
needs no acquisition. Acklam is deferred to the Sobol unit, where a monotone transform genuinely
earns its keep (Box-Muller's trigonometric wrapping does not preserve low-discrepancy structure).

**The anchor oracle is an exact identity, not a statistical one.** Since `sin^2 + cos^2 = 1`,

    x^2 + y^2 = -2 ln u

holds to machine precision for every pair — so a wrong `sqrt`, a missing factor of 2, a `log10` for
`ln`, or `sin`/`cos` swapped into the wrong slots shows immediately and without any sampling noise.
The N(0,1) moment checks and the pair-independence check are the *companions*: they are statistical,
and they are what would catch a transcription that satisfied the identity but mis-assigned the angle
(e.g. `2*pi*v` written as `pi*v`).

**Jaeckel's two caveats, both read and both handled.** His Figure 9.5 warning is against the
*rejection* variant (eq 9.8) combined with low-discrepancy numbers, not against this trigonometric
form; and the Neave effect (§9.3.2) is an interaction with *simple multiplicative congruential*
generators, which PCG64 is not. Neither is assumed away: the N(0,1) and independence oracles below
would fail if either bit us.
"""

import math

import pytest

from pricebook.foundation.rng import UniformSource
from pricebook.foundation.variates import standard_normals


def test_the_exact_identity_x_squared_plus_y_squared_equals_minus_two_ln_u():
    """**The anchor, and it is exact.** `sin^2 + cos^2 = 1`, so each PAIR satisfies
    `x^2 + y^2 = -2 ln u` to machine precision — no sampling noise, nothing to tolerate."""
    source = UniformSource(seed=11)
    uniforms = source.uniforms(200)
    draws = standard_normals(uniforms)
    assert len(draws) == 200
    for i in range(0, 200, 2):
        u = uniforms[i]
        x, y = draws[i], draws[i + 1]
        assert x * x + y * y == pytest.approx(-2.0 * math.log(u), rel=1e-13), i


def test_the_pointwise_transcription_of_eq_9_6():
    """A hand-computed pair, so the formula is checked and not only its invariant."""
    u, v = 0.25, 0.125
    radius = math.sqrt(-2.0 * math.log(u))
    expected = [radius * math.sin(2.0 * math.pi * v), radius * math.cos(2.0 * math.pi * v)]
    assert standard_normals([u, v]) == pytest.approx(expected, rel=1e-15)


def test_a_WRONG_ANGLE_would_satisfy_the_identity_but_not_this():
    """The companion that gives the identity its teeth: `pi*v` instead of `2*pi*v` preserves
    `x^2+y^2` exactly, so the anchor alone cannot see it. The pointwise check can."""
    u, v = 0.25, 0.125
    radius = math.sqrt(-2.0 * math.log(u))
    wrong = [radius * math.sin(math.pi * v), radius * math.cos(math.pi * v)]
    got = standard_normals([u, v])
    assert got != pytest.approx(wrong, rel=1e-6)
    assert sum(g * g for g in got) == pytest.approx(sum(w * w for w in wrong), rel=1e-13), (
        "the wrong angle must still satisfy the identity, or this companion proves nothing"
    )


def test_the_draws_are_STANDARD_NORMAL_in_mean_and_variance():
    """Statistical, with the tolerance MEASURED from the sample size rather than chosen: the mean of
    N standard normals has standard error `1/sqrt(N)`, so 4 SEs is the band."""
    n = 40_000
    draws = standard_normals(UniformSource(seed=3).uniforms(n))
    mean = math.fsum(draws) / n
    variance = math.fsum(d * d for d in draws) / n - mean * mean
    assert abs(mean) < 4.0 / math.sqrt(n), mean
    assert abs(variance - 1.0) < 4.0 * math.sqrt(2.0 / n), variance


def test_the_PAIRED_draws_are_INDEPENDENT():
    """Box-Muller's whole claim is that `x` and `y` are independent, not merely normal. The sample
    correlation of N/2 pairs has standard error `1/sqrt(N/2)`, so again the band is measured."""
    n = 40_000
    draws = standard_normals(UniformSource(seed=4).uniforms(n))
    xs, ys = draws[0::2], draws[1::2]
    pairs = len(xs)
    covariance = math.fsum(a * b for a, b in zip(xs, ys)) / pairs
    assert abs(covariance) < 4.0 / math.sqrt(pairs), covariance


def test_the_TAILS_are_reached_so_the_sample_is_not_a_narrow_band():
    """A generator stuck near zero would pass the moment checks on variance alone only if the
    variance matched — but asserting the tails are populated catches a truncated transform."""
    draws = standard_normals(UniformSource(seed=5).uniforms(40_000))
    assert max(draws) > 3.0 and min(draws) < -3.0, (min(draws), max(draws))
    assert sum(1 for d in draws if abs(d) > 1.96) / len(draws) == pytest.approx(0.05, abs=0.01)


def test_the_SAME_uniforms_give_the_same_normals():
    uniforms = UniformSource(seed=77).uniforms(100)
    assert standard_normals(uniforms) == standard_normals(uniforms)


def test_a_ZERO_uniform_is_REFUSED_because_ln_zero_diverges():
    """**The domain guard, now where it can be exercised.** Tranche 15 moved it here from the uniform
    source, where the branch was statistically unreachable and a mutant breaking it survived all 468
    tests. Here it is one call away."""
    with pytest.raises(ValueError, match="strictly inside"):
        standard_normals([0.0, 0.5])
    with pytest.raises(ValueError, match="strictly inside"):
        standard_normals([1.0, 0.5])


def test_an_ODD_number_of_uniforms_is_REFUSED():
    """Box-Muller consumes uniforms in pairs and produces normals in pairs. Silently dropping the
    last uniform would quietly change the stream a caller believed it was consuming."""
    with pytest.raises(ValueError, match="pairs"):
        standard_normals([0.25, 0.5, 0.75])


def test_an_EMPTY_input_is_refused():
    with pytest.raises(ValueError, match="pairs"):
        standard_normals([])
