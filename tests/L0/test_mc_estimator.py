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

from pricebook.foundation.mc_estimator import estimate


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


def test_the_standard_error_FALLS_as_one_over_root_N():
    """The rate the whole MC oracle rests on: quadrupling the sample halves the SE. Asserted on a
    deterministic sample so there is no sampling noise to argue about."""
    import itertools

    base = [1.0, -1.0]
    se = {}
    for reps in (1, 4, 16):
        sample = list(itertools.chain.from_iterable([base] * reps))
        se[reps] = estimate(sample).standard_error
    assert se[4] == pytest.approx(se[1] / 2.0, rel=1e-12)
    assert se[16] == pytest.approx(se[1] / 4.0, rel=1e-12)


def test_the_estimate_is_FROZEN():
    out = estimate([1.0, 2.0])
    with pytest.raises(Exception):
        out.mean = 9.0   # type: ignore[misc]


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
