"""The least-squares solver, behind our own API (L0).

A thin scipy adapter, on the seam `distributions.py` already established: scipy stays behind our
own API so models and calibrators are scipy-free and a port has one swap point. **The oracles test
*this adapter*, never scipy's numerics** -- that it finds the minimum of a problem whose answer is
known by hand, that it reports non-convergence rather than returning a plausible number, and that
it honours bounds. A test asserting scipy's iteration count would be testing the wrong library.
"""

import math

import pytest

from pricebook.foundation.solvers import SolveOutcome, least_squares_fit


def test_finds_the_minimum_of_a_hand_solvable_problem():
    """Residuals `(x0 - 3, x1 + 1)`: the minimiser is exactly `(3, -1)` with zero cost."""
    out = least_squares_fit(lambda x: [x[0] - 3.0, x[1] + 1.0], (0.0, 0.0))
    assert out.converged
    assert out.x[0] == pytest.approx(3.0, abs=1e-10)
    assert out.x[1] == pytest.approx(-1.0, abs=1e-10)
    assert out.cost < 1e-20
    assert out.iterations >= 1


def test_overdetermined_least_squares_is_the_mean():
    """Three residuals `x - 1`, `x - 2`, `x - 3` in one unknown: the least-squares answer is the
    mean, 2, and the cost is the sum of squared deviations halved (scipy's convention), 1.0."""
    out = least_squares_fit(lambda x: [x[0] - 1.0, x[0] - 2.0, x[0] - 3.0], (0.0,))
    assert out.x[0] == pytest.approx(2.0, abs=1e-10)
    assert out.cost == pytest.approx(1.0, rel=1e-9)


def test_bounds_are_honoured():
    """The unconstrained minimiser is 3; bounded above at 1 the answer must sit on the bound."""
    out = least_squares_fit(lambda x: [x[0] - 3.0], (0.5,), bounds=((0.0,), (1.0,)))
    assert out.x[0] == pytest.approx(1.0, abs=1e-8)


def test_a_non_finite_residual_is_reported_not_returned():
    """The failure mode that matters: a residual going NaN must surface as `converged=False`, not
    as a confident-looking `x`. A calibrator turns this into a `PricingFailure`."""
    out = least_squares_fit(lambda x: [math.nan], (1.0,))
    assert isinstance(out, SolveOutcome)
    assert not out.converged

