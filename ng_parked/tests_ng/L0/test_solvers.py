"""S08 oracle — 1-D bracketed root solver (L0 numerical toolkit).

Bisection on a bracketed sign change: exact roots of known functions, and a
clear failure when the bracket does not straddle a root.
"""

import math

import pytest

from pricebook_ng.foundation.solvers import bisect_root, nelder_mead


def test_finds_sqrt_two():
    root = bisect_root(lambda x: x * x - 2.0, 0.0, 2.0)
    assert root == pytest.approx(math.sqrt(2.0), abs=1e-12)


def test_decreasing_function():
    # exp(-x) - 0.5 = 0 -> x = ln 2
    root = bisect_root(lambda x: math.exp(-x) - 0.5, -1.0, 5.0)
    assert root == pytest.approx(math.log(2.0), abs=1e-12)


def test_no_sign_change_raises():
    with pytest.raises(ValueError):
        bisect_root(lambda x: x * x + 1.0, -1.0, 1.0)  # never crosses zero


# ---- nelder_mead: derivative-free multivariate minimiser ----------------------
def test_nelder_mead_quadratic_bowl():
    # min of (x-3)^2 + (y+1)^2 is (3, -1)
    x = nelder_mead(lambda p: (p[0] - 3.0) ** 2 + (p[1] + 1.0) ** 2, [0.0, 0.0])
    assert x[0] == pytest.approx(3.0, abs=1e-5)
    assert x[1] == pytest.approx(-1.0, abs=1e-5)


def test_nelder_mead_rosenbrock():
    # classic curved valley; global min at (1, 1)
    def rosen(p):
        return (1.0 - p[0]) ** 2 + 100.0 * (p[1] - p[0] ** 2) ** 2

    x = nelder_mead(rosen, [-1.2, 1.0], max_iter=4000)
    assert x[0] == pytest.approx(1.0, abs=1e-3)
    assert x[1] == pytest.approx(1.0, abs=1e-3)
