"""S08 oracle — 1-D bracketed root solver (L0 numerical toolkit).

Bisection on a bracketed sign change: exact roots of known functions, and a
clear failure when the bracket does not straddle a root.
"""

import math

import pytest

from pricebook_ng.foundation.solvers import bisect_root


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
