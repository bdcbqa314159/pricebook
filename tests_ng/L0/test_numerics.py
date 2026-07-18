"""Distributions, solvers, interpolation oracles (L0) — Topic 0 Slice 6.

Finance-free numerics: the standard normal, root-finding + Nelder-Mead, and an
interpolation MECHANISM (curve extrapolation policy is L1, not here).
"""

import math

import pytest

from pricebook_ng.foundation.distributions import norm_cdf, norm_pdf, norm_ppf
from pricebook_ng.foundation.interpolation import Interpolation, interpolate
from pricebook_ng.foundation.solvers import bisect_root, nelder_mead


def test_norm_cdf_pdf():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert norm_cdf(1.0) == pytest.approx(0.8413447460685429, abs=1e-10)
    assert norm_cdf(-1.0) == pytest.approx(0.15865525393145707, abs=1e-10)
    assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2 * math.pi), abs=1e-12)


def test_norm_ppf_inverts_cdf():
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-7)
    assert norm_cdf(norm_ppf(0.3)) == pytest.approx(0.3, abs=1e-9)


def test_bisect_root():
    assert bisect_root(lambda x: x - 2.0, 0.0, 5.0) == pytest.approx(2.0, abs=1e-10)


def test_nelder_mead_minimises():
    x = nelder_mead(lambda v: (v[0] - 3.0) ** 2 + (v[1] + 1.0) ** 2, [0.0, 0.0])
    assert x[0] == pytest.approx(3.0, abs=1e-4)
    assert x[1] == pytest.approx(-1.0, abs=1e-4)


def test_linear_interpolation():
    xs, ys = [0.0, 1.0, 2.0], [0.0, 10.0, 20.0]
    assert interpolate(xs, ys, 0.5, Interpolation.LINEAR) == pytest.approx(5.0, abs=1e-12)
    assert interpolate(xs, ys, 1.5, Interpolation.LINEAR) == pytest.approx(15.0, abs=1e-12)


def test_log_linear_interpolation():
    # log-linear on discount-factor-like ys: interpolate log(y) linearly
    xs, ys = [0.0, 2.0], [1.0, math.exp(-0.1)]  # exp of a linear log => y at x=1 is exp(-0.05)
    assert interpolate(xs, ys, 1.0, Interpolation.LOG_LINEAR) == pytest.approx(math.exp(-0.05), abs=1e-12)


def test_interpolation_no_extrapolation():
    with pytest.raises(ValueError):
        interpolate([0.0, 1.0], [0.0, 1.0], 2.0, Interpolation.LINEAR)  # extrapolation is L1
