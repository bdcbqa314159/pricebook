"""Distributions, solvers, interpolation oracles (L0) — Topic 0 gate Slice 5 (S17).

Finance-free numerics are **thin scipy adapters** (S17 ratified scipy): the standard normal
(`scipy.stats.norm`), root-finding + least-squares (`scipy.optimize`), and interpolation
(`scipy.interpolate` for cubic/PCHIP/Akima; linear/log-linear ours). Extrapolation policy is
stated **per end** (`FLAT | CONTINUE_SLOPE | RAISE`) — closing C4's silent divergence. The
hand-rolled `bisect_root`/`nelder_mead` are gone (no duplicates).
"""

import math

import pytest

from pricebook_ng.foundation.distributions import norm_cdf, norm_pdf, norm_ppf
from pricebook_ng.foundation.interpolation import (
    Extrapolation,
    ExtrapolationEnds,
    Interpolation,
    interpolate,
)
from pricebook_ng.foundation.solvers import brent, least_squares, newton, secant


# ── distributions (scipy.stats.norm) ──
def test_norm_cdf_pdf():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert norm_cdf(1.0) == pytest.approx(0.8413447460685429, abs=1e-10)
    assert norm_cdf(-1.0) == pytest.approx(0.15865525393145707, abs=1e-10)
    assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2 * math.pi), abs=1e-12)


def test_norm_ppf_inverts_cdf():
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-7)
    assert norm_cdf(norm_ppf(0.3)) == pytest.approx(0.3, abs=1e-9)
    with pytest.raises(ValueError):
        norm_ppf(0.0)


# ── solvers (scipy.optimize) ──
def test_brent_bracketed_root():
    assert brent(lambda x: x * x - 2.0, 0.0, 5.0) == pytest.approx(math.sqrt(2.0), abs=1e-12)


def test_newton_with_derivative():
    # x^2 - 2 = 0, f' = 2x
    assert newton(lambda x: x * x - 2.0, 1.0, lambda x: 2.0 * x) == pytest.approx(math.sqrt(2.0), abs=1e-12)


def test_secant_no_derivative():
    assert secant(lambda x: x * x - 2.0, 1.0, 2.0) == pytest.approx(math.sqrt(2.0), abs=1e-10)


def test_least_squares_lm():
    # fit a, b to y = a*x + b through (0,1),(1,3),(2,5) → a=2, b=1
    xs, ys = [0.0, 1.0, 2.0], [1.0, 3.0, 5.0]
    def resid(p):
        return [p[0] * x + p[1] - y for x, y in zip(xs, ys)]
    a, b = least_squares(resid, [0.0, 0.0])
    assert a == pytest.approx(2.0, abs=1e-9)
    assert b == pytest.approx(1.0, abs=1e-9)


# ── interpolation (in-range) ──
def test_linear_interpolation():
    xs, ys = [0.0, 1.0, 2.0], [0.0, 10.0, 20.0]
    assert interpolate(xs, ys, 0.5, Interpolation.LINEAR) == pytest.approx(5.0, abs=1e-12)
    assert interpolate(xs, ys, 1.5, Interpolation.LINEAR) == pytest.approx(15.0, abs=1e-12)


def test_log_linear_interpolation():
    xs, ys = [0.0, 2.0], [1.0, math.exp(-0.1)]
    assert interpolate(xs, ys, 1.0, Interpolation.LOG_LINEAR) == pytest.approx(math.exp(-0.05), abs=1e-12)


def test_cubic_pchip_akima_pass_through_nodes():
    xs, ys = [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 8.0, 27.0]  # y = x^3
    for method in (Interpolation.CUBIC_SPLINE, Interpolation.MONOTONE_CUBIC, Interpolation.AKIMA):
        for x, y in zip(xs, ys):
            assert interpolate(xs, ys, x, method) == pytest.approx(y, abs=1e-9)
    # cubic spline reproduces x^3 well at an interior point
    assert interpolate(xs, ys, 1.5, Interpolation.CUBIC_SPLINE) == pytest.approx(3.375, abs=0.3)


def test_monotone_cubic_does_not_overshoot():
    # PCHIP preserves monotonicity where a natural cubic would ring
    xs, ys = [0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]
    y = interpolate(xs, ys, 1.5, Interpolation.MONOTONE_CUBIC)
    assert y == pytest.approx(0.0, abs=1e-12)  # flat region stays flat, no overshoot below 0


# ── extrapolation policy, stated per end (C4) ──
def test_default_extrapolation_raises_both_ends():
    with pytest.raises(ValueError):
        interpolate([0.0, 1.0], [0.0, 1.0], 2.0, Interpolation.LINEAR)   # right, default RAISE
    with pytest.raises(ValueError):
        interpolate([0.0, 1.0], [0.0, 1.0], -1.0, Interpolation.LINEAR)  # left, default RAISE


def test_flat_extrapolation():
    xs, ys = [0.0, 1.0, 2.0], [10.0, 20.0, 30.0]
    ends = ExtrapolationEnds(left=Extrapolation.FLAT, right=Extrapolation.FLAT)
    assert interpolate(xs, ys, -5.0, Interpolation.LINEAR, ends) == pytest.approx(10.0)  # first y
    assert interpolate(xs, ys, 9.0, Interpolation.LINEAR, ends) == pytest.approx(30.0)   # last y


def test_continue_slope_extrapolation():
    xs, ys = [0.0, 1.0, 2.0], [0.0, 10.0, 20.0]  # slope 10
    ends = ExtrapolationEnds(left=Extrapolation.CONTINUE_SLOPE, right=Extrapolation.CONTINUE_SLOPE)
    assert interpolate(xs, ys, 3.0, Interpolation.LINEAR, ends) == pytest.approx(30.0, abs=1e-6)
    assert interpolate(xs, ys, -1.0, Interpolation.LINEAR, ends) == pytest.approx(-10.0, abs=1e-6)


def test_per_end_policies_can_differ():
    xs, ys = [0.0, 1.0, 2.0], [10.0, 20.0, 30.0]
    ends = ExtrapolationEnds(left=Extrapolation.FLAT, right=Extrapolation.RAISE)
    assert interpolate(xs, ys, -3.0, Interpolation.LINEAR, ends) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        interpolate(xs, ys, 5.0, Interpolation.LINEAR, ends)  # right end still raises
