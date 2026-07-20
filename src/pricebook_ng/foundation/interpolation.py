"""Interpolation mechanism — thin scipy adapters + explicit extrapolation policy (L0, S17).

Linear and log-linear are ours (cheap, exact); cubic / monotone-cubic (PCHIP) / Akima are
`scipy.interpolate` behind our own API. **Extrapolation is stated per end** —
`FLAT | CONTINUE_SLOPE | RAISE`, defaulting to RAISE both ends — which closes C4: a curve
never silently extends itself, it must say how each end behaves. `CONTINUE_SLOPE` continues
the end slope in the interpolation's **own space** (log space for `LOG_LINEAR`, so a DF curve
extends a constant continuously-compounded forward and stays positive — audit A1). (Hagan-West monotone-convex
is *not* here — it interpolates from interval averages and is curve construction, Topic 1.)

Provenance:
  quarry: python/pricebook/core/interpolation.py
  source: linear/log-linear (ours); scipy.interpolate CubicSpline/PchipInterpolator/Akima1DInterpolator
  oracle: nodes reproduced; PCHIP no-overshoot; per-end FLAT/CONTINUE_SLOPE/RAISE
  slice:  l0-numerics (Topic 0 gate S17)
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator


class Interpolation(Enum):
    LINEAR = "linear"
    LOG_LINEAR = "log_linear"  # linear in log(y) — discount factors
    CUBIC_SPLINE = "cubic_spline"
    MONOTONE_CUBIC = "monotone_cubic"  # PCHIP — shape-preserving, no overshoot
    AKIMA = "akima"


class Extrapolation(Enum):
    """What an interpolant does past a node range end."""

    FLAT = "flat"  # hold the end value
    CONTINUE_SLOPE = (
        "continue_slope"  # extend linearly at the interpolant's boundary slope
    )
    RAISE = "raise"  # refuse — extrapolation is a deliberate choice


@dataclass(frozen=True)
class ExtrapolationEnds:
    """The extrapolation policy at each end (they may differ). Defaults to RAISE both."""

    left: Extrapolation = Extrapolation.RAISE
    right: Extrapolation = Extrapolation.RAISE


_SCIPY: dict[Interpolation, Callable] = {
    Interpolation.CUBIC_SPLINE: CubicSpline,
    Interpolation.MONOTONE_CUBIC: PchipInterpolator,
    Interpolation.AKIMA: Akima1DInterpolator,
}


def _segment(xs: Sequence[float], x: float) -> int:
    return min(bisect_right(xs, x) - 1, len(xs) - 2)


def _evaluate(
    xs: Sequence[float], ys: Sequence[float], x: float, method: Interpolation
) -> float:
    """Value at `x` strictly inside ``[xs[0], xs[-1]]``."""
    if method in (Interpolation.LINEAR, Interpolation.LOG_LINEAR):
        i = _segment(xs, x)
        w = (x - xs[i]) / (xs[i + 1] - xs[i])
        if method is Interpolation.LINEAR:
            return ys[i] + w * (ys[i + 1] - ys[i])
        return math.exp(math.log(ys[i]) + w * (math.log(ys[i + 1]) - math.log(ys[i])))
    # scipy spline — rebuilt per call. ponytail: a hot curve caches the interpolator itself (Topic 1).
    spline = _SCIPY[method](np.asarray(xs, float), np.asarray(ys, float))
    return float(spline(x))


def _boundary_slope(
    xs: Sequence[float], ys: Sequence[float], method: Interpolation, at_left: bool
) -> float:
    """The interpolant's slope at an end node. LINEAR is exact — the end segment's slope, no
    finite step (mirrors `_boundary_slope_log`). Spline methods take a one-sided step just
    inside the range, since their end slope is not the chord slope."""
    if method is Interpolation.LINEAR:
        if at_left:
            return (ys[1] - ys[0]) / (xs[1] - xs[0])
        return (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    if at_left:
        xb, x_in = xs[0], xs[0] + (xs[1] - xs[0]) * 1e-6
    else:
        xb, x_in = xs[-1], xs[-1] - (xs[-1] - xs[-2]) * 1e-6
    return (_evaluate(xs, ys, x_in, method) - _evaluate(xs, ys, xb, method)) / (
        x_in - xb
    )


def _extrapolate(
    xs: Sequence[float],
    ys: Sequence[float],
    x: float,
    method: Interpolation,
    policy: Extrapolation,
) -> float:
    if policy is Extrapolation.RAISE:
        raise ValueError(f"{x} outside [{xs[0]}, {xs[-1]}] and the end policy is RAISE")
    at_left = x < xs[0]  # which end determines the anchor node
    y_end = ys[0] if at_left else ys[-1]
    if policy is Extrapolation.FLAT:
        return y_end
    x_end = xs[0] if at_left else xs[-1]
    if method is Interpolation.LOG_LINEAR:
        # A1: CONTINUE_SLOPE extends in the interpolation's OWN space — log space here — so
        # a DF curve extrapolates a constant continuously-compounded forward and the extended
        # DFs stay positive (value-space extension gave DF(30) = −0.275; audit 1.2).
        slope_log = _boundary_slope_log(xs, ys, at_left)
        return math.exp(math.log(y_end) + slope_log * (x - x_end))
    return y_end + _boundary_slope(xs, ys, method, at_left) * (x - x_end)


def _boundary_slope_log(
    xs: Sequence[float], ys: Sequence[float], at_left: bool
) -> float:
    """Slope of ``log(y)`` vs ``x`` at an end node (the end segment's log-slope — exact for
    log-linear)."""
    if at_left:
        return (math.log(ys[1]) - math.log(ys[0])) / (xs[1] - xs[0])
    return (math.log(ys[-1]) - math.log(ys[-2])) / (xs[-1] - xs[-2])


def interpolate(
    xs: Sequence[float],
    ys: Sequence[float],
    x: float,
    method: Interpolation,
    ends: ExtrapolationEnds = ExtrapolationEnds(),
) -> float:
    """Interpolate `y` at `x` from ascending `xs`. Outside ``[xs[0], xs[-1]]`` the per-end
    `ends` policy applies (default: raise)."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need matching xs, ys of length >= 2")
    if x < xs[0]:
        return _extrapolate(xs, ys, x, method, ends.left)
    if x > xs[-1]:
        return _extrapolate(xs, ys, x, method, ends.right)
    return _evaluate(xs, ys, x, method)
