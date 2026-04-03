"""AAD-aware interpolation: linear and log-linear with Number-valued y."""

from __future__ import annotations

import numpy as np

from pricebook.aad import Number


def aad_linear_interp(x: float, xs: list[float], ys: list[Number]) -> Number:
    """Piecewise linear interpolation with Number-valued y.

    Args:
        x: query point (plain float).
        xs: knot x-values (plain floats, sorted increasing).
        ys: knot y-values (Number, on tape).

    Returns:
        Interpolated Number whose adjoint graph links to the y[i] inputs.
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs and ys must have the same length")
    if n < 2:
        raise ValueError("need at least 2 points")

    # Flat extrapolation
    if x <= xs[0]:
        return ys[0] * 1.0  # put on tape
    if x >= xs[-1]:
        return ys[-1] * 1.0

    # Find segment
    i = 0
    for j in range(n - 1):
        if xs[j + 1] >= x:
            i = j
            break

    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] * (1.0 - t) + ys[i + 1] * t


def aad_log_linear_interp(x: float, xs: list[float], ys: list[Number]) -> Number:
    """Log-linear interpolation with Number-valued y.

    Interpolates linearly in log(y), then exponentiates.
    Standard for discount factor curves.
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs and ys must have the same length")
    if n < 2:
        raise ValueError("need at least 2 points")

    # Flat extrapolation
    if x <= xs[0]:
        return ys[0] * 1.0
    if x >= xs[-1]:
        return ys[-1] * 1.0

    # Find segment
    i = 0
    for j in range(n - 1):
        if xs[j + 1] >= x:
            i = j
            break

    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    log_y0 = ys[i].log()
    log_y1 = ys[i + 1].log()
    log_interp = log_y0 * (1.0 - t) + log_y1 * t
    return log_interp.exp()
