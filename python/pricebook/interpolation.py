"""
Interpolation methods for curve construction.

Implements univariate interpolation with support for extrapolation.
Methods range from simple linear to monotone cubic (shape-preserving).
"""

import math
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import CubicSpline


class InterpolationMethod(Enum):
    LINEAR = "linear"
    LOG_LINEAR = "log_linear"
    CUBIC_SPLINE = "cubic_spline"
    MONOTONE_CUBIC = "monotone_cubic"


class Interpolator(ABC):
    """Base interpolator. Flat extrapolation beyond boundaries."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) < 2:
            raise ValueError("need at least 2 points")
        if not np.all(np.diff(x) > 0):
            raise ValueError("x must be strictly increasing")

        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)

    @abstractmethod
    def _interpolate(self, x: float) -> float:
        """Interpolate at a point within the domain."""

    def __call__(self, x: float) -> float:
        """Evaluate at x with flat extrapolation."""
        if x <= self._x[0]:
            return float(self._y[0])
        if x >= self._x[-1]:
            return float(self._y[-1])
        return self._interpolate(x)

    def _find_segment(self, x: float) -> int:
        """Find the index i such that x[i] <= x < x[i+1]."""
        idx = int(np.searchsorted(self._x, x)) - 1
        return max(0, min(idx, len(self._x) - 2))


class LinearInterpolator(Interpolator):
    """
    Piecewise linear interpolation.

    Simple and fast. Produces continuous but non-smooth curves.
    Forward rates will be piecewise constant with jumps at knot points.
    """

    def _interpolate(self, x: float) -> float:
        i = self._find_segment(x)
        x0, x1 = self._x[i], self._x[i + 1]
        y0, y1 = self._y[i], self._y[i + 1]
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)


class LogLinearInterpolator(Interpolator):
    """
    Log-linear interpolation: linear in log(y).

    The standard method for discount factors. Interpolating linearly in
    log(df) is equivalent to assuming piecewise constant forward rates
    between knot points. This is the most common choice in practice.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__(x, y)
        if np.any(self._y <= 0):
            raise ValueError("y values must be positive for log-linear interpolation")
        self._log_y = np.log(self._y)

    def _interpolate(self, x: float) -> float:
        i = self._find_segment(x)
        x0, x1 = self._x[i], self._x[i + 1]
        log_y0, log_y1 = self._log_y[i], self._log_y[i + 1]
        t = (x - x0) / (x1 - x0)
        return math.exp(log_y0 + t * (log_y1 - log_y0))


class CubicSplineInterpolator(Interpolator):
    """
    Natural cubic spline interpolation.

    Produces smooth curves with continuous first and second derivatives.
    Can produce negative forward rates in some configurations — use
    monotone cubic if this is a concern.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__(x, y)
        self._spline = CubicSpline(self._x, self._y, bc_type="natural")

    def _interpolate(self, x: float) -> float:
        return float(self._spline(x))


class MonotoneCubicInterpolator(Interpolator):
    """
    Monotone-preserving cubic Hermite interpolation (Hyman filter).

    Uses Fritsch-Carlson slopes with the Hyman monotonicity constraint.
    Preserves the shape of the data and prevents overshoots, which is
    critical for discount curves (no negative forward rates).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__(x, y)
        self._slopes = self._compute_slopes()

    def _compute_slopes(self) -> np.ndarray:
        """Fritsch-Carlson method with Hyman filter."""
        n = len(self._x)
        h = np.diff(self._x)
        delta = np.diff(self._y) / h

        # Interior slopes: harmonic mean of adjacent secants
        slopes = np.zeros(n)
        for i in range(1, n - 1):
            if delta[i - 1] * delta[i] > 0:
                # Harmonic mean
                slopes[i] = 2.0 * delta[i - 1] * delta[i] / (delta[i - 1] + delta[i])
            else:
                slopes[i] = 0.0

        # Endpoint slopes: one-sided
        slopes[0] = delta[0]
        slopes[-1] = delta[-1]

        # Hyman filter: enforce alpha^2 + beta^2 <= 9
        for i in range(n - 1):
            if abs(delta[i]) < 1e-30:
                slopes[i] = 0.0
                slopes[i + 1] = 0.0
            else:
                alpha = slopes[i] / delta[i]
                beta = slopes[i + 1] / delta[i]
                r2 = alpha * alpha + beta * beta
                if r2 > 9.0:
                    tau = 3.0 / math.sqrt(r2)
                    slopes[i] = tau * alpha * delta[i]
                    slopes[i + 1] = tau * beta * delta[i]

        return slopes

    def _interpolate(self, x: float) -> float:
        i = self._find_segment(x)
        x0, x1 = self._x[i], self._x[i + 1]
        y0, y1 = self._y[i], self._y[i + 1]
        m0, m1 = self._slopes[i], self._slopes[i + 1]

        h = x1 - x0
        t = (x - x0) / h

        # Hermite basis functions
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2

        return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


def create_interpolator(method: InterpolationMethod, x, y) -> Interpolator:
    """Factory function to create an interpolator by method enum."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if method == InterpolationMethod.LINEAR:
        return LinearInterpolator(x, y)
    elif method == InterpolationMethod.LOG_LINEAR:
        return LogLinearInterpolator(x, y)
    elif method == InterpolationMethod.CUBIC_SPLINE:
        return CubicSplineInterpolator(x, y)
    elif method == InterpolationMethod.MONOTONE_CUBIC:
        return MonotoneCubicInterpolator(x, y)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
