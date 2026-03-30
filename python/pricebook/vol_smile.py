"""
Volatility smile: strike-dependent vol at a single expiry.

Given market-observed (strike, vol) pairs, interpolates to produce
vol(strike) at any strike. Cubic spline in strike space with flat
extrapolation at the wings.

    smile = VolSmile(
        strikes=[90, 95, 100, 105, 110],
        vols=[0.25, 0.22, 0.20, 0.22, 0.25],
    )
    v = smile.vol(strike=97.5)  # interpolated
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


class VolSmile:
    """Vol as a function of strike at a single expiry.

    Args:
        strikes: sorted list of strikes (at least 2).
        vols: corresponding implied vols (must be non-negative).
    """

    def __init__(self, strikes: list[float], vols: list[float]):
        if len(strikes) != len(vols):
            raise ValueError("strikes and vols must have the same length")
        if len(strikes) < 2:
            raise ValueError("need at least 2 strike/vol pairs")
        for v in vols:
            if v < 0:
                raise ValueError(f"vols must be non-negative, got {v}")

        self._strikes = np.array(strikes, dtype=float)
        self._vols = np.array(vols, dtype=float)

        order = np.argsort(self._strikes)
        self._strikes = self._strikes[order]
        self._vols = self._vols[order]

        self._spline = CubicSpline(
            self._strikes, self._vols, bc_type="clamped",
        )

    def vol(self, strike: float) -> float:
        """Implied vol at the given strike. Flat extrapolation at wings."""
        if strike <= self._strikes[0]:
            return float(self._vols[0])
        if strike >= self._strikes[-1]:
            return float(self._vols[-1])
        return float(self._spline(strike))

    @property
    def strikes(self) -> np.ndarray:
        return self._strikes.copy()

    @property
    def vols(self) -> np.ndarray:
        return self._vols.copy()
