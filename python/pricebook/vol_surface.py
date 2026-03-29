"""Volatility surface: flat vol and vol term structure."""

from datetime import date

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.interpolation import (
    InterpolationMethod,
    create_interpolator,
    Interpolator,
)


class FlatVol:
    """Constant volatility across all expiries and strikes."""

    def __init__(self, vol: float):
        if vol < 0:
            raise ValueError(f"vol must be non-negative, got {vol}")
        self._vol = vol

    def vol(self, expiry: date | None = None, strike: float | None = None) -> float:
        return self._vol


class VolTermStructure:
    """
    Volatility as a function of expiry (flat across strikes).

    Interpolates between pillar vols. Flat extrapolation at boundaries.
    Strike dimension is anticipated in the interface but not used yet.
    """

    def __init__(
        self,
        reference_date: date,
        expiries: list[date],
        vols: list[float],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
        interpolation: InterpolationMethod = InterpolationMethod.LINEAR,
    ):
        if len(expiries) != len(vols):
            raise ValueError("expiries and vols must have the same length")
        if len(expiries) < 1:
            raise ValueError("need at least 1 expiry")
        for v in vols:
            if v < 0:
                raise ValueError(f"vols must be non-negative, got {v}")

        self.reference_date = reference_date
        self.day_count = day_count

        times = [year_fraction(reference_date, d, day_count) for d in expiries]

        if len(times) == 1:
            # Single pillar: constant vol
            self._single_vol = vols[0]
            self._interpolator = None
        else:
            self._single_vol = None
            self._interpolator: Interpolator = create_interpolator(
                interpolation, np.array(times), np.array(vols),
            )

    def vol(self, expiry: date, strike: float | None = None) -> float:
        """Volatility at the given expiry. Strike ignored (flat smile for now)."""
        if self._single_vol is not None:
            return self._single_vol
        t = year_fraction(self.reference_date, expiry, self.day_count)
        if t <= 0:
            return float(self._interpolator(0.0))
        return float(self._interpolator(t))
