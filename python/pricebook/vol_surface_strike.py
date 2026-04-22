"""
Volatility surface with strike dimension: vol(expiry, strike).

Multiple VolSmile objects, one per expiry, with linear interpolation
between expiries and flat extrapolation at boundaries.

    surface = VolSurfaceStrike(
        reference_date=date(2024, 1, 15),
        expiries=[date(2024, 7, 15), date(2025, 1, 15)],
        smiles=[smile_6m, smile_1y],
    )
    v = surface.vol(expiry=date(2024, 10, 15), strike=105)
"""

from __future__ import annotations

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.vol_smile import VolSmile


class VolSurfaceStrike:
    """2D vol surface: vol(expiry, strike) via per-expiry smiles.

    Args:
        reference_date: valuation date.
        expiries: list of expiry dates (one per smile).
        smiles: list of VolSmile objects, one per expiry.
        day_count: day count for expiry → year fraction conversion.
    """

    def __init__(
        self,
        reference_date: date,
        expiries: list[date],
        smiles: list[VolSmile],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    ):
        if len(expiries) != len(smiles):
            raise ValueError("expiries and smiles must have the same length")
        if len(expiries) < 1:
            raise ValueError("need at least 1 expiry")

        self.reference_date = reference_date
        self.day_count = day_count
        self._expiry_times = [
            year_fraction(reference_date, d, day_count) for d in expiries
        ]
        self._smiles = smiles

    def vol(self, expiry: date, strike: float | None = None) -> float:
        """Vol at (expiry, strike). Linearly interpolates between expiry smiles.

        If strike is None, uses the middle strike of the nearest smile
        (for backward compatibility with flat-vol interfaces).
        """
        t = year_fraction(self.reference_date, expiry, self.day_count)

        if strike is None:
            idx = self._nearest_idx(t)
            mid_k = self._smiles[idx].strikes[len(self._smiles[idx].strikes) // 2]
            strike = mid_k

        if len(self._smiles) == 1:
            return self._smiles[0].vol(strike)

        # Flat extrapolation
        if t <= self._expiry_times[0]:
            return self._smiles[0].vol(strike)
        if t >= self._expiry_times[-1]:
            return self._smiles[-1].vol(strike)

        # Find bracket and linearly interpolate
        for i in range(len(self._expiry_times) - 1):
            if self._expiry_times[i] <= t <= self._expiry_times[i + 1]:
                dt = self._expiry_times[i + 1] - self._expiry_times[i]
                w = (t - self._expiry_times[i]) / dt
                v0 = self._smiles[i].vol(strike)
                v1 = self._smiles[i + 1].vol(strike)
                return v0 * (1 - w) + v1 * w

        return self._smiles[-1].vol(strike)

    def bumped(self, shift: float) -> "VolSurfaceStrike":
        """Return a new surface with all vols shifted by `shift`."""
        bumped_smiles = [s.bumped(shift) for s in self._smiles]
        new = VolSurfaceStrike.__new__(VolSurfaceStrike)
        new.reference_date = self.reference_date
        new.day_count = self.day_count
        new._expiry_times = list(self._expiry_times)
        new._smiles = bumped_smiles
        return new

    def _nearest_idx(self, t: float) -> int:
        best = 0
        best_dist = abs(t - self._expiry_times[0])
        for i in range(1, len(self._expiry_times)):
            dist = abs(t - self._expiry_times[i])
            if dist < best_dist:
                best = i
                best_dist = dist
        return best
