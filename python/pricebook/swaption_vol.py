"""
Swaption volatility surface: vol(expiry, tenor).

A 2D grid of ATM normal or lognormal vols, indexed by option expiry
and underlying swap tenor. Bilinear interpolation between grid points,
flat extrapolation at boundaries.

Strike dimension is anticipated in the interface (vol(expiry, strike))
but ignored for now — this is an ATM-only surface.

    surface = SwaptionVolSurface(
        reference_date=date(2024, 1, 15),
        expiries=[date(2025, 1, 15), date(2026, 1, 15), ...],
        tenors=[1, 2, 5, 10],        # underlying swap tenor in years
        vols=[[0.20, 0.19, ...],      # expiry × tenor grid
              [0.21, 0.20, ...]],
    )
    v = surface.vol(expiry=date(2025, 7, 15), strike=0.03)
"""

from __future__ import annotations

from datetime import date

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction


class SwaptionVolSurface:
    """
    2D ATM swaption vol surface: vol(expiry, tenor).

    Args:
        reference_date: valuation date.
        expiries: option expiry dates (rows of the grid).
        tenors: underlying swap tenors in years (columns of the grid).
        vols: 2D list, shape (len(expiries), len(tenors)). Each entry is
            the ATM vol for that (expiry, tenor) pair.
        day_count: day count for converting expiry dates to year fractions.
    """

    def __init__(
        self,
        reference_date: date,
        expiries: list[date],
        tenors: list[float],
        vols: list[list[float]],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    ):
        if len(expiries) < 1 or len(tenors) < 1:
            raise ValueError("need at least 1 expiry and 1 tenor")
        if len(vols) != len(expiries):
            raise ValueError(
                f"vols has {len(vols)} rows but expected {len(expiries)} (one per expiry)"
            )
        for i, row in enumerate(vols):
            if len(row) != len(tenors):
                raise ValueError(
                    f"vols row {i} has {len(row)} columns but expected {len(tenors)}"
                )
            for v in row:
                if v < 0:
                    raise ValueError(f"vols must be non-negative, got {v}")

        self.reference_date = reference_date
        self.day_count = day_count
        self._expiry_times = np.array(
            [year_fraction(reference_date, d, day_count) for d in expiries]
        )
        self._tenors = np.array(tenors, dtype=float)
        self._vols = np.array(vols, dtype=float)  # shape (n_expiries, n_tenors)

    def vol(self, expiry: date, strike: float | None = None) -> float:
        """
        ATM vol at the given expiry.

        The tenor is inferred from the swaption's context (expiry → swap_end),
        but for now we interpolate in the expiry dimension only, using the
        first tenor column. To query a specific tenor, use vol_expiry_tenor().

        Strike is accepted for interface compatibility but ignored (ATM surface).
        """
        t = year_fraction(self.reference_date, expiry, self.day_count)
        t = max(t, self._expiry_times[0])
        # Use the middle tenor as default for the vol(expiry, strike) interface
        mid = len(self._tenors) // 2
        return float(self._interp_expiry(t, mid))

    def vol_expiry_tenor(self, expiry: date, tenor: float) -> float:
        """
        Bilinear interpolation on the (expiry, tenor) grid.

        Args:
            expiry: option expiry date.
            tenor: underlying swap tenor in years.
        """
        t_exp = year_fraction(self.reference_date, expiry, self.day_count)
        t_exp = max(t_exp, self._expiry_times[0])
        return float(self._interp_2d(t_exp, tenor))

    def _interp_expiry(self, t_exp: float, tenor_idx: int) -> float:
        """1D interpolation along expiry axis at a fixed tenor index."""
        times = self._expiry_times
        col = self._vols[:, tenor_idx]

        if len(times) == 1:
            return col[0]

        # Flat extrapolation
        if t_exp <= times[0]:
            return col[0]
        if t_exp >= times[-1]:
            return col[-1]

        # Linear interpolation
        idx = np.searchsorted(times, t_exp) - 1
        idx = max(0, min(idx, len(times) - 2))
        frac = (t_exp - times[idx]) / (times[idx + 1] - times[idx])
        return col[idx] + frac * (col[idx + 1] - col[idx])

    def _interp_2d(self, t_exp: float, tenor: float) -> float:
        """Bilinear interpolation on (expiry_time, tenor) grid."""
        times = self._expiry_times
        tenors = self._tenors

        t_exp = np.clip(t_exp, times[0], times[-1])
        tenor = np.clip(tenor, tenors[0], tenors[-1])

        if len(times) == 1 and len(tenors) == 1:
            return self._vols[0, 0]
        if len(times) == 1:
            return self._interp_tenor(tenor, 0)
        if len(tenors) == 1:
            return self._interp_expiry(t_exp, 0)

        i = int(np.searchsorted(times, t_exp)) - 1
        i = max(0, min(i, len(times) - 2))
        fx = (t_exp - times[i]) / (times[i + 1] - times[i])
        j = int(np.searchsorted(tenors, tenor)) - 1
        j = max(0, min(j, len(tenors) - 2))
        fy = (tenor - tenors[j]) / (tenors[j + 1] - tenors[j])

        v00 = self._vols[i, j]
        v01 = self._vols[i, j + 1]
        v10 = self._vols[i + 1, j]
        v11 = self._vols[i + 1, j + 1]

        return (
            v00 * (1 - fx) * (1 - fy)
            + v01 * (1 - fx) * fy
            + v10 * fx * (1 - fy)
            + v11 * fx * fy
        )

    def _interp_tenor(self, tenor: float, expiry_idx: int) -> float:
        """1D interpolation along tenor axis at a fixed expiry index."""
        tenors = self._tenors
        row = self._vols[expiry_idx, :]

        if len(tenors) == 1:
            return row[0]

        if tenor <= tenors[0]:
            return row[0]
        if tenor >= tenors[-1]:
            return row[-1]

        idx = int(np.searchsorted(tenors, tenor)) - 1
        idx = max(0, min(idx, len(tenors) - 2))
        frac = (tenor - tenors[idx]) / (tenors[idx + 1] - tenors[idx])
        return row[idx] + frac * (row[idx + 1] - row[idx])
