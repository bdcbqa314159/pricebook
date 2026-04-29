"""Volatility surface: flat vol, vol term structure, and arbitrage checks."""

from __future__ import annotations

from dataclasses import dataclass
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

    def bumped(self, shift: float) -> "FlatVol":
        """Return a new FlatVol with vol shifted by `shift`."""
        return FlatVol(max(self._vol + shift, 0.0))


class VolTermStructure:
    """
    Volatility as a function of expiry (flat across strikes).

    Interpolates between pillar vols. Flat extrapolation at boundaries.
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
        self._expiries = list(expiries)
        self._vols = list(vols)
        self._interpolation = interpolation

        times = [year_fraction(reference_date, d, day_count) for d in expiries]

        if len(times) == 1:
            self._single_vol = vols[0]
            self._interpolator = None
        else:
            self._single_vol = None
            self._interpolator: Interpolator = create_interpolator(
                interpolation, np.array(times), np.array(vols),
            )

    def vol(self, expiry: date, strike: float | None = None) -> float:
        """Volatility at the given expiry. Strike ignored (flat smile)."""
        if self._single_vol is not None:
            return self._single_vol
        t = year_fraction(self.reference_date, expiry, self.day_count)
        if t <= 0:
            return float(self._interpolator(0.0))
        return float(self._interpolator(t))

    def bumped(self, shift: float) -> "VolTermStructure":
        """Return a new term structure with all vols shifted by `shift`."""
        new_vols = [max(v + shift, 0.0) for v in self._vols]
        return VolTermStructure(
            self.reference_date, self._expiries, new_vols,
            self.day_count, self._interpolation,
        )


# ---- Arbitrage checks ----

@dataclass
class VolSurfaceArbitrageResult:
    """Result of vol surface arbitrage checks."""
    is_arbitrage_free: bool
    calendar_violations: list[str]
    butterfly_violations: list[str]
    total_variance_monotone: bool


def check_calendar_arbitrage(
    expiry_times: list[float],
    atm_vols: list[float],
) -> list[str]:
    """Check that total variance σ²T is non-decreasing in T.

    Violation means you can construct a riskless profit from calendar spreads.
    """
    violations = []
    for i in range(1, len(expiry_times)):
        tv_prev = atm_vols[i - 1] ** 2 * expiry_times[i - 1]
        tv_curr = atm_vols[i] ** 2 * expiry_times[i]
        if tv_curr < tv_prev - 1e-10:
            violations.append(
                f"T={expiry_times[i]:.3f}: total_var={tv_curr:.6f} < "
                f"T={expiry_times[i-1]:.3f}: total_var={tv_prev:.6f}"
            )
    return violations


def check_butterfly_arbitrage(
    strikes: list[float],
    call_prices: list[float],
) -> list[str]:
    """Check that d²C/dK² ≥ 0 (no negative butterflies).

    Equivalent to: call prices are convex in strike.
    Violation means a butterfly spread has negative value.
    """
    violations = []
    for i in range(1, len(strikes) - 1):
        dk1 = strikes[i] - strikes[i - 1]
        dk2 = strikes[i + 1] - strikes[i]
        # Second finite difference
        d2c = (call_prices[i + 1] - call_prices[i]) / dk2 - \
              (call_prices[i] - call_prices[i - 1]) / dk1
        d2c /= 0.5 * (dk1 + dk2)
        if d2c < -1e-10:
            violations.append(
                f"K={strikes[i]:.2f}: d²C/dK²={d2c:.6f} < 0 (negative butterfly)"
            )
    return violations


def validate_vol_surface(
    expiry_times: list[float],
    atm_vols: list[float],
    strikes: list[float] | None = None,
    call_prices: list[float] | None = None,
) -> VolSurfaceArbitrageResult:
    """Run all arbitrage checks on a vol surface.

    Args:
        expiry_times: year fractions for each expiry.
        atm_vols: ATM vol at each expiry.
        strikes: strikes for butterfly check (optional).
        call_prices: call prices at those strikes (optional).
    """
    cal = check_calendar_arbitrage(expiry_times, atm_vols)
    tv_mono = len(cal) == 0

    bfly = []
    if strikes is not None and call_prices is not None:
        bfly = check_butterfly_arbitrage(strikes, call_prices)

    return VolSurfaceArbitrageResult(
        is_arbitrage_free=tv_mono and len(bfly) == 0,
        calendar_violations=cal,
        butterfly_violations=bfly,
        total_variance_monotone=tv_mono,
    )

from pricebook.serialisable import _register

FlatVol._SERIAL_TYPE = "flat_vol"

def _fv_to_dict(self):
    return {"type": "flat_vol", "params": {"vol": self._vol}}

@classmethod
def _fv_from_dict(cls, d):
    return cls(d["params"]["vol"])

FlatVol.to_dict = _fv_to_dict
FlatVol.from_dict = _fv_from_dict
_register(FlatVol)
