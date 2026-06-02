"""Credit spread vol surface: term structure of CDS spread vol.

ATM vol grid indexed by (expiry × tenor), with bilinear interpolation.

* :class:`CreditSpreadVolSurface` — 2D ATM credit spread vol surface.
* :func:`build_credit_vol_surface` — bootstrap from CDS swaption quotes.
* :func:`synthetic_credit_vol_surface` — generate realistic vol surface.

References:
    Pedersen, *Valuation of Portfolio Credit Default Swaptions*,
    Lehman Brothers QR, 2003.
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*,
    Ch. 15 (CDS options).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve


class CreditSpreadVolSurface:
    """2D ATM credit spread vol surface: expiry × tenor.

    Structure:
    - **ATM grid**: bilinear interpolation on (expiry_years, tenor_years).
    - Strike dimension not modelled (ATM-only); pass strike=None.

    Usage:
        vol = surface.vol(2.0, 5.0)
        bumped = surface.bumped(0.01)
    """

    def __init__(
        self,
        reference_date: date,
        expiry_years: list[float],
        tenor_years: list[float],
        vols: list[list[float]],
    ):
        self.reference_date = reference_date
        self._expiries = np.array(expiry_years)
        self._tenors = np.array(tenor_years)
        self._atm_grid = np.array(vols)  # (n_expiries, n_tenors)

        if self._atm_grid.shape != (len(expiry_years), len(tenor_years)):
            raise ValueError(
                f"vols shape {self._atm_grid.shape} does not match "
                f"({len(expiry_years)}, {len(tenor_years)})"
            )

    def vol(
        self,
        expiry_years: float,
        tenor_years: float,
        strike: float | None = None,
    ) -> float:
        """Interpolated ATM vol at (expiry_years, tenor_years).

        Bilinear interpolation on the ATM grid, same pattern as
        ``SwaptionVolCube._interp_atm``.  The *strike* argument is
        accepted for interface compatibility but ignored (ATM-only surface).
        """
        return self._interp_atm(expiry_years, tenor_years)

    def _interp_atm(self, t_exp: float, tenor: float) -> float:
        """Bilinear interpolation on ATM grid."""
        i = int(np.searchsorted(self._expiries, t_exp)) - 1
        j = int(np.searchsorted(self._tenors, tenor)) - 1
        i = max(0, min(i, len(self._expiries) - 2))
        j = max(0, min(j, len(self._tenors) - 2))

        e0, e1 = self._expiries[i], self._expiries[i + 1]
        t0, t1 = self._tenors[j], self._tenors[j + 1]

        we = (t_exp - e0) / (e1 - e0) if e1 > e0 else 0.0
        wt = (tenor - t0) / (t1 - t0) if t1 > t0 else 0.0
        we = max(0.0, min(1.0, we))
        wt = max(0.0, min(1.0, wt))

        v00 = self._atm_grid[i, j]
        v01 = self._atm_grid[i, j + 1]
        v10 = self._atm_grid[i + 1, j]
        v11 = self._atm_grid[i + 1, j + 1]

        return float(
            (1 - we) * (1 - wt) * v00
            + (1 - we) * wt * v01
            + we * (1 - wt) * v10
            + we * wt * v11
        )

    def bumped(self, shift: float) -> "CreditSpreadVolSurface":
        """Parallel shift all vols by *shift* (additive)."""
        new_atm = (self._atm_grid + shift).tolist()
        return CreditSpreadVolSurface(
            self.reference_date,
            self._expiries.tolist(),
            self._tenors.tolist(),
            new_atm,
        )

    def to_dict(self) -> dict:
        return {
            "reference_date": self.reference_date.isoformat(),
            "expiries": self._expiries.tolist(),
            "tenors": self._tenors.tolist(),
            "atm_grid": self._atm_grid.tolist(),
        }


# ---- Bootstrap from CDS swaption quotes ----

def _invert_black_vol(
    premium: float,
    forward_spread: float,
    strike: float,
    expiry: float,
    risky_annuity: float,
    survival_to_expiry: float,
    notional: float,
) -> float:
    """Invert CDS swaption premium to Black implied vol via bisection."""
    from pricebook.credit.cds_swaption import cds_swaption_black

    lo, hi = 0.001, 5.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        result = cds_swaption_black(
            forward_spread, strike, mid, expiry,
            risky_annuity, survival_to_expiry, notional, "payer",
        )
        if result.premium < premium:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-8:
            break
    return 0.5 * (lo + hi)


def build_credit_vol_surface(
    swaption_quotes: list[dict[str, Any]],
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> CreditSpreadVolSurface:
    """Bootstrap a credit spread vol surface from CDS swaption quotes.

    Each quote is a dict with keys:
        expiry_years, tenor_years, premium, strike, notional.

    The function inverts each premium to a Black implied vol using
    :func:`cds_swaption_black` from ``credit/cds_swaption.py``.
    """
    from pricebook.credit.cds_swaption import forward_cds_spread

    ref = discount_curve.reference_date

    # Collect unique expiries and tenors
    expiry_set: set[float] = set()
    tenor_set: set[float] = set()
    for q in swaption_quotes:
        expiry_set.add(q["expiry_years"])
        tenor_set.add(q["tenor_years"])

    expiry_years = sorted(expiry_set)
    tenor_years = sorted(tenor_set)

    # Build vol grid
    vol_map: dict[tuple[float, float], float] = {}
    for q in swaption_quotes:
        e = q["expiry_years"]
        t = q["tenor_years"]
        notional = q.get("notional", 1_000_000)

        # Forward CDS spread from flat hazard approximation
        flat_hazard = -math.log(max(survival_curve.survival(e + t), 1e-15)) / (e + t)
        flat_rate = -math.log(max(discount_curve.df(e), 1e-15)) / max(e, 0.01)
        fwd = forward_cds_spread(e, e + t, flat_hazard, flat_rate)

        implied_vol = _invert_black_vol(
            q["premium"],
            fwd.forward_spread,
            q["strike"],
            e,
            fwd.risky_annuity,
            fwd.survival_to_start,
            notional,
        )
        vol_map[(e, t)] = implied_vol

    # Assemble grid — fill missing nodes with nearest neighbour
    vols: list[list[float]] = []
    for e in expiry_years:
        row: list[float] = []
        for t in tenor_years:
            if (e, t) in vol_map:
                row.append(vol_map[(e, t)])
            else:
                # Nearest available vol by expiry/tenor distance
                if vol_map:
                    nearest = min(
                        vol_map.items(),
                        key=lambda kv: (kv[0][0] - e) ** 2 + (kv[0][1] - t) ** 2,
                    )[1]
                else:
                    nearest = 0.40
                row.append(nearest)
        vols.append(row)

    return CreditSpreadVolSurface(ref, expiry_years, tenor_years, vols)


# ---- Synthetic surface for testing ----

def synthetic_credit_vol_surface(
    hazard_rate: float,
    reference_date: date,
) -> CreditSpreadVolSurface:
    """Generate a realistic credit spread vol surface.

    Args:
        hazard_rate: annualised hazard rate (proxy for credit quality).
            IG ~ 0.01–0.02, HY ~ 0.05+.
        reference_date: valuation date.

    Returns:
        ``CreditSpreadVolSurface`` with standard expiry/tenor grid.

    Base vol is ~40% for IG (hazard_rate < 0.03), ~60% for HY.
    Vol decreases with expiry (mean-reversion effect).
    """
    expiry_years = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    tenor_years = [3.0, 5.0, 7.0, 10.0]

    # Base vol: IG vs HY
    base_vol = 0.40 if hazard_rate < 0.03 else 0.60

    # Mean-reversion decay: vol falls with expiry
    expiry_decay = [1.15, 1.08, 1.00, 0.90, 0.82, 0.70]

    # Tenor effect: longer tenors slightly lower vol (diversification)
    tenor_factor = [1.05, 1.00, 0.97, 0.94]

    vols: list[list[float]] = []
    for k, _e in enumerate(expiry_years):
        row: list[float] = []
        for m, _t in enumerate(tenor_years):
            v = base_vol * expiry_decay[k] * tenor_factor[m]
            row.append(round(v, 4))
        vols.append(row)

    return CreditSpreadVolSurface(reference_date, expiry_years, tenor_years, vols)
