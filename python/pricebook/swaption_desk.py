"""Swaption desk tools: vol surface management, combo strategies, and hedging.

Manages ATM vol grids with SABR smile calibration, builds straddle/strangle/
risk-reversal strategies, and computes delta/vega hedges.

    from pricebook.swaption_desk import (
        VolCube, straddle, strangle, risk_reversal,
        delta_hedge, vega_hedge,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pricebook.black76 import OptionType, black76_price
from pricebook.day_count import DayCountConvention, year_fraction, date_from_year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
from pricebook.sabr import sabr_calibrate, sabr_implied_vol, shifted_sabr_implied_vol
from pricebook.swaption import Swaption, SwaptionType
from pricebook.swaption_vol import SwaptionVolSurface
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.vol_surface import FlatVol


# ---- Vol Cube: ATM grid + SABR smile per cell ----

@dataclass
class SABRCell:
    """SABR parameters for one (expiry, tenor) cell."""
    alpha: float
    beta: float
    rho: float
    nu: float
    shift: float = 0.0


class VolCube:
    """ATM vol grid with optional SABR smile at each cell.

    The ATM layer is a SwaptionVolSurface. On top, each cell can carry
    SABR params for strike-dependent vol.

    Args:
        atm_surface: ATM vol grid (expiry × tenor).
        sabr_cells: dict mapping (expiry_idx, tenor_idx) to SABRCell.
    """

    def __init__(
        self,
        atm_surface: SwaptionVolSurface,
        sabr_cells: dict[tuple[int, int], SABRCell] | None = None,
    ):
        self.atm = atm_surface
        self._sabr: dict[tuple[int, int], SABRCell] = sabr_cells or {}

    def set_sabr(self, expiry_idx: int, tenor_idx: int, cell: SABRCell) -> None:
        self._sabr[(expiry_idx, tenor_idx)] = cell

    def get_sabr(self, expiry_idx: int, tenor_idx: int) -> SABRCell | None:
        return self._sabr.get((expiry_idx, tenor_idx))

    def vol(self, expiry: date, strike: float | None = None) -> float:
        """ATM vol (delegates to underlying surface)."""
        return self.atm.vol(expiry, strike)

    def vol_sabr(
        self,
        expiry_idx: int,
        tenor_idx: int,
        forward: float,
        strike: float,
        T: float,
    ) -> float:
        """Vol from SABR smile at a specific cell, or ATM fallback."""
        cell = self._sabr.get((expiry_idx, tenor_idx))
        if cell is None:
            return self.atm._vols[expiry_idx, tenor_idx]
        if cell.shift != 0.0:
            return shifted_sabr_implied_vol(
                forward, strike, T, cell.alpha, cell.beta,
                cell.rho, cell.nu, cell.shift,
            )
        return sabr_implied_vol(
            forward, strike, T, cell.alpha, cell.beta, cell.rho, cell.nu,
        )

    def bump_parallel(self, shift: float) -> VolCube:
        """Return a new cube with all ATM vols shifted."""
        import numpy as np
        new_vols = (self.atm._vols + shift).tolist()
        new_atm = SwaptionVolSurface(
            self.atm.reference_date,
            [date_from_year_fraction(self.atm.reference_date, t)
             for t in self.atm._expiry_times],
            self.atm._tenors.tolist(),
            new_vols,
        )
        return VolCube(new_atm, dict(self._sabr))

    def bump_term(self, expiry_idx: int, shift: float) -> VolCube:
        """Bump all tenors at one expiry."""
        import numpy as np
        new_vols = self.atm._vols.copy()
        new_vols[expiry_idx, :] += shift
        new_atm = SwaptionVolSurface(
            self.atm.reference_date,
            [date_from_year_fraction(self.atm.reference_date, t)
             for t in self.atm._expiry_times],
            self.atm._tenors.tolist(),
            new_vols.tolist(),
        )
        return VolCube(new_atm, dict(self._sabr))


# ---- Swaption combo strategies ----

@dataclass
class SwaptionCombo:
    """A combination of swaptions with aggregate risk."""
    name: str
    legs: list[tuple[Swaption, int]]  # (swaption, direction: +1/-1)
    pv: float = 0.0
    delta: float = 0.0
    vega: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0


def _swaption_greeks(
    swn: Swaption,
    curve: DiscountCurve,
    vol_surface,
    rate_shift: float = 0.0001,
    vol_shift: float = 0.01,
    time_shift: float = 1.0 / 365,
) -> dict[str, float]:
    """Compute Greeks for a swaption via bump-and-reprice."""
    base_pv = swn.pv(curve, vol_surface)

    # Delta (rate sensitivity)
    bumped_curve = curve.bumped(rate_shift)
    delta = (swn.pv(bumped_curve, vol_surface) - base_pv) / rate_shift

    # Gamma
    down_curve = curve.bumped(-rate_shift)
    gamma = (swn.pv(bumped_curve, vol_surface) - 2 * base_pv
             + swn.pv(down_curve, vol_surface)) / (rate_shift ** 2)

    # Vega (vol sensitivity)
    if isinstance(vol_surface, FlatVol):
        bumped_vol = FlatVol(vol_surface._vol + vol_shift)
    elif hasattr(vol_surface, '_vols'):
        # SwaptionVolSurface or VolCube: parallel bump all vols
        import numpy as np
        from pricebook.swaption_vol import SwaptionVolSurface
        raw = vol_surface.atm if hasattr(vol_surface, 'atm') else vol_surface
        new_vols = (raw._vols + vol_shift).tolist()
        expiry_dates = [
            date_from_year_fraction(raw.reference_date, t)
            for t in raw._expiry_times
        ]
        bumped_vol = SwaptionVolSurface(
            raw.reference_date, expiry_dates, raw._tenors.tolist(), new_vols,
        )
    else:
        bumped_vol = vol_surface  # fallback: vega = 0
    vega = (swn.pv(curve, bumped_vol) - base_pv) / vol_shift

    # Theta (1-day time decay)
    ref = curve.reference_date
    new_ref = date.fromordinal(ref.toordinal() + 1)
    theta = swn.pv(curve, vol_surface, valuation_date=new_ref) - base_pv

    return {"pv": base_pv, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def straddle(
    expiry: date,
    swap_end: date,
    strike: float,
    curve: DiscountCurve,
    vol_surface,
    notional: float = 1_000_000.0,
) -> SwaptionCombo:
    """ATM straddle: payer + receiver at the same strike."""
    payer = Swaption(expiry, swap_end, strike, SwaptionType.PAYER, notional)
    receiver = Swaption(expiry, swap_end, strike, SwaptionType.RECEIVER, notional)

    g_p = _swaption_greeks(payer, curve, vol_surface)
    g_r = _swaption_greeks(receiver, curve, vol_surface)

    return SwaptionCombo(
        name=f"straddle_{strike:.4f}",
        legs=[(payer, 1), (receiver, 1)],
        pv=g_p["pv"] + g_r["pv"],
        delta=g_p["delta"] + g_r["delta"],
        vega=g_p["vega"] + g_r["vega"],
        gamma=g_p["gamma"] + g_r["gamma"],
        theta=g_p["theta"] + g_r["theta"],
    )


def strangle(
    expiry: date,
    swap_end: date,
    strike_low: float,
    strike_high: float,
    curve: DiscountCurve,
    vol_surface,
    notional: float = 1_000_000.0,
) -> SwaptionCombo:
    """Strangle: receiver at low strike + payer at high strike."""
    receiver = Swaption(expiry, swap_end, strike_low, SwaptionType.RECEIVER, notional)
    payer = Swaption(expiry, swap_end, strike_high, SwaptionType.PAYER, notional)

    g_r = _swaption_greeks(receiver, curve, vol_surface)
    g_p = _swaption_greeks(payer, curve, vol_surface)

    return SwaptionCombo(
        name=f"strangle_{strike_low:.4f}_{strike_high:.4f}",
        legs=[(receiver, 1), (payer, 1)],
        pv=g_r["pv"] + g_p["pv"],
        delta=g_r["delta"] + g_p["delta"],
        vega=g_r["vega"] + g_p["vega"],
        gamma=g_r["gamma"] + g_p["gamma"],
        theta=g_r["theta"] + g_p["theta"],
    )


def risk_reversal(
    expiry: date,
    swap_end: date,
    strike_low: float,
    strike_high: float,
    curve: DiscountCurve,
    vol_surface,
    notional: float = 1_000_000.0,
) -> SwaptionCombo:
    """Risk reversal: long payer (high strike) + short receiver (low strike)."""
    payer = Swaption(expiry, swap_end, strike_high, SwaptionType.PAYER, notional)
    receiver = Swaption(expiry, swap_end, strike_low, SwaptionType.RECEIVER, notional)

    g_p = _swaption_greeks(payer, curve, vol_surface)
    g_r = _swaption_greeks(receiver, curve, vol_surface)

    return SwaptionCombo(
        name=f"rr_{strike_low:.4f}_{strike_high:.4f}",
        legs=[(payer, 1), (receiver, -1)],
        pv=g_p["pv"] - g_r["pv"],
        delta=g_p["delta"] - g_r["delta"],
        vega=g_p["vega"] - g_r["vega"],
        gamma=g_p["gamma"] - g_r["gamma"],
        theta=g_p["theta"] - g_r["theta"],
    )


# ---- Hedging ----

def delta_hedge(
    swaptions: list[tuple[Swaption, int]],
    curve: DiscountCurve,
    vol_surface,
    hedge_tenor_years: int = 10,
) -> dict[str, Any]:
    """Delta-hedge a swaption book with a swap.

    Returns the hedge swap notional and residual delta.
    """
    ref = curve.reference_date
    total_delta = 0.0
    for swn, direction in swaptions:
        g = _swaption_greeks(swn, curve, vol_surface)
        total_delta += direction * g["delta"]

    # Build a unit-notional swap to compute its delta
    from dateutil.relativedelta import relativedelta
    hedge_end = ref + relativedelta(years=hedge_tenor_years)
    unit_swap = InterestRateSwap(
        ref, hedge_end, fixed_rate=0.05,
        direction=SwapDirection.PAYER, notional=1_000_000.0,
    )
    par = unit_swap.par_rate(curve)
    hedge_swap = InterestRateSwap(
        ref, hedge_end, fixed_rate=par,
        direction=SwapDirection.PAYER, notional=1_000_000.0,
    )

    # Swap delta per unit notional
    shift = 0.0001
    bumped = curve.bumped(shift)
    swap_delta_unit = (hedge_swap.pv(bumped) - hedge_swap.pv(curve)) / shift

    if abs(swap_delta_unit) < 1e-12:
        return {"hedge_notional": 0.0, "residual_delta": total_delta}

    # Notional to offset portfolio delta
    hedge_notional = -total_delta / swap_delta_unit * 1_000_000.0

    hedge_swap_scaled = InterestRateSwap(
        ref, hedge_end, fixed_rate=par,
        direction=SwapDirection.PAYER, notional=abs(hedge_notional),
    )
    hedge_direction = 1 if hedge_notional > 0 else -1
    hedge_delta = hedge_direction * (
        hedge_swap_scaled.pv(bumped) - hedge_swap_scaled.pv(curve)
    ) / shift

    return {
        "portfolio_delta": total_delta,
        "hedge_notional": hedge_notional,
        "hedge_tenor": hedge_tenor_years,
        "hedge_delta": hedge_delta,
        "residual_delta": total_delta + hedge_delta,
    }


def vega_hedge(
    swaptions: list[tuple[Swaption, int]],
    hedge_swaption: Swaption,
    curve: DiscountCurve,
    vol_surface,
) -> dict[str, Any]:
    """Vega-hedge a swaption book with another swaption.

    Returns the hedge notional and residual vega.
    """
    total_vega = 0.0
    total_delta = 0.0
    for swn, direction in swaptions:
        g = _swaption_greeks(swn, curve, vol_surface)
        total_vega += direction * g["vega"]
        total_delta += direction * g["delta"]

    g_h = _swaption_greeks(hedge_swaption, curve, vol_surface)
    hedge_vega_unit = g_h["vega"]

    if abs(hedge_vega_unit) < 1e-12:
        return {"hedge_ratio": 0.0, "residual_vega": total_vega}

    hedge_ratio = -total_vega / hedge_vega_unit
    residual_vega = total_vega + hedge_ratio * hedge_vega_unit
    residual_delta = total_delta + hedge_ratio * g_h["delta"]

    return {
        "portfolio_vega": total_vega,
        "portfolio_delta": total_delta,
        "hedge_ratio": hedge_ratio,
        "hedge_vega": hedge_ratio * hedge_vega_unit,
        "residual_vega": residual_vega,
        "residual_delta": residual_delta,
    }
