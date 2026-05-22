"""FX forward curve construction from swap points.

Builds FX-implied discount curves from spot + FX swap points + domestic OIS.

    from pricebook.fx.fx_forward_builder import (
        build_fx_implied_curve, FXSwapPointQuote, FXForwardBuildResult,
    )

References:
    Hull (2018). Options, Futures, and Other Derivatives, Ch 5.
    Baba et al. (2008). The Spillover of Money Market Turbulence to FX Swap Markets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention, year_fraction


@dataclass
class FXSwapPointQuote:
    """FX swap point quote."""
    tenor: str                   # "O/N", "1W", "1M", "3M", "6M", "1Y", etc.
    delivery_date: date
    points: float                # swap points (pips)
    pip_factor: float = 10_000   # 10000 for most pairs, 100 for JPY

    @property
    def points_decimal(self) -> float:
        return self.points / self.pip_factor

    def to_dict(self) -> dict:
        return {"tenor": self.tenor, "delivery_date": self.delivery_date.isoformat(),
                "points": self.points, "pip_factor": self.pip_factor}


@dataclass
class FXForwardBuildResult:
    """Result of FX forward curve construction."""
    foreign_curve: DiscountCurve
    forward_rates: list[float]
    pillar_dates: list[date]
    spot: float
    pair: str
    basis_spreads_bp: list[float]

    def to_dict(self) -> dict:
        return {
            "pair": self.pair, "spot": self.spot,
            "n_pillars": len(self.pillar_dates),
            "pillar_dates": [d.isoformat() for d in self.pillar_dates],
            "forward_rates": self.forward_rates,
            "basis_spreads_bp": self.basis_spreads_bp,
        }


# FX pair conventions: settlement lag, pip factor
_FX_CONVENTIONS: dict[str, dict] = {
    "EURUSD": {"settle": 2, "pips": 10_000},
    "USDJPY": {"settle": 2, "pips": 100},
    "GBPUSD": {"settle": 2, "pips": 10_000},
    "USDCHF": {"settle": 2, "pips": 10_000},
    "AUDUSD": {"settle": 2, "pips": 10_000},
    "USDCAD": {"settle": 1, "pips": 10_000},
    "NZDUSD": {"settle": 2, "pips": 10_000},
    "USDMXN": {"settle": 2, "pips": 10_000},
    "USDBRL": {"settle": 2, "pips": 10_000},
    "USDCNY": {"settle": 2, "pips": 10_000},
    "USDINR": {"settle": 2, "pips": 10_000},
    "USDKRW": {"settle": 2, "pips": 10_000},
    "USDTRY": {"settle": 1, "pips": 10_000},
    "USDZAR": {"settle": 2, "pips": 10_000},
}


def build_fx_implied_curve(
    pair: str,
    spot: float,
    reference_date: date,
    swap_points: list[FXSwapPointQuote],
    domestic_curve: DiscountCurve,
    foreign_curve_known: DiscountCurve | None = None,
) -> FXForwardBuildResult:
    """Build an FX-implied foreign discount curve from swap points.

    FX forward: F(T) = Spot + swap_points
    CIP: F(T) = Spot × df_dom(T) / df_for(T)
    => df_for(T) = df_dom(T) × Spot / F(T)

    If foreign_curve_known is provided, extract basis spreads instead.

    Args:
        pair: FX pair code (e.g. "EURUSD", "USDJPY").
        spot: FX spot rate.
        reference_date: valuation date.
        swap_points: list of swap point quotes.
        domestic_curve: discount curve for the domestic (base) currency.
        foreign_curve_known: if provided, compute basis vs this curve.
    """
    if not swap_points:
        raise ValueError("At least one swap point quote required")
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")

    sorted_points = sorted(swap_points, key=lambda q: q.delivery_date)
    dc = DayCountConvention.ACT_365_FIXED

    pillar_dates = []
    foreign_dfs = []
    forward_rates = []
    basis_spreads = []

    for q in sorted_points:
        if q.delivery_date <= reference_date:
            continue

        fwd = spot + q.points_decimal
        if fwd <= 0:
            continue

        df_dom = domestic_curve.df(q.delivery_date)
        df_for = df_dom * spot / fwd
        df_for = max(df_for, 1e-15)

        pillar_dates.append(q.delivery_date)
        foreign_dfs.append(df_for)
        forward_rates.append(fwd)

        # Basis spread vs known foreign curve
        if foreign_curve_known is not None:
            df_known = foreign_curve_known.df(q.delivery_date)
            t = year_fraction(reference_date, q.delivery_date, dc)
            if t > 0 and df_known > 0 and df_for > 0:
                implied_z = -math.log(df_for) / t
                known_z = -math.log(df_known) / t
                basis_spreads.append((implied_z - known_z) * 10_000)
            else:
                basis_spreads.append(0.0)
        else:
            basis_spreads.append(0.0)

    if not pillar_dates:
        raise ValueError("No valid swap points produced discount factors")

    foreign_curve = DiscountCurve(reference_date, pillar_dates, foreign_dfs)

    return FXForwardBuildResult(
        foreign_curve=foreign_curve,
        forward_rates=forward_rates,
        pillar_dates=pillar_dates,
        spot=spot,
        pair=pair.upper(),
        basis_spreads_bp=basis_spreads,
    )


def get_fx_conventions(pair: str) -> dict:
    """Get FX pair conventions."""
    key = pair.upper()
    conv = _FX_CONVENTIONS.get(key)
    if conv is None:
        return {"settle": 2, "pips": 10_000}
    return conv


def list_fx_pairs() -> list[str]:
    """Return supported FX pairs."""
    return sorted(_FX_CONVENTIONS.keys())
