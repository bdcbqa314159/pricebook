"""Dual real+nominal curve builder for inflation-linked markets.

Joint calibration: nominal from swaps/bonds, real from linker prices.
BEI = nominal_zero - real_zero at each tenor.

    from pricebook.curves.inflation_curve import (
        build_real_nominal_curves, RealNominalResult,
    )

References:
    Deacon, Derry & Mirfendereski (2004). Inflation-Indexed Securities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.interpolation import InterpolationMethod
from pricebook.core.day_count import DayCountConvention, year_fraction


@dataclass
class RealNominalResult:
    """Result of dual real+nominal curve construction."""
    nominal_curve: DiscountCurve
    real_curve: DiscountCurve
    bei_term_structure: list[dict]  # [{years, nominal_rate, real_rate, bei}]
    currency: str

    def to_dict(self) -> dict:
        return {"currency": self.currency, "n_bei_points": len(self.bei_term_structure)}


def build_real_nominal_curves(
    reference_date: date,
    currency: str,
    nominal_zeros: list[tuple[float, float]],
    real_zeros: list[tuple[float, float]],
    bei_tenors: list[float] | None = None,
) -> RealNominalResult:
    """Build paired nominal + real discount curves and BEI term structure.

    Args:
        nominal_zeros: [(tenor_years, zero_rate), ...] for nominal curve.
        real_zeros: [(tenor_years, zero_rate), ...] for real curve.
        bei_tenors: maturities for BEI computation (default: all available).

    Returns:
        RealNominalResult with both curves and BEI.
    """
    from dateutil.relativedelta import relativedelta
    dc = DayCountConvention.ACT_365_FIXED

    # Build nominal curve
    nom_dates = [reference_date + relativedelta(years=int(t), months=int((t % 1) * 12))
                 for t, _ in nominal_zeros]
    nom_dfs = [math.exp(-r * t) for t, r in nominal_zeros]
    nominal = DiscountCurve(reference_date, nom_dates, nom_dfs,
                             interpolation=InterpolationMethod.LOG_LINEAR)

    # Build real curve
    real_dates = [reference_date + relativedelta(years=int(t), months=int((t % 1) * 12))
                  for t, _ in real_zeros]
    real_dfs = [math.exp(-r * t) for t, r in real_zeros]
    real = DiscountCurve(reference_date, real_dates, real_dfs,
                          interpolation=InterpolationMethod.LOG_LINEAR)

    # BEI term structure
    if bei_tenors is None:
        all_tenors = sorted(set(t for t, _ in nominal_zeros) | set(t for t, _ in real_zeros))
        bei_tenors = [t for t in all_tenors if t >= 1]

    bei = []
    for T in bei_tenors:
        mat = reference_date + relativedelta(years=int(T))
        df_nom = nominal.df(mat)
        df_real = real.df(mat)
        if df_nom > 0 and df_real > 0 and T > 0:
            nom_rate = -math.log(df_nom) / T
            real_rate = -math.log(df_real) / T
            bei_val = nom_rate - real_rate
        else:
            nom_rate = real_rate = bei_val = 0.0
        bei.append({"years": T, "nominal_rate": nom_rate,
                     "real_rate": real_rate, "bei": bei_val})

    return RealNominalResult(nominal, real, bei, currency)
