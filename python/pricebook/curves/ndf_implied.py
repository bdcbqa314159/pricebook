"""NDF-implied discount curve construction for restricted EM currencies.

For currencies with capital controls (CNY, INR, KRW, BRL, TWD, etc.),
offshore interest rates are implied from FX NDF prices and a known G10
base curve. This is often the only source of an EM discount curve for
offshore participants.

    from pricebook.curves.ndf_implied import build_ndf_implied_curve

    em_curve = build_ndf_implied_curve(
        ref_date=date(2024,1,15),
        spot_rate=7.18,           # USD/CNY spot
        ndf_quotes=[              # NDF outright prices
            NDFQuote(tenor_months=1, outright=7.20),
            NDFQuote(tenor_months=3, outright=7.25),
            NDFQuote(tenor_months=6, outright=7.32),
            NDFQuote(tenor_months=12, outright=7.45),
        ],
        base_curve=usd_ois_curve, # known G10 discount curve
    )

The fundamental relationship:

    NDF(T) = Spot × df_base(T) / df_em(T)

    =>  df_em(T) = df_base(T) × Spot / NDF(T)

This is covered interest parity applied to non-deliverable forwards.

References:
    Della Corte, Sarno & Tsiakas (2009). An Economic Evaluation of
    Empirical Exchange Rate Models, Review of Financial Studies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention, year_fraction


@dataclass
class NDFQuote:
    """A single NDF outright quote."""
    tenor_months: int       # 1, 3, 6, 12, 24, etc.
    outright: float         # NDF outright price (e.g. 7.25 for USD/CNY)
    bid: float | None = None
    ask: float | None = None

    @property
    def mid(self) -> float:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.outright

    def to_dict(self) -> dict:
        return {
            "tenor_months": self.tenor_months,
            "outright": self.outright,
            "bid": self.bid,
            "ask": self.ask,
        }


@dataclass
class NDFImpliedResult:
    """Result of NDF-implied curve construction."""
    em_curve: DiscountCurve
    pillar_dates: list[date]
    implied_dfs: list[float]
    implied_zero_rates: list[float]     # continuously compounded
    ndf_forward_points: list[float]     # NDF - Spot for each tenor
    base_currency: str
    em_currency: str
    spot_rate: float
    n_pillars: int

    def to_dict(self) -> dict:
        return {
            "pillar_dates": [d.isoformat() for d in self.pillar_dates],
            "implied_dfs": self.implied_dfs,
            "implied_zero_rates": self.implied_zero_rates,
            "ndf_forward_points": self.ndf_forward_points,
            "base_currency": self.base_currency,
            "em_currency": self.em_currency,
            "spot_rate": self.spot_rate,
            "n_pillars": self.n_pillars,
        }


def build_ndf_implied_curve(
    reference_date: date,
    spot_rate: float,
    ndf_quotes: list[NDFQuote],
    base_curve: DiscountCurve,
    base_currency: str = "USD",
    em_currency: str = "EM",
) -> NDFImpliedResult:
    """Build an EM discount curve from NDF prices and a G10 base curve.

    The covered interest parity relationship gives:

        df_em(T) = df_base(T) × Spot / NDF(T)

    Args:
        reference_date: valuation date.
        spot_rate: FX spot rate (base/em, e.g. 7.18 for USD/CNY).
        ndf_quotes: list of NDF quotes at various tenors.
        base_curve: risk-free discount curve for the base currency (e.g. USD OIS).
        base_currency: ISO code for base (default "USD").
        em_currency: ISO code for EM currency.

    Returns:
        NDFImpliedResult with the implied EM discount curve.
    """
    if not ndf_quotes:
        raise ValueError("At least one NDF quote required")
    if spot_rate <= 0:
        raise ValueError(f"spot_rate must be positive, got {spot_rate}")

    sorted_quotes = sorted(ndf_quotes, key=lambda q: q.tenor_months)

    pillar_dates = []
    implied_dfs = []
    implied_zeros = []
    fwd_points = []

    dc = DayCountConvention.ACT_365_FIXED

    for q in sorted_quotes:
        # Pillar date from tenor
        pillar = _add_months(reference_date, q.tenor_months)
        t = year_fraction(reference_date, pillar, dc)

        if t <= 0:
            continue

        ndf = q.mid
        if ndf <= 0:
            continue

        # Base (G10) discount factor at this tenor
        df_base = base_curve.df(pillar)

        # Implied EM discount factor: df_em = df_base × Spot / NDF
        df_em = df_base * spot_rate / ndf

        # Clamp to (0, 2] — df > 1 implies negative EM rates (possible for some markets)
        # df > 2 indicates data error
        df_em = max(df_em, 1e-10)
        if df_em > 2.0:
            continue  # skip clearly erroneous data point

        # Implied zero rate (continuous compounding)
        zero_rate = -math.log(df_em) / t

        pillar_dates.append(pillar)
        implied_dfs.append(df_em)
        implied_zeros.append(zero_rate)
        fwd_points.append(ndf - spot_rate)

    if not pillar_dates:
        raise ValueError("No valid NDF quotes produced implied discount factors")

    em_curve = DiscountCurve(reference_date, pillar_dates, implied_dfs)

    return NDFImpliedResult(
        em_curve=em_curve,
        pillar_dates=pillar_dates,
        implied_dfs=implied_dfs,
        implied_zero_rates=implied_zeros,
        ndf_forward_points=fwd_points,
        base_currency=base_currency,
        em_currency=em_currency,
        spot_rate=spot_rate,
        n_pillars=len(pillar_dates),
    )


def ndf_from_curves(
    reference_date: date,
    spot_rate: float,
    base_curve: DiscountCurve,
    em_curve: DiscountCurve,
    tenor_months: list[int],
) -> list[float]:
    """Compute theoretical NDF prices from two discount curves.

    NDF(T) = Spot × df_base(T) / df_em(T)

    Useful for:
    - Checking CIP deviations (compare theoretical vs market NDFs)
    - Generating synthetic NDF quotes for stress testing
    """
    ndfs = []
    for m in tenor_months:
        pillar = _add_months(reference_date, m)
        df_base = base_curve.df(pillar)
        df_em = em_curve.df(pillar)
        if df_em > 0:
            ndfs.append(spot_rate * df_base / df_em)
        else:
            ndfs.append(float("inf"))
    return ndfs


def cip_basis(
    reference_date: date,
    spot_rate: float,
    ndf_quotes: list[NDFQuote],
    base_curve: DiscountCurve,
    em_curve: DiscountCurve,
) -> list[dict]:
    """Compute CIP basis (deviation from covered interest parity).

    CIP basis = implied_em_rate - actual_em_rate

    A non-zero CIP basis indicates funding stress or market segmentation.
    Positive basis = EM funding cheaper than NDF-implied.
    Negative basis = EM funding more expensive (typical for restricted currencies).
    """
    dc = DayCountConvention.ACT_365_FIXED
    results = []
    for q in sorted(ndf_quotes, key=lambda q: q.tenor_months):
        pillar = _add_months(reference_date, q.tenor_months)
        t = year_fraction(reference_date, pillar, dc)
        if t <= 0:
            continue

        df_base = base_curve.df(pillar)
        ndf = q.mid

        # Implied EM rate from NDF
        df_em_implied = df_base * spot_rate / ndf
        implied_rate = -math.log(max(df_em_implied, 1e-10)) / t

        # Actual EM rate from curve
        df_em_actual = em_curve.df(pillar)
        actual_rate = -math.log(max(df_em_actual, 1e-10)) / t

        basis_bp = (implied_rate - actual_rate) * 10_000

        results.append({
            "tenor_months": q.tenor_months,
            "implied_rate": implied_rate,
            "actual_rate": actual_rate,
            "basis_bp": basis_bp,
        })

    return results


def _add_months(d: date, months: int) -> date:
    """Add months to a date, handling month-end."""
    year = d.year + (d.month + months - 1) // 12
    month = (d.month + months - 1) % 12 + 1
    day = min(d.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day
