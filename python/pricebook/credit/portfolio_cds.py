"""Weighted portfolio CDS: arbitrary-weight basket pricing.

Unlike equal-weight CDS indices, this supports arbitrary long/short
positions with different notionals per name.

* :class:`PortfolioCDSResult` — pricing result.
* :func:`portfolio_cds_pv` — PV of a weighted CDS basket.
* :func:`portfolio_par_spread` — par spread of the basket.
* :func:`constituent_cs01` — per-name CS01 contributions.

References:
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*,
    Ch. 7-8, 2008.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.day_count import DayCountConvention, year_fraction


@dataclass
class PortfolioPosition:
    """Single position in the CDS portfolio."""
    name: str
    notional: float       # signed: positive = long protection, negative = short
    spread: float         # running spread (decimal)
    recovery: float = 0.4
    weight: float = 1.0   # relative weight (auto-normalised if needed)


@dataclass
class PortfolioCDSResult:
    """Portfolio CDS pricing result."""
    pv: float
    par_spread: float
    total_notional: float
    net_notional: float     # sum of signed notionals
    n_positions: int
    gross_cs01: float
    net_cs01: float

    def to_dict(self) -> dict:
        return vars(self)


def portfolio_cds_pv(
    reference_date: date,
    maturity_years: float,
    positions: list[PortfolioPosition],
    discount_curve: DiscountCurve,
    survival_curves: list[SurvivalCurve],
    coupon_frequency: int = 4,
) -> PortfolioCDSResult:
    """PV of a weighted CDS basket.

    Each position contributes: notional × (protection_pv − spread × rpv01).

    Args:
        reference_date: valuation date.
        maturity_years: common maturity for all positions.
        positions: list of portfolio positions.
        discount_curve: risk-free discount curve.
        survival_curves: one per position (same order).
        coupon_frequency: premium payments per year.
    """
    if len(positions) != len(survival_curves):
        raise ValueError("positions and survival_curves must have same length")

    dt = 1.0 / coupon_frequency
    n_periods = int(maturity_years * coupon_frequency)

    total_pv = 0.0
    total_gross = 0.0
    total_net = 0.0
    gross_cs01 = 0.0
    net_cs01 = 0.0

    par_prot_total = 0.0
    par_prem_total = 0.0

    for pos, sc in zip(positions, survival_curves):
        prot_pv = 0.0
        rpv01 = 0.0

        prev_q = 1.0
        for i in range(1, n_periods + 1):
            t = i * dt
            t_date = reference_date + timedelta(days=round(t * 365.25))
            q = sc.survival(t_date)
            df = discount_curve.df(t_date)
            default_prob = max(prev_q - q, 0)

            prot_pv += (1 - pos.recovery) * default_prob * df
            rpv01 += dt * q * df
            prev_q = q

        # PV: notional × (protection - premium)
        pos_pv = pos.notional * (prot_pv - pos.spread * rpv01)
        total_pv += pos_pv

        # CS01: sensitivity to 1bp spread change
        cs01 = abs(pos.notional) * rpv01 * 0.0001
        gross_cs01 += cs01
        net_cs01 += math.copysign(cs01, pos.notional)

        total_gross += abs(pos.notional)
        total_net += pos.notional

        # For par spread computation
        par_prot_total += abs(pos.notional) * prot_pv
        par_prem_total += abs(pos.notional) * rpv01

    par_spread = par_prot_total / par_prem_total if par_prem_total > 0 else 0.0

    return PortfolioCDSResult(
        pv=total_pv,
        par_spread=par_spread,
        total_notional=total_gross,
        net_notional=total_net,
        n_positions=len(positions),
        gross_cs01=gross_cs01,
        net_cs01=net_cs01,
    )


def constituent_cs01(
    reference_date: date,
    maturity_years: float,
    positions: list[PortfolioPosition],
    discount_curve: DiscountCurve,
    survival_curves: list[SurvivalCurve],
    coupon_frequency: int = 4,
) -> list[dict]:
    """Per-name CS01 contributions.

    Returns a list of dicts with name, notional, cs01, and % contribution.
    """
    dt = 1.0 / coupon_frequency
    n_periods = int(maturity_years * coupon_frequency)

    results = []
    total_cs01 = 0.0

    for pos, sc in zip(positions, survival_curves):
        rpv01 = 0.0
        prev_q = 1.0
        for i in range(1, n_periods + 1):
            t = i * dt
            t_date = reference_date + timedelta(days=round(t * 365.25))
            q = sc.survival(t_date)
            df = discount_curve.df(t_date)
            rpv01 += dt * q * df
            prev_q = q

        cs01 = pos.notional * rpv01 * 0.0001
        total_cs01 += abs(cs01)
        results.append({
            "name": pos.name,
            "notional": pos.notional,
            "cs01": cs01,
        })

    for r in results:
        r["pct_contribution"] = abs(r["cs01"]) / total_cs01 * 100 if total_cs01 > 0 else 0.0

    return results
