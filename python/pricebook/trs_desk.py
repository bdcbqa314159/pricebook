"""TRS trading desk: risk metrics, carry, P&L attribution, book management.

Same depth as repo_desk.py — production desk infrastructure for TRS.

    from pricebook.trs_desk import (
        trs_risk_metrics, trs_carry_decomposition,
        trs_daily_pnl, TRSBook, trs_dashboard,
        trs_funding_dv01, trs_repo_dv01,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.trs import TotalReturnSwap, TRSResult, FundingLegSpec
from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class TRSRiskMetrics:
    """Complete risk metrics for a TRS position."""
    pv: float
    delta: float            # dPV/dS (equity) or dPV/dy (bond)
    gamma: float            # d²PV/dS² or d²PV/dy²
    dv01: float             # parallel rate shift
    funding_dv01: float     # funding spread shift
    repo_dv01: float        # repo spread shift
    vega: float             # vol sensitivity (equity only)
    cs01: float             # credit spread sensitivity
    notional: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "delta": self.delta, "gamma": self.gamma,
            "dv01": self.dv01, "funding_dv01": self.funding_dv01,
            "repo_dv01": self.repo_dv01, "vega": self.vega,
            "cs01": self.cs01, "notional": self.notional,
        }


def trs_risk_metrics(
    trs: TotalReturnSwap,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
) -> TRSRiskMetrics:
    """Compute all risk metrics for a TRS via bump-and-reprice."""
    base = trs.price(curve, projection_curve)
    h = 0.0001  # 1bp

    # DV01: parallel OIS shift
    pv_up = trs.price(curve.bumped(h), projection_curve).value
    dv01 = pv_up - base.value

    # Funding DV01: bump funding spread
    old_spread = trs.funding.spread
    trs.funding = FundingLegSpec(
        spread=old_spread + h,
        day_count=trs.funding.day_count,
        compounding=trs.funding.compounding,
    )
    pv_fund_up = trs.price(curve, projection_curve).value
    trs.funding = FundingLegSpec(
        spread=old_spread,
        day_count=trs.funding.day_count,
        compounding=trs.funding.compounding,
    )
    funding_dv01 = pv_fund_up - base.value

    # Repo DV01: bump repo spread
    old_repo = trs.repo_spread
    trs.repo_spread = old_repo + h
    pv_repo_up = trs.price(curve, projection_curve).value
    trs.repo_spread = old_repo
    repo_dv01 = pv_repo_up - base.value

    # Delta + Gamma
    delta = 0.0
    gamma = 0.0
    if trs._underlying_type == "equity":
        old_spot = float(trs.underlying)
        bump = old_spot * 0.01  # 1% of spot
        trs.underlying = old_spot + bump
        pv_up_s = trs.price(curve, projection_curve).value
        trs.underlying = old_spot - bump
        pv_dn_s = trs.price(curve, projection_curve).value
        trs.underlying = old_spot
        # Delta per unit spot move (not per %)
        delta = (pv_up_s - pv_dn_s) / (2 * bump)
        gamma = (pv_up_s - 2 * base.value + pv_dn_s) / (bump ** 2)

    # Vega (equity only)
    vega = 0.0
    if trs._underlying_type == "equity" and trs.sigma > 0:
        old_sigma = trs.sigma
        trs.sigma = old_sigma + 0.01
        pv_vol_up = trs.price(curve, projection_curve).value
        trs.sigma = old_sigma
        vega = (pv_vol_up - base.value)

    # CS01: credit spread sensitivity (bond/loan/CLN only)
    cs01 = 0.0
    if trs.survival_curve is not None:
        from pricebook.credit_risk import _bump_survival_curve
        bumped_sc = _bump_survival_curve(trs.survival_curve, h)
        old_sc = trs.survival_curve
        trs.survival_curve = bumped_sc
        pv_cs_up = trs.price(curve, projection_curve).value
        trs.survival_curve = old_sc
        cs01 = pv_cs_up - base.value

    return TRSRiskMetrics(
        pv=base.value, delta=delta, gamma=gamma,
        dv01=dv01, funding_dv01=funding_dv01, repo_dv01=repo_dv01,
        vega=vega, cs01=cs01, notional=trs.notional,
    )


# ---------------------------------------------------------------------------
# Carry decomposition
# ---------------------------------------------------------------------------

@dataclass
class TRSCarryDecomposition:
    """Carry P&L split for a TRS position."""
    total_carry: float
    income: float          # coupon / dividend income
    funding_cost: float    # funding leg cost
    repo_cost: float       # repo financing (FVA)
    net_carry: float       # income - funding - repo

    def to_dict(self) -> dict:
        return {
            "total": self.total_carry, "income": self.income,
            "funding": self.funding_cost, "repo": self.repo_cost,
            "net": self.net_carry,
        }


def trs_carry_decomposition(
    trs: TotalReturnSwap,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
) -> TRSCarryDecomposition:
    """Decompose TRS carry into income, funding, and repo components."""
    result = trs.price(curve, projection_curve)

    income = result.income_return
    funding = result.funding_leg
    repo = result.fva
    net = income - funding - repo

    return TRSCarryDecomposition(
        total_carry=result.value,
        income=income, funding_cost=funding,
        repo_cost=repo, net_carry=net,
    )


# ---------------------------------------------------------------------------
# Daily P&L attribution
# ---------------------------------------------------------------------------

@dataclass
class TRSDailyPnL:
    """Daily P&L decomposition."""
    date: date
    total: float
    delta_pnl: float
    carry_pnl: float
    funding_pnl: float
    unexplained: float

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "total": self.total,
            "delta": self.delta_pnl, "carry": self.carry_pnl,
            "funding": self.funding_pnl, "unexplained": self.unexplained,
        }


def trs_daily_pnl(
    trs: TotalReturnSwap,
    curve_t0: DiscountCurve,
    curve_t1: DiscountCurve,
    date_t1: date,
    projection_curve: DiscountCurve | None = None,
) -> TRSDailyPnL:
    """Daily P&L: PV change + attribution."""
    pv_t0 = trs.price(curve_t0, projection_curve).value
    pv_t1 = trs.price(curve_t1, projection_curve).value
    total = pv_t1 - pv_t0

    # Carry: 1-day accrual at t0 curves
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    daily_carry = trs.price(curve_t0, projection_curve).income_return / max(T * 365, 1)

    # Delta P&L: rate move component
    delta_pnl = total - daily_carry

    # Funding P&L: funding leg change
    funding_t0 = trs.price(curve_t0, projection_curve).funding_leg
    funding_t1 = trs.price(curve_t1, projection_curve).funding_leg
    funding_pnl = -(funding_t1 - funding_t0)

    unexplained = total - delta_pnl - daily_carry

    return TRSDailyPnL(date_t1, total, delta_pnl, daily_carry, funding_pnl, unexplained)


# ---------------------------------------------------------------------------
# TRS Book
# ---------------------------------------------------------------------------

@dataclass
class TRSBookEntry:
    """A TRS position in the book."""
    trade_id: str
    trs: TotalReturnSwap
    counterparty: str = ""
    underlying_type: str = ""
    independent_amount: float = 0.0  # initial margin / IA

    def __post_init__(self):
        if not self.underlying_type:
            self.underlying_type = self.trs._underlying_type


class TRSBook:
    """Collection of TRS positions with aggregation."""

    def __init__(self, name: str = "trs_book"):
        self.name = name
        self._entries: list[TRSBookEntry] = []

    def add(self, entry: TRSBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[TRSBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_pv(self, curve: DiscountCurve, projection: DiscountCurve | None = None) -> float:
        return sum(e.trs.price(curve, projection).value for e in self._entries)

    def total_notional(self) -> float:
        return sum(e.trs.notional for e in self._entries)

    def total_independent_amount(self) -> float:
        """Total Independent Amount (IA) posted across all trades."""
        return sum(e.independent_amount for e in self._entries)

    def by_type(self) -> dict[str, list[TRSBookEntry]]:
        result: dict[str, list[TRSBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.underlying_type, []).append(e)
        return result

    def by_counterparty(self) -> dict[str, list[TRSBookEntry]]:
        result: dict[str, list[TRSBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.counterparty, []).append(e)
        return result

    def aggregate_risk(
        self, curve: DiscountCurve, projection: DiscountCurve | None = None,
    ) -> dict[str, float]:
        """Aggregate risk metrics across the book."""
        total_pv = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_dv01 = 0.0
        total_funding_dv01 = 0.0
        total_vega = 0.0

        for e in self._entries:
            rm = trs_risk_metrics(e.trs, curve, projection)
            total_pv += rm.pv
            total_delta += rm.delta
            total_gamma += rm.gamma
            total_dv01 += rm.dv01
            total_funding_dv01 += rm.funding_dv01
            total_vega += rm.vega

        return {
            "total_pv": total_pv,
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_dv01": total_dv01,
            "total_funding_dv01": total_funding_dv01,
            "total_vega": total_vega,
            "n_positions": len(self._entries),
            "total_notional": self.total_notional(),
            "total_ia": self.total_independent_amount(),
        }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class TRSDashboard:
    """Morning-meeting summary for the TRS desk."""
    date: date
    n_positions: int
    total_pv: float
    total_notional: float
    total_delta: float
    total_dv01: float
    total_funding_dv01: float
    total_vega: float
    total_ia: float
    by_type: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "pv": self.total_pv, "notional": self.total_notional,
            "delta": self.total_delta, "dv01": self.total_dv01,
            "funding_dv01": self.total_funding_dv01, "vega": self.total_vega,
            "ia": self.total_ia, "by_type": self.by_type,
        }


def trs_dashboard(
    book: TRSBook,
    reference_date: date,
    curve: DiscountCurve,
    projection: DiscountCurve | None = None,
) -> TRSDashboard:
    """Build the TRS desk morning dashboard."""
    risk = book.aggregate_risk(curve, projection)
    by_type = {k: len(v) for k, v in book.by_type().items()}

    return TRSDashboard(
        date=reference_date,
        n_positions=risk["n_positions"],
        total_pv=risk["total_pv"],
        total_notional=risk["total_notional"],
        total_delta=risk["total_delta"],
        total_dv01=risk["total_dv01"],
        total_funding_dv01=risk["total_funding_dv01"],
        total_vega=risk["total_vega"],
        total_ia=risk["total_ia"],
        by_type=by_type,
    )


# ---------------------------------------------------------------------------
# TRS XVA: all-in cost
# ---------------------------------------------------------------------------

@dataclass
class TRSAllInCost:
    """True cost of a TRS beyond headline funding spread."""
    funding_cost: float
    fva: float
    kva: float
    mva: float
    total_cost: float
    headline_spread_bps: float
    all_in_spread_bps: float
    hidden_cost_bps: float

    def to_dict(self) -> dict:
        return {
            "funding": self.funding_cost, "fva": self.fva,
            "kva": self.kva, "mva": self.mva,
            "total": self.total_cost,
            "headline_bps": self.headline_spread_bps,
            "all_in_bps": self.all_in_spread_bps,
            "hidden_bps": self.hidden_cost_bps,
        }


def trs_all_in_cost(
    trs: TotalReturnSwap,
    curve: DiscountCurve,
    funding_spread: float = 0.002,
    capital_charge: float = 0.0,
    hurdle_rate: float = 0.10,
    initial_margin: float = 0.0,
) -> TRSAllInCost:
    """Total cost of a TRS including XVA.

    Shows the hidden cost in bps above the headline funding spread.
    """
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    result = trs.price(curve)

    funding = result.funding_leg
    fva = result.fva
    kva = capital_charge * hurdle_rate * T
    mva = initial_margin * funding_spread * T

    total = funding + fva + kva + mva

    # Convert to spread
    annuity = trs.notional * T
    headline = trs.funding.spread * 10_000
    all_in = total / annuity * 10_000 if annuity > 0 else headline
    hidden = all_in - headline

    return TRSAllInCost(
        funding_cost=funding, fva=fva, kva=kva, mva=mva,
        total_cost=total, headline_spread_bps=headline,
        all_in_spread_bps=all_in, hidden_cost_bps=hidden,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class TRSStressResult:
    """One stress scenario result."""
    scenario: str
    description: str
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    funding_pnl: float
    total_pnl: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario, "description": self.description,
            "delta": self.delta_pnl, "gamma": self.gamma_pnl,
            "vega": self.vega_pnl, "funding": self.funding_pnl,
            "total": self.total_pnl,
        }


def trs_stress_suite(
    book: TRSBook,
    curve: DiscountCurve,
    projection: DiscountCurve | None = None,
) -> list[TRSStressResult]:
    """Pre-built stress scenarios for the TRS book."""
    risk = book.aggregate_risk(curve, projection)
    delta = risk["total_delta"]
    gamma = risk["total_gamma"]
    vega = risk["total_vega"]
    fdv01 = risk["total_funding_dv01"]

    scenarios = [
        ("equity_crash", "Equity -20%", -0.20, 0.0, 0.0, 0.0),
        ("rates_up", "Rates +100bp", 0.0, 100.0, 0.0, 0.0),
        ("credit_wide", "Credit spreads +200bp", 0.0, 0.0, 0.0, 200.0),
        ("vol_spike", "Vol +10%", 0.0, 0.0, 10.0, 0.0),
        ("combined", "Equity -10%, rates +50bp, vol +5%", -0.10, 50.0, 5.0, 100.0),
    ]

    results = []
    for name, desc, eq_shock, rate_bp, vol_pct, fund_bp in scenarios:
        d_pnl = delta * eq_shock * risk["total_notional"] / risk["n_positions"] if risk["n_positions"] > 0 else 0
        g_pnl = 0.5 * gamma * (eq_shock * risk["total_notional"] / max(risk["n_positions"], 1)) ** 2
        v_pnl = vega * vol_pct
        f_pnl = fdv01 * fund_bp
        total = d_pnl + g_pnl + v_pnl + f_pnl

        results.append(TRSStressResult(name, desc, d_pnl, g_pnl, v_pnl, f_pnl, total))

    return results
