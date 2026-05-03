"""CLN trading desk: risk metrics, carry, P&L attribution, book management.

Same depth as trs_desk.py / repo_desk.py — production desk infrastructure
for Credit-Linked Notes.

    from pricebook.cln_desk import (
        cln_risk_metrics, CLNRiskMetrics,
        cln_carry_decomposition, CLNCarryDecomposition,
        cln_daily_pnl, CLNDailyPnL,
        CLNBook, CLNBookEntry,
        cln_dashboard, CLNDashboard,
    )

Known out-of-scope:
- Real-time Greeks refresh: pricing library, not trading system.
- Corporate actions on reference entity: edge case, per-issuer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.cln import CreditLinkedNote, CLNResult
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class CLNRiskMetrics:
    """Complete risk metrics for a CLN position."""
    pv: float
    dv01: float                    # parallel rate shift +1bp (centred)
    cs01: float                    # credit spread (hazard) shift +1bp
    recovery_sensitivity: float    # recovery +1%
    jump_to_default_pnl: float     # immediate default: R×N - PV
    notional: float
    leverage: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "dv01": self.dv01, "cs01": self.cs01,
            "recovery_sensitivity": self.recovery_sensitivity,
            "jtd": self.jump_to_default_pnl,
            "notional": self.notional, "leverage": self.leverage,
        }


def cln_risk_metrics(
    cln: CreditLinkedNote,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> CLNRiskMetrics:
    """Compute all risk metrics for a CLN via bump-and-reprice.

    DV01: centred difference on discount curve (O(h²)).
    CS01: +1bp hazard rate shift.
    Recovery: +1% recovery bump.
    JTD: loss on immediate default = R × N - PV.
    """
    base = cln.dirty_price(discount_curve, survival_curve)
    greeks = cln.greeks(discount_curve, survival_curve)

    # Centred DV01 (override one-sided from greeks)
    h = 0.0001
    pv_up = cln.dirty_price(discount_curve.bumped(h), survival_curve)
    pv_dn = cln.dirty_price(discount_curve.bumped(-h), survival_curve)
    dv01 = (pv_up - pv_dn) / 2

    # JTD: on immediate default, investor receives recovery × notional
    # but loses their investment (current PV)
    jtd = cln.recovery * cln.notional - base

    return CLNRiskMetrics(
        pv=base,
        dv01=dv01,
        cs01=greeks["cs01"],
        recovery_sensitivity=greeks["recovery_sensitivity"],
        jump_to_default_pnl=jtd,
        notional=cln.notional,
        leverage=cln.leverage,
    )


# ---------------------------------------------------------------------------
# Carry decomposition
# ---------------------------------------------------------------------------

@dataclass
class CLNCarryDecomposition:
    """Carry P&L split for a CLN position.

    CLN carry is fundamentally different from TRS:
    - Coupon income: periodic coupon accrual
    - Default drag: expected loss accrual = h × (1-R) × N per year
    - Funding cost: risk-free opportunity cost
    """
    total_carry: float
    coupon_income: float     # annualised coupon accrual
    default_drag: float      # -h × (1-R) × N (expected loss per year)
    funding_cost: float      # risk-free rate × N (opportunity cost)
    net_carry: float         # coupon - |default_drag| - funding

    def to_dict(self) -> dict:
        return {
            "total": self.total_carry, "coupon": self.coupon_income,
            "default_drag": self.default_drag, "funding": self.funding_cost,
            "net": self.net_carry,
        }


def cln_carry_decomposition(
    cln: CreditLinkedNote,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> CLNCarryDecomposition:
    """Decompose CLN carry into coupon, default drag, and funding."""
    T = year_fraction(cln.start, cln.end, DayCountConvention.ACT_365_FIXED)
    base = cln.dirty_price(discount_curve, survival_curve)

    # Annualised coupon income
    coupon = cln.coupon_rate * cln.notional

    # Default drag: hazard × (1-R) × notional per year
    base_hazard = -math.log(max(survival_curve.survival(cln.end), 1e-15)) / max(T, 1e-10)
    lgd = (1 - cln.recovery) * cln.leverage
    default_drag = -base_hazard * lgd * cln.notional

    # Funding cost: risk-free rate × notional
    r = -math.log(discount_curve.df(cln.end)) / max(T, 1e-10)
    funding = r * cln.notional

    net = coupon + default_drag - funding

    return CLNCarryDecomposition(
        total_carry=base, coupon_income=coupon,
        default_drag=default_drag, funding_cost=funding,
        net_carry=net,
    )


# ---------------------------------------------------------------------------
# Daily P&L attribution
# ---------------------------------------------------------------------------

@dataclass
class CLNDailyPnL:
    """Daily P&L decomposition with credit-specific attribution."""
    date: date
    total: float
    spread_pnl: float       # CS01 × spread change
    rate_pnl: float          # DV01 × rate change
    recovery_pnl: float      # recovery_sens × recovery change
    carry_pnl: float
    theta_pnl: float         # time decay via curve roll-down
    unexplained: float

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "total": self.total,
            "spread": self.spread_pnl, "rate": self.rate_pnl,
            "recovery": self.recovery_pnl, "carry": self.carry_pnl,
            "theta": self.theta_pnl, "unexplained": self.unexplained,
        }


def cln_daily_pnl(
    cln: CreditLinkedNote,
    curve_t0: DiscountCurve,
    curve_t1: DiscountCurve,
    surv_t0: SurvivalCurve,
    surv_t1: SurvivalCurve,
    date_t1: date,
    recovery_change: float = 0.0,
) -> CLNDailyPnL:
    """Daily P&L with credit-specific attribution.

    Spread change inferred from survival curve shift.
    Rate change inferred from discount factor shift.
    """
    from pricebook.pnl_explain import greek_pnl, compute_rolldown

    pv_t0 = cln.dirty_price(curve_t0, surv_t0)
    pv_t1 = cln.dirty_price(curve_t1, surv_t1)
    total = pv_t1 - pv_t0

    rm = cln_risk_metrics(cln, curve_t0, surv_t0)
    T = year_fraction(cln.start, cln.end, DayCountConvention.ACT_365_FIXED)

    # Carry: 1-day accrual
    carry = rm.notional * cln.coupon_rate / 365

    # Spread P&L: infer spread change from survival curves
    h0 = -math.log(max(surv_t0.survival(cln.end), 1e-15)) / max(T, 1e-10)
    h1 = -math.log(max(surv_t1.survival(cln.end), 1e-15)) / max(T, 1e-10)
    spread_change = h1 - h0
    spread_pnl = rm.cs01 * spread_change / 0.0001 if abs(spread_change) > 1e-12 else 0.0

    # Rate P&L: infer from DF change
    try:
        t1y = cln.start + timedelta(days=365)
        r0 = -math.log(curve_t0.df(t1y))
        r1 = -math.log(curve_t1.df(t1y))
        rate_change = r1 - r0
    except (ValueError, AttributeError):
        rate_change = 0.0
    rate_pnl = greek_pnl(rm.dv01, rate_change * 10_000)

    # Recovery P&L
    recovery_pnl = rm.recovery_sensitivity * recovery_change / 0.01 if recovery_change != 0 else 0.0

    # Theta via curve roll-down
    theta_pnl = compute_rolldown(
        lambda c: cln.dirty_price(c, surv_t0), curve_t0, days=1,
    )

    explained = spread_pnl + rate_pnl + recovery_pnl + carry + theta_pnl
    unexplained = total - explained

    return CLNDailyPnL(
        date_t1, total, spread_pnl, rate_pnl, recovery_pnl,
        carry, theta_pnl, unexplained,
    )


# ---------------------------------------------------------------------------
# CLN Book
# ---------------------------------------------------------------------------

@dataclass
class CLNBookEntry:
    """A CLN position in the book."""
    trade_id: str
    cln: CreditLinkedNote
    survival_curve: SurvivalCurve
    counterparty: str = ""
    issuer: str = ""
    seniority: str = "senior_unsecured"
    independent_amount: float = 0.0

    @property
    def underlying_type(self) -> str:
        return "leveraged" if self.cln.leverage > 1.0 else "vanilla"


class CLNBook:
    """Collection of CLN positions with aggregation."""

    def __init__(self, name: str = "cln_book"):
        self.name = name
        self._entries: list[CLNBookEntry] = []

    def add(self, entry: CLNBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[CLNBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_pv(self, curve: DiscountCurve) -> float:
        return sum(
            e.cln.dirty_price(curve, e.survival_curve) for e in self._entries
        )

    def total_notional(self) -> float:
        return sum(e.cln.notional for e in self._entries)

    def total_independent_amount(self) -> float:
        return sum(e.independent_amount for e in self._entries)

    def by_issuer(self) -> dict[str, list[CLNBookEntry]]:
        result: dict[str, list[CLNBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.issuer, []).append(e)
        return result

    def by_seniority(self) -> dict[str, list[CLNBookEntry]]:
        result: dict[str, list[CLNBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.seniority, []).append(e)
        return result

    def by_counterparty(self) -> dict[str, list[CLNBookEntry]]:
        result: dict[str, list[CLNBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.counterparty, []).append(e)
        return result

    def aggregate_risk(self, curve: DiscountCurve) -> dict[str, float]:
        """Aggregate risk metrics across the book."""
        total_pv = 0.0
        total_dv01 = 0.0
        total_cs01 = 0.0
        total_jtd = 0.0
        total_recovery_sens = 0.0

        for e in self._entries:
            rm = cln_risk_metrics(e.cln, curve, e.survival_curve)
            total_pv += rm.pv
            total_dv01 += rm.dv01
            total_cs01 += rm.cs01
            total_jtd += rm.jump_to_default_pnl
            total_recovery_sens += rm.recovery_sensitivity

        return {
            "total_pv": total_pv,
            "total_dv01": total_dv01,
            "total_cs01": total_cs01,
            "total_jtd": total_jtd,
            "total_recovery_sens": total_recovery_sens,
            "n_positions": len(self._entries),
            "total_notional": self.total_notional(),
            "total_ia": self.total_independent_amount(),
        }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class CLNDashboard:
    """Morning-meeting summary for the CLN desk."""
    date: date
    n_positions: int
    total_pv: float
    total_notional: float
    total_cs01: float
    total_dv01: float
    total_jtd: float
    total_recovery_sens: float
    total_ia: float
    by_issuer: dict[str, int]
    by_seniority: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "pv": self.total_pv, "notional": self.total_notional,
            "cs01": self.total_cs01, "dv01": self.total_dv01,
            "jtd": self.total_jtd, "recovery_sens": self.total_recovery_sens,
            "ia": self.total_ia, "by_issuer": self.by_issuer,
            "by_seniority": self.by_seniority,
        }


def cln_dashboard(
    book: CLNBook,
    reference_date: date,
    curve: DiscountCurve,
) -> CLNDashboard:
    """Build the CLN desk morning dashboard."""
    risk = book.aggregate_risk(curve)
    by_issuer = {k: len(v) for k, v in book.by_issuer().items()}
    by_seniority = {k: len(v) for k, v in book.by_seniority().items()}

    return CLNDashboard(
        date=reference_date,
        n_positions=risk["n_positions"],
        total_pv=risk["total_pv"],
        total_notional=risk["total_notional"],
        total_cs01=risk["total_cs01"],
        total_dv01=risk["total_dv01"],
        total_jtd=risk["total_jtd"],
        total_recovery_sens=risk["total_recovery_sens"],
        total_ia=risk["total_ia"],
        by_issuer=by_issuer,
        by_seniority=by_seniority,
    )
