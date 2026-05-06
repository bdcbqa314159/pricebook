"""Risk participation trading desk: book, risk, carry, stress, capital, lifecycle.

9-component desk protocol for risk participations (unfunded credit risk transfer).

    from pricebook.risk_participation_desk import (
        rp_risk_metrics, RPRiskMetrics,
        RPBook, RPBookEntry,
        rp_carry_decomposition, RPCarryDecomposition,
        rp_daily_pnl, RPDailyPnL,
        rp_dashboard, RPDashboard,
        rp_stress_suite, RPStressResult,
        rp_capital, RPCapitalResult,
        rp_hedge_recommendations, RPHedgeRecommendation,
        RPLifecycle,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.risk_participation import RiskParticipation


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class RPRiskMetrics:
    """Risk metrics for a risk participation position."""
    pv: float
    cs01: float              # credit spread sensitivity (1bp hazard)
    dv01: float              # rate sensitivity (1bp discount curve)
    jtd: float               # jump-to-default P&L
    par_spread: float        # spread at which PV = 0
    notional: float          # participation notional

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "cs01": self.cs01, "dv01": self.dv01,
            "jtd": self.jtd, "par_spread": self.par_spread,
            "notional": self.notional,
        }


def rp_risk_metrics(
    rp: RiskParticipation,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    bump: float = 0.0001,
) -> RPRiskMetrics:
    """Compute risk participation risk metrics."""
    result = rp.price(discount_curve, survival_curve)
    base_pv = result.pv

    # CS01: centred difference on survival curve
    pv_up = rp.price(discount_curve, survival_curve.bumped(bump)).pv
    pv_dn = rp.price(discount_curve, survival_curve.bumped(-bump)).pv
    cs01 = (pv_up - pv_dn) / 2

    # DV01: centred difference on discount curve
    pv_disc_up = rp.price(discount_curve.bumped(bump), survival_curve).pv
    pv_disc_dn = rp.price(discount_curve.bumped(-bump), survival_curve).pv
    dv01 = (pv_disc_up - pv_disc_dn) / 2

    jtd = rp.jtd(discount_curve, survival_curve)

    return RPRiskMetrics(
        pv=base_pv, cs01=cs01, dv01=dv01, jtd=jtd,
        par_spread=result.par_spread, notional=rp.notional,
    )


# ---------------------------------------------------------------------------
# Book
# ---------------------------------------------------------------------------

@dataclass
class RPBookEntry:
    """A single risk participation position."""
    trade_id: str
    instrument: RiskParticipation
    survival_curve: SurvivalCurve
    counterparty: str = ""
    borrower: str = ""
    sector: str = ""
    direction: int = 1         # +1 = participant (long risk), -1 = originator


class RPBook:
    """Risk participation position book."""

    def __init__(self, name: str = "rp_book"):
        self.name = name
        self._entries: list[RPBookEntry] = []

    def add(self, entry: RPBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[RPBookEntry]:
        return list(self._entries)

    def positions(self) -> list[RPBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_notional(self) -> float:
        return sum(e.instrument.notional for e in self._entries)

    def by_borrower(self) -> dict[str, list[RPBookEntry]]:
        result: dict[str, list[RPBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.borrower, []).append(e)
        return result

    def by_sector(self) -> dict[str, list[RPBookEntry]]:
        result: dict[str, list[RPBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.sector, []).append(e)
        return result

    def aggregate_risk(self, discount_curve: DiscountCurve) -> dict:
        """Aggregate risk across all positions."""
        total_pv = 0.0
        total_cs01 = 0.0
        total_dv01 = 0.0
        total_jtd = 0.0
        total_notional = 0.0

        for e in self._entries:
            rm = rp_risk_metrics(e.instrument, discount_curve, e.survival_curve)
            total_pv += e.direction * rm.pv
            total_cs01 += e.direction * rm.cs01
            total_dv01 += e.direction * rm.dv01
            total_jtd += e.direction * rm.jtd
            total_notional += rm.notional

        return {
            "total_pv": total_pv,
            "total_cs01": total_cs01,
            "total_dv01": total_dv01,
            "total_jtd": total_jtd,
            "n_positions": len(self._entries),
            "total_notional": total_notional,
        }


# ---------------------------------------------------------------------------
# Carry
# ---------------------------------------------------------------------------

@dataclass
class RPCarryDecomposition:
    """Risk participation carry attribution."""
    fee_income: float          # running spread accrual
    default_drag: float        # expected loss accrual
    net_carry: float

    def to_dict(self) -> dict:
        return {"fee_income": self.fee_income, "default_drag": self.default_drag,
                "net": self.net_carry}


def rp_carry_decomposition(
    rp: RiskParticipation,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    horizon_days: int = 1,
) -> RPCarryDecomposition:
    """Carry decomposition for a risk participation."""
    dt = horizon_days / 365.0
    T = year_fraction(rp.start, rp.end, DayCountConvention.ACT_365_FIXED)
    h = -math.log(max(survival_curve.survival(rp.end), 1e-15)) / max(T, 1e-10)

    fee_income = rp.spread * rp.notional * dt
    default_drag = -h * (1 - rp.recovery) * rp.notional * dt
    net = fee_income + default_drag

    return RPCarryDecomposition(
        fee_income=fee_income,
        default_drag=default_drag,
        net_carry=net,
    )


# ---------------------------------------------------------------------------
# Daily P&L
# ---------------------------------------------------------------------------

@dataclass
class RPDailyPnL:
    """Risk participation daily P&L attribution."""
    date: date
    total: float
    spread_pnl: float
    rate_pnl: float
    carry_pnl: float
    unexplained: float

    def to_dict(self) -> dict:
        return {"date": self.date.isoformat(), "total": self.total,
                "spread": self.spread_pnl, "rate": self.rate_pnl,
                "carry": self.carry_pnl, "unexplained": self.unexplained}


def rp_daily_pnl(
    rp: RiskParticipation,
    disc_t0: DiscountCurve,
    surv_t0: SurvivalCurve,
    disc_t1: DiscountCurve,
    surv_t1: SurvivalCurve,
    date_t1: date,
) -> RPDailyPnL:
    """Daily P&L attribution for a risk participation."""
    pv_t0 = rp.price(disc_t0, surv_t0).pv
    pv_t1 = rp.price(disc_t1, surv_t1).pv
    total = pv_t1 - pv_t0

    # Spread P&L: new survival, old discount
    pv_spread_only = rp.price(disc_t0, surv_t1).pv
    spread_pnl = pv_spread_only - pv_t0

    # Rate P&L: new discount, old survival
    pv_rate_only = rp.price(disc_t1, surv_t0).pv
    rate_pnl = pv_rate_only - pv_t0

    carry = rp_carry_decomposition(rp, disc_t0, surv_t0, horizon_days=1)
    carry_pnl = carry.net_carry

    unexplained = total - spread_pnl - rate_pnl - carry_pnl

    return RPDailyPnL(
        date=date_t1, total=total,
        spread_pnl=spread_pnl, rate_pnl=rate_pnl,
        carry_pnl=carry_pnl, unexplained=unexplained,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class RPDashboard:
    """Risk participation desk morning summary."""
    date: date
    n_positions: int
    total_notional: float
    total_pv: float
    total_cs01: float
    total_jtd: float
    by_sector: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "notional": self.total_notional, "pv": self.total_pv,
            "cs01": self.total_cs01, "jtd": self.total_jtd,
            "by_sector": self.by_sector,
        }


def rp_dashboard(
    book: RPBook,
    reference_date: date,
    discount_curve: DiscountCurve,
) -> RPDashboard:
    """Build risk participation desk morning dashboard."""
    risk = book.aggregate_risk(discount_curve)
    by_sector = {k: len(v) for k, v in book.by_sector().items()}

    return RPDashboard(
        date=reference_date,
        n_positions=risk["n_positions"],
        total_notional=risk["total_notional"],
        total_pv=risk["total_pv"],
        total_cs01=risk["total_cs01"],
        total_jtd=risk["total_jtd"],
        by_sector=by_sector,
    )


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------

@dataclass
class RPStressResult:
    scenario: str
    description: str
    pnl: float

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "description": self.description, "pnl": self.pnl}


def rp_stress_suite(
    book: RPBook,
    discount_curve: DiscountCurve,
) -> list[RPStressResult]:
    """Parametric stress scenarios for risk participation book."""
    risk = book.aggregate_risk(discount_curve)
    cs01 = risk["total_cs01"]
    dv01 = risk["total_dv01"]

    scenarios = [
        ("spread_up_100", "Credit spreads +100bp", cs01 * 100),
        ("spread_dn_100", "Credit spreads -100bp", cs01 * -100),
        ("spread_up_250", "Credit spreads +250bp (stress)", cs01 * 250),
        ("rates_up_100", "Rates +100bp", dv01 * 100),
        ("combined", "Spreads +200bp, rates +50bp", cs01 * 200 + dv01 * 50),
    ]
    return [RPStressResult(n, d, p) for n, d, p in scenarios]


# ---------------------------------------------------------------------------
# Capital
# ---------------------------------------------------------------------------

@dataclass
class RPCapitalResult:
    ead: float
    rwa: float
    capital: float
    simm_im: float

    def to_dict(self) -> dict:
        return {"ead": self.ead, "rwa": self.rwa, "capital": self.capital,
                "simm_im": self.simm_im}


def rp_capital(
    rp: RiskParticipation,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    counterparty_rw: float = 1.0,
) -> RPCapitalResult:
    """SA-CCR capital for a risk participation."""
    rm = rp_risk_metrics(rp, discount_curve, survival_curve)
    mtm = max(rm.pv, 0)
    T = year_fraction(rp.start, rp.end, DayCountConvention.ACT_365_FIXED)

    sf = 0.005  # CSR supervisory factor
    mf = math.sqrt(min(T, 1.0))
    ead = 1.4 * (mtm + rp.notional * sf * mf)
    rwa = ead * counterparty_rw
    capital = rwa * 0.08

    # SIMM: CS01 into CSR non-sec bucket
    csr_rw = 0.005
    simm_im = abs(rm.cs01) * csr_rw * math.sqrt(10.0 / 252.0)

    return RPCapitalResult(ead=ead, rwa=rwa, capital=capital, simm_im=simm_im)


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class RPHedgeRecommendation:
    risk_type: str
    current: float
    limit: float
    breach_pct: float
    action: str

    def to_dict(self) -> dict:
        return {"risk": self.risk_type, "current": self.current,
                "limit": self.limit, "breach_pct": self.breach_pct,
                "action": self.action}


def rp_hedge_recommendations(
    book: RPBook,
    discount_curve: DiscountCurve,
    cs01_limit: float = 100_000,
    jtd_limit: float = 50_000_000,
) -> list[RPHedgeRecommendation]:
    """Hedge recommendations for risk participation book."""
    risk = book.aggregate_risk(discount_curve)
    recs = []

    if cs01_limit > 0 and abs(risk["total_cs01"]) > cs01_limit * 0.75:
        recs.append(RPHedgeRecommendation(
            "cs01", abs(risk["total_cs01"]), cs01_limit,
            abs(risk["total_cs01"]) / cs01_limit,
            "Hedge credit risk via CDS or offsetting participation"))

    if jtd_limit > 0 and abs(risk["total_jtd"]) > jtd_limit * 0.75:
        recs.append(RPHedgeRecommendation(
            "jtd", abs(risk["total_jtd"]), jtd_limit,
            abs(risk["total_jtd"]) / jtd_limit,
            "Reduce jump-to-default exposure via CDS protection"))

    # Concentration by borrower
    by_borrower = book.by_borrower()
    total = risk["total_notional"]
    for borrower, entries in by_borrower.items():
        n = sum(e.instrument.notional for e in entries)
        if total > 0 and n / total > 0.25:
            recs.append(RPHedgeRecommendation(
                "concentration", n / total, 0.25, (n / total) / 0.25,
                f"Reduce concentration in {borrower}"))

    return recs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class RPEventType:
    LOAN_DEFAULT = "loan_default"
    FEE_PAYMENT = "fee_payment"
    MATURITY = "maturity"
    NOVATION = "novation"


class RPLifecycle:
    """Lifecycle management for risk participation positions."""

    def __init__(self, instrument: RiskParticipation, trade_id: str = ""):
        self._instrument = instrument
        self._trade_id = trade_id
        self._events: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return sorted(self._events, key=lambda x: x.get("date", ""))

    def maturity_alert(self, as_of: date, alert_days: int = 30) -> dict | None:
        days = (self._instrument.end - as_of).days
        if 0 < days <= alert_days:
            return {
                "type": RPEventType.MATURITY,
                "date": self._instrument.end.isoformat(),
                "days_remaining": days,
            }
        return None

    def record_fee_payment(self, payment_date: date, amount: float) -> dict:
        event = {
            "type": RPEventType.FEE_PAYMENT,
            "date": payment_date.isoformat(),
            "amount": amount,
        }
        self._events.append(event)
        return event

    def record_default(self, default_date: date, loss_amount: float,
                       recovery_amount: float) -> dict:
        event = {
            "type": RPEventType.LOAN_DEFAULT,
            "date": default_date.isoformat(),
            "loss": loss_amount,
            "recovery": recovery_amount,
            "net_payout": loss_amount - recovery_amount,
        }
        self._events.append(event)
        return event
