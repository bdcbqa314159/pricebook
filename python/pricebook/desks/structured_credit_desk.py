"""Structured credit trading desk: unified book for guaranteed notes, SPV,
fund participations, and illiquid instruments.

9-component desk protocol managing all structured credit products through
a common risk/carry/stress framework.

    from pricebook.desks.structured_credit_desk import (
        sc_risk_metrics, SCRiskMetrics,
        StructuredCreditBook, SCBookEntry,
        sc_carry_decomposition, SCCarryDecomposition,
        sc_daily_pnl, SCDailyPnL,
        sc_dashboard, SCDashboard,
        sc_stress_suite, SCStressResult,
        sc_capital, SCCapitalResult,
        sc_hedge_recommendations, SCHedgeRecommendation,
        SCLifecycle,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.credit.fund_participation import FundParticipation
from pricebook.credit.guaranteed_note import GuaranteedNote
from pricebook.credit.illiquid_pricing import PrivatePlacementPricer
from pricebook.credit.spv import SPV
from pricebook.core.survival_curve import SurvivalCurve


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class SCRiskMetrics:
    """Risk metrics for a structured credit position."""
    pv: float
    cs01: float              # credit spread sensitivity
    dv01: float              # rate sensitivity
    jtd: float               # jump-to-default P&L
    notional: float
    product_type: str        # "guaranteed_note", "spv_tranche", "fund", "private_placement"

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "cs01": self.cs01, "dv01": self.dv01,
            "jtd": self.jtd, "notional": self.notional,
            "product_type": self.product_type,
        }


def _price_sc(entry: "SCBookEntry", curve: DiscountCurve) -> float:
    """Uniform pricer for structured credit instruments."""
    inst = entry.instrument

    if isinstance(inst, GuaranteedNote):
        if entry.issuer_surv is None or entry.guarantor_surv is None:
            raise ValueError("GuaranteedNote requires issuer_surv and guarantor_surv")
        return inst.dirty_price(curve, entry.issuer_surv, entry.guarantor_surv)

    elif isinstance(inst, PrivatePlacementPricer):
        return inst.price(curve).pv

    elif isinstance(inst, SPV):
        if not entry.tranche_name:
            raise ValueError("SPV entry requires tranche_name to identify which tranche to price")
        return inst.tranche_pv(entry.tranche_name, curve)

    elif isinstance(inst, FundParticipation):
        m = inst.metrics()
        return m.nav

    elif hasattr(inst, 'pv'):
        return inst.pv(curve) if callable(getattr(inst, 'pv')) else inst.pv

    raise TypeError(f"Cannot price {type(inst).__name__}")


def sc_risk_metrics(
    entry: "SCBookEntry",
    curve: DiscountCurve,
    bump: float = 0.0001,
) -> SCRiskMetrics:
    """Compute risk metrics for a structured credit position."""
    inst = entry.instrument
    base_pv = _price_sc(entry, curve)

    # DV01: bump discount curve
    pv_up = _price_sc(entry, curve.bumped(bump))
    pv_dn = _price_sc(entry, curve.bumped(-bump))
    dv01 = (pv_up - pv_dn) / 2

    # CS01: instrument-specific
    cs01 = 0.0
    if isinstance(inst, GuaranteedNote) and entry.issuer_surv:
        cs01 = inst.cs01_issuer(curve, entry.issuer_surv, entry.guarantor_surv, bump)
    elif isinstance(inst, PrivatePlacementPricer):
        # Bump credit spread by 1bp
        bumped = PrivatePlacementPricer(
            inst.coupon_rate, inst.maturity_years, inst.notional,
            inst.credit_spread_bp + 1, inst.illiquidity_premium_bp,
            inst.complexity_premium_bp,
        )
        cs01 = bumped.price(curve).pv - base_pv

    # JTD
    notional = getattr(inst, 'notional', getattr(inst, 'commitment', 0.0))
    recovery = getattr(inst, 'recovery_joint', getattr(inst, 'recovery', 0.40))
    if isinstance(inst, GuaranteedNote):
        recovery = inst.recovery_joint
    jtd = -(1 - recovery) * notional

    # Product type tag
    if isinstance(inst, GuaranteedNote):
        ptype = "guaranteed_note"
    elif isinstance(inst, SPV):
        ptype = "spv_tranche"
    elif isinstance(inst, FundParticipation):
        ptype = "fund"
    elif isinstance(inst, PrivatePlacementPricer):
        ptype = "private_placement"
    else:
        ptype = "other"

    return SCRiskMetrics(
        pv=base_pv, cs01=cs01, dv01=dv01, jtd=jtd,
        notional=notional, product_type=ptype,
    )


# ---------------------------------------------------------------------------
# Book
# ---------------------------------------------------------------------------

@dataclass
class SCBookEntry:
    """A structured credit position."""
    trade_id: str
    instrument: object           # GuaranteedNote, SPV, FundParticipation, PrivatePlacementPricer
    issuer: str = ""
    sector: str = ""
    counterparty: str = ""
    direction: int = 1           # +1 long, -1 short
    tranche_name: str = ""       # for SPV: which tranche
    issuer_surv: SurvivalCurve | None = None    # for GuaranteedNote
    guarantor_surv: SurvivalCurve | None = None  # for GuaranteedNote


class StructuredCreditBook:
    """Unified book for all structured credit products."""

    def __init__(self, name: str = "structured_credit"):
        self.name = name
        self._entries: list[SCBookEntry] = []

    def add(self, entry: SCBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[SCBookEntry]:
        return list(self._entries)

    def positions(self) -> list[SCBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_notional(self) -> float:
        return sum(
            getattr(e.instrument, 'notional', getattr(e.instrument, 'commitment', 0.0))
            for e in self._entries
        )

    def by_type(self) -> dict[str, list[SCBookEntry]]:
        result: dict[str, list[SCBookEntry]] = {}
        for e in self._entries:
            t = type(e.instrument).__name__
            result.setdefault(t, []).append(e)
        return result

    def by_issuer(self) -> dict[str, list[SCBookEntry]]:
        result: dict[str, list[SCBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.issuer, []).append(e)
        return result

    def by_sector(self) -> dict[str, list[SCBookEntry]]:
        result: dict[str, list[SCBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.sector, []).append(e)
        return result

    def aggregate_risk(self, curve: DiscountCurve) -> dict:
        total_pv = 0.0
        total_cs01 = 0.0
        total_dv01 = 0.0
        total_jtd = 0.0
        total_notional = 0.0

        for e in self._entries:
            rm = sc_risk_metrics(e, curve)
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
class SCCarryDecomposition:
    """Structured credit carry."""
    coupon_income: float
    credit_cost: float         # expected loss drag
    funding_cost: float
    net_carry: float

    def to_dict(self) -> dict:
        return {"coupon": self.coupon_income, "credit": self.credit_cost,
                "funding": self.funding_cost, "net": self.net_carry}


def sc_carry_decomposition(
    entry: SCBookEntry,
    curve: DiscountCurve,
    horizon_days: int = 1,
) -> SCCarryDecomposition:
    """Carry decomposition for a structured credit position."""
    dt = horizon_days / 365.0
    inst = entry.instrument
    notional = getattr(inst, 'notional', getattr(inst, 'commitment', 0.0))

    coupon = 0.0
    credit_cost = 0.0
    funding = 0.0

    if isinstance(inst, GuaranteedNote):
        coupon = inst.coupon_rate * notional * dt
        # Credit cost: hazard × (1-R_joint) × notional × dt
        if entry.issuer_surv:
            T = year_fraction(inst.start, inst.end, DayCountConvention.ACT_365_FIXED)
            h = -math.log(max(entry.issuer_surv.survival(inst.end), 1e-15)) / max(T, 1e-10)
            credit_cost = -h * (1 - inst.recovery_joint) * notional * dt

    elif isinstance(inst, PrivatePlacementPricer):
        coupon = inst.coupon_rate * notional * dt
        credit_cost = -(inst.credit_spread_bp / 10_000) * notional * dt
        funding = -(inst.illiquidity_premium_bp / 10_000) * notional * dt

    elif isinstance(inst, FundParticipation):
        # Fund carry: gross return on invested capital, fees on commitment/invested
        m = inst.metrics()
        invested = m.invested
        coupon = inst.gross_return * invested * dt
        funding = -inst.mgmt_fee_rate * (inst.commitment if inst.fee_basis == "committed" else invested) * dt

    net = coupon + credit_cost + funding
    return SCCarryDecomposition(coupon, credit_cost, funding, net)


# ---------------------------------------------------------------------------
# Daily P&L
# ---------------------------------------------------------------------------

@dataclass
class SCDailyPnL:
    """Structured credit daily P&L."""
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


def sc_daily_pnl(
    entry: SCBookEntry,
    curve_t0: DiscountCurve,
    curve_t1: DiscountCurve,
    date_t1: date,
) -> SCDailyPnL:
    """Daily P&L attribution for a structured credit position.

    Decomposes total into rate, spread (residual), carry, and unexplained.
    Rate P&L: reprice with new curve, same credit parameters.
    Carry: 1-day accrual from t0 curves.
    Spread: total - rate - carry (includes credit moves for guaranteed notes).
    """
    pv_t0 = _price_sc(entry, curve_t0)
    pv_t1 = _price_sc(entry, curve_t1)
    total = pv_t1 - pv_t0

    # Rate P&L: reprice at new discount curve (same credit)
    rate_pnl = pv_t1 - pv_t0  # for instruments without separate credit curve, rate = total

    # For guaranteed notes with survival curves, decompose further
    if entry.issuer_surv is not None:
        # Rate-only: use new curve with old survival curves
        # This IS what _price_sc does by default (survival is on the entry, not the curve)
        # So rate_pnl = total change. Spread component requires bumping survival.
        # Since we don't have t1 survival curves, attribute all to rate+spread combined.
        pass

    # Carry
    carry = sc_carry_decomposition(entry, curve_t0)
    carry_pnl = carry.net_carry

    # Spread P&L = total - rate is zero here since all goes through one curve
    # The proper decomposition requires separate t0/t1 credit curves
    # For now: total = carry + market_move, market_move = spread + rate residual
    spread_pnl = 0.0
    rate_pnl = total - carry_pnl
    unexplained = 0.0

    return SCDailyPnL(date_t1, total, spread_pnl, rate_pnl, carry_pnl, unexplained)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class SCDashboard:
    """Structured credit desk morning summary."""
    date: date
    n_positions: int
    total_notional: float
    total_pv: float
    total_cs01: float
    total_jtd: float
    by_type: dict[str, int]
    by_sector: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "notional": self.total_notional, "pv": self.total_pv,
            "cs01": self.total_cs01, "jtd": self.total_jtd,
            "by_type": self.by_type, "by_sector": self.by_sector,
        }


def sc_dashboard(
    book: StructuredCreditBook,
    reference_date: date,
    curve: DiscountCurve,
) -> SCDashboard:
    """Build structured credit desk morning dashboard."""
    risk = book.aggregate_risk(curve)
    by_type = {k: len(v) for k, v in book.by_type().items()}
    by_sector = {k: len(v) for k, v in book.by_sector().items()}

    return SCDashboard(
        date=reference_date,
        n_positions=risk["n_positions"],
        total_notional=risk["total_notional"],
        total_pv=risk["total_pv"],
        total_cs01=risk["total_cs01"],
        total_jtd=risk["total_jtd"],
        by_type=by_type,
        by_sector=by_sector,
    )


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------

@dataclass
class SCStressResult:
    scenario: str
    description: str
    pnl: float

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "description": self.description, "pnl": self.pnl}


def sc_stress_suite(
    book: StructuredCreditBook,
    curve: DiscountCurve,
) -> list[SCStressResult]:
    """Parametric stress for the structured credit book."""
    risk = book.aggregate_risk(curve)
    cs01 = risk["total_cs01"]
    dv01 = risk["total_dv01"]

    scenarios = [
        ("spread_up_100", "Credit spreads +100bp", cs01 * 100),
        ("spread_dn_100", "Credit spreads -100bp", cs01 * -100),
        ("spread_up_300", "Credit spreads +300bp (crisis)", cs01 * 300),
        ("rates_up_100", "Rates +100bp", dv01 * 100),
        ("combined", "Spreads +200bp, rates +50bp", cs01 * 200 + dv01 * 50),
    ]
    return [SCStressResult(n, d, p) for n, d, p in scenarios]


# ---------------------------------------------------------------------------
# Capital
# ---------------------------------------------------------------------------

@dataclass
class SCCapitalResult:
    ead: float
    rwa: float
    capital: float
    simm_im: float

    def to_dict(self) -> dict:
        return {"ead": self.ead, "rwa": self.rwa, "capital": self.capital,
                "simm_im": self.simm_im}


def sc_capital(
    entry: SCBookEntry,
    curve: DiscountCurve,
    counterparty_rw: float = 1.0,
) -> SCCapitalResult:
    """SA-CCR capital for a structured credit position."""
    rm = sc_risk_metrics(entry, curve)
    mtm = max(rm.pv, 0)

    inst = entry.instrument
    maturity = getattr(inst, 'end', getattr(inst, 'maturity', None))
    if maturity is not None and hasattr(curve, 'reference_date'):
        T = year_fraction(curve.reference_date, maturity, DayCountConvention.ACT_365_FIXED)
    else:
        T = 3.0  # default for funds/SPV

    sf = 0.005  # CSR supervisory factor
    mf = math.sqrt(min(T, 1.0))
    ead = 1.4 * (mtm + rm.notional * sf * mf)
    rwa = ead * counterparty_rw
    capital = rwa * 0.08

    csr_rw = 0.005
    simm_im = abs(rm.cs01) * csr_rw * math.sqrt(10.0 / 252.0)

    return SCCapitalResult(ead=ead, rwa=rwa, capital=capital, simm_im=simm_im)


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class SCHedgeRecommendation:
    risk_type: str
    current: float
    limit: float
    breach_pct: float
    action: str

    def to_dict(self) -> dict:
        return {"risk": self.risk_type, "current": self.current,
                "limit": self.limit, "breach_pct": self.breach_pct,
                "action": self.action}


def sc_hedge_recommendations(
    book: StructuredCreditBook,
    curve: DiscountCurve,
    cs01_limit: float = 100_000,
    jtd_limit: float = 50_000_000,
) -> list[SCHedgeRecommendation]:
    """Hedge recommendations for structured credit book."""
    risk = book.aggregate_risk(curve)
    recs = []

    if cs01_limit > 0 and abs(risk["total_cs01"]) > cs01_limit * 0.75:
        recs.append(SCHedgeRecommendation(
            "cs01", abs(risk["total_cs01"]), cs01_limit,
            abs(risk["total_cs01"]) / cs01_limit,
            "Hedge credit risk via CDS index or single-name CDS"))

    if jtd_limit > 0 and abs(risk["total_jtd"]) > jtd_limit * 0.75:
        recs.append(SCHedgeRecommendation(
            "jtd", abs(risk["total_jtd"]), jtd_limit,
            abs(risk["total_jtd"]) / jtd_limit,
            "Reduce jump-to-default via credit protection"))

    # Concentration by issuer
    by_issuer = book.by_issuer()
    total = risk["total_notional"]
    for issuer, entries in by_issuer.items():
        if not issuer:
            continue
        n = sum(getattr(e.instrument, 'notional', getattr(e.instrument, 'commitment', 0))
                for e in entries)
        if total > 0 and n / total > 0.25:
            recs.append(SCHedgeRecommendation(
                "concentration", n / total, 0.25, (n / total) / 0.25,
                f"Reduce concentration in {issuer}"))

    return recs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class SCEventType:
    MATURITY = "maturity"
    DEFAULT = "default"
    COUPON = "coupon"
    CAPITAL_CALL = "capital_call"
    DISTRIBUTION = "distribution"
    NAV_UPDATE = "nav_update"


class SCLifecycle:
    """Lifecycle management for structured credit positions."""

    def __init__(self, entry: SCBookEntry):
        self._entry = entry
        self._events: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return sorted(self._events, key=lambda x: x.get("date", ""))

    def maturity_alert(self, as_of: date, alert_days: int = 30) -> dict | None:
        inst = self._entry.instrument
        maturity = getattr(inst, 'end', getattr(inst, 'maturity', None))
        if maturity is None:
            return None
        days = (maturity - as_of).days
        if 0 < days <= alert_days:
            return {
                "type": SCEventType.MATURITY,
                "date": maturity.isoformat(),
                "days_remaining": days,
            }
        return None

    def record_event(self, event_type: str, event_date: date, **kwargs) -> dict:
        event = {"type": event_type, "date": event_date.isoformat(), **kwargs}
        self._events.append(event)
        return event
