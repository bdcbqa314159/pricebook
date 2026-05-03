"""CDS trading desk: single-name, index, swaption — unified risk, P&L, book.

Consolidates CDS pricing (cds.py, cds_index_product.py, cds_swaption.py)
into a unified desk layer.

    from pricebook.cds_desk import (
        cds_risk_metrics, CDSRiskMetrics,
        CDSBook, CDSBookEntry,
        cds_carry_decomposition, cds_daily_pnl,
        cds_dashboard, CDSDashboard,
        cds_stress_suite, cds_scenario_stress,
        cds_capital, cds_hedge_recommendations,
        CDSLifecycle,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

from pricebook.cds import CDS
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.pricing_context import PricingContext
from pricebook.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class CDSRiskMetrics:
    """Unified risk metrics for a CDS position."""
    pv: float
    par_spread: float
    rpv01: float               # risky annuity
    cs01: float                # centred 1bp spread sensitivity
    bucket_cs01: dict[str, float]  # per-pillar CS01
    rec01: float               # recovery sensitivity
    theta: float               # 1-day time decay
    carry: float               # 30-day premium carry
    roll_down: float           # 30-day roll-down P&L
    spread_duration: float
    spread_convexity: float
    jump_to_default: float     # immediate default P&L
    notional: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "par_spread": self.par_spread, "rpv01": self.rpv01,
            "cs01": self.cs01, "bucket_cs01": self.bucket_cs01,
            "rec01": self.rec01, "theta": self.theta,
            "carry": self.carry, "roll_down": self.roll_down,
            "spread_duration": self.spread_duration,
            "spread_convexity": self.spread_convexity,
            "jtd": self.jump_to_default, "notional": self.notional,
        }


def cds_risk_metrics(
    cds: CDS,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> CDSRiskMetrics:
    """Compute unified risk metrics for a CDS."""
    pv = cds.pv(discount_curve, survival_curve)
    par = cds.par_spread(discount_curve, survival_curve)
    rpv01 = cds.rpv01(discount_curve, survival_curve)

    # Centred CS01
    h = 0.0001
    surv_up = survival_curve.bumped(h)
    surv_dn = survival_curve.bumped(-h)
    cs01 = (cds.pv(discount_curve, surv_up) - cds.pv(discount_curve, surv_dn)) / 2

    # Bucket CS01
    try:
        bucket = cds.bucket_cs01(discount_curve, survival_curve)
    except (AttributeError, TypeError):
        bucket = {}

    rec01 = cds.rec01(discount_curve, survival_curve)
    theta = cds.theta(discount_curve, survival_curve)
    carry = cds.carry(discount_curve, survival_curve)
    roll = cds.roll_down(discount_curve, survival_curve)
    sdur = cds.spread_duration(discount_curve, survival_curve)
    sconv = cds.spread_convexity(discount_curve, survival_curve)

    # JTD: on immediate default, protection buyer receives (1-R)*N, loses accrued premium
    jtd = (1 - cds.recovery) * cds.notional - pv  # net gain/loss

    return CDSRiskMetrics(
        pv=pv, par_spread=par, rpv01=rpv01,
        cs01=cs01, bucket_cs01=bucket,
        rec01=rec01, theta=theta, carry=carry, roll_down=roll,
        spread_duration=sdur, spread_convexity=sconv,
        jump_to_default=jtd, notional=cds.notional,
    )


# ---------------------------------------------------------------------------
# CDS Book (unified: single-name + index + swaption)
# ---------------------------------------------------------------------------

class CDSProductType(Enum):
    SINGLE_NAME = "single_name"
    INDEX = "index"
    SWAPTION = "swaption"


@dataclass
class CDSBookEntry:
    """A CDS position in the book."""
    trade_id: str
    instrument: object  # CDS, CDSIndexProduct, or swaption
    survival_curve: SurvivalCurve
    product_type: CDSProductType = CDSProductType.SINGLE_NAME
    counterparty: str = ""
    reference_name: str = ""
    sector: str = ""
    rating: str = ""


class CDSBook:
    """Collection of CDS positions with multi-dimensional aggregation."""

    def __init__(self, name: str = "cds_book"):
        self.name = name
        self._entries: list[CDSBookEntry] = []

    def add(self, entry: CDSBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[CDSBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_notional(self) -> float:
        return sum(
            e.instrument.notional for e in self._entries
            if hasattr(e.instrument, 'notional')
        )

    def by_name(self) -> dict[str, list[CDSBookEntry]]:
        result: dict[str, list[CDSBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.reference_name, []).append(e)
        return result

    def by_sector(self) -> dict[str, list[CDSBookEntry]]:
        result: dict[str, list[CDSBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.sector, []).append(e)
        return result

    def by_type(self) -> dict[str, list[CDSBookEntry]]:
        result: dict[str, list[CDSBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.product_type.value, []).append(e)
        return result

    def by_counterparty(self) -> dict[str, list[CDSBookEntry]]:
        result: dict[str, list[CDSBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.counterparty, []).append(e)
        return result

    def aggregate_risk(self, curve: DiscountCurve) -> dict[str, float]:
        """Aggregate risk across all single-name CDS positions."""
        total_pv = 0.0
        total_cs01 = 0.0
        total_jtd = 0.0
        total_rec01 = 0.0
        total_notional = 0.0

        for e in self._entries:
            if e.product_type == CDSProductType.SINGLE_NAME and isinstance(e.instrument, CDS):
                rm = cds_risk_metrics(e.instrument, curve, e.survival_curve)
                total_pv += rm.pv
                total_cs01 += rm.cs01
                total_jtd += rm.jump_to_default
                total_rec01 += rm.rec01
                total_notional += rm.notional

        return {
            "total_pv": total_pv,
            "total_cs01": total_cs01,
            "total_jtd": total_jtd,
            "total_rec01": total_rec01,
            "n_positions": len(self._entries),
            "total_notional": total_notional,
        }


# ---------------------------------------------------------------------------
# Carry + P&L
# ---------------------------------------------------------------------------

@dataclass
class CDSCarryDecomposition:
    """Carry decomposition for a CDS position."""
    premium_income: float    # spread accrual (what protection buyer pays)
    default_risk: float      # expected loss component
    roll_down: float         # curve aging P&L
    net_carry: float

    def to_dict(self) -> dict:
        return {
            "premium": self.premium_income, "default_risk": self.default_risk,
            "roll_down": self.roll_down, "net": self.net_carry,
        }


def cds_carry_decomposition(
    cds: CDS,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> CDSCarryDecomposition:
    """Decompose CDS carry into premium, default risk, and roll-down."""
    carry = cds.carry(discount_curve, survival_curve)
    roll = cds.roll_down(discount_curve, survival_curve)

    T = year_fraction(cds.start, cds.end, DayCountConvention.ACT_365_FIXED)
    h = -math.log(max(survival_curve.survival(cds.end), 1e-15)) / max(T, 1e-10)

    # Premium income: spread × notional × (30/360)
    premium = cds.spread * cds.notional * 30 / 360

    # Default risk: hazard × (1-R) × notional × (30/365)
    default_risk = -h * (1 - cds.recovery) * cds.notional * 30 / 365

    net = premium + default_risk + roll

    return CDSCarryDecomposition(
        premium_income=premium, default_risk=default_risk,
        roll_down=roll, net_carry=net,
    )


@dataclass
class CDSDailyPnL:
    """Daily P&L decomposition for a CDS."""
    date: date
    total: float
    spread_pnl: float
    carry: float
    roll_down: float
    convexity_pnl: float
    unexplained: float

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "total": self.total,
            "spread": self.spread_pnl, "carry": self.carry,
            "roll_down": self.roll_down, "convexity": self.convexity_pnl,
            "unexplained": self.unexplained,
        }


def cds_daily_pnl(
    cds: CDS,
    curve_t0: DiscountCurve,
    curve_t1: DiscountCurve,
    surv_t0: SurvivalCurve,
    surv_t1: SurvivalCurve,
    date_t1: date,
) -> CDSDailyPnL:
    """Daily P&L with credit-specific attribution."""
    from pricebook.cds import cds_pnl_attribution

    pv_t0 = cds.pv(curve_t0, surv_t0)
    pv_t1 = cds.pv(curve_t1, surv_t1)
    total = pv_t1 - pv_t0

    # Use existing P&L attribution
    try:
        attr = cds_pnl_attribution(cds, curve_t0, surv_t0, curve_t1, surv_t1)
        return CDSDailyPnL(
            date_t1, total, attr.spread, attr.carry, attr.roll_down,
            attr.convexity, attr.residual,
        )
    except (AttributeError, TypeError):
        # Fallback: simple attribution
        rm = cds_risk_metrics(cds, curve_t0, surv_t0)
        T = year_fraction(cds.start, cds.end, DayCountConvention.ACT_365_FIXED)
        h0 = -math.log(max(surv_t0.survival(cds.end), 1e-15)) / max(T, 1e-10)
        h1 = -math.log(max(surv_t1.survival(cds.end), 1e-15)) / max(T, 1e-10)
        spread_change = h1 - h0
        spread_pnl = rm.cs01 * spread_change / 0.0001
        carry = rm.carry / 12  # monthly → daily approx
        unexplained = total - spread_pnl - carry

        return CDSDailyPnL(date_t1, total, spread_pnl, carry, 0.0, 0.0, unexplained)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class CDSDashboard:
    """Morning-meeting summary for the CDS desk."""
    date: date
    n_positions: int
    total_notional: float
    total_pv: float
    total_cs01: float
    total_jtd: float
    total_rec01: float
    by_name: dict[str, int]
    by_sector: dict[str, int]
    by_type: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "notional": self.total_notional, "pv": self.total_pv,
            "cs01": self.total_cs01, "jtd": self.total_jtd,
            "rec01": self.total_rec01,
            "by_name": self.by_name, "by_sector": self.by_sector,
            "by_type": self.by_type,
        }


def cds_dashboard(
    book: CDSBook,
    reference_date: date,
    curve: DiscountCurve,
) -> CDSDashboard:
    """Build CDS desk morning dashboard."""
    risk = book.aggregate_risk(curve)
    by_name = {k: len(v) for k, v in book.by_name().items()}
    by_sector = {k: len(v) for k, v in book.by_sector().items()}
    by_type = {k: len(v) for k, v in book.by_type().items()}

    return CDSDashboard(
        date=reference_date, n_positions=risk["n_positions"],
        total_notional=risk["total_notional"], total_pv=risk["total_pv"],
        total_cs01=risk["total_cs01"], total_jtd=risk["total_jtd"],
        total_rec01=risk["total_rec01"],
        by_name=by_name, by_sector=by_sector, by_type=by_type,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class CDSStressResult:
    scenario: str
    description: str
    spread_pnl: float
    recovery_pnl: float
    total_pnl: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario, "description": self.description,
            "spread": self.spread_pnl, "recovery": self.recovery_pnl,
            "total": self.total_pnl,
        }


def cds_stress_suite(
    book: CDSBook,
    curve: DiscountCurve,
) -> list[CDSStressResult]:
    """Five CDS-specific stress scenarios."""
    risk = book.aggregate_risk(curve)
    cs01 = risk["total_cs01"]
    rec01 = risk["total_rec01"]

    scenarios = [
        ("spread_wide_200", "Spreads +200bp", 200.0, 0.0),
        ("spread_tight_100", "Spreads -100bp", -100.0, 0.0),
        ("recovery_down", "Recovery -20%", 0.0, -0.20),
        ("spread_wide_100", "Spreads +100bp", 100.0, 0.0),
        ("combined", "Spreads +200bp, recovery -20%", 200.0, -0.20),
    ]

    results = []
    for name, desc, spread_bp, rec_pct in scenarios:
        s_pnl = cs01 * spread_bp
        r_pnl = rec01 * rec_pct / 0.01 if rec_pct != 0 else 0.0
        results.append(CDSStressResult(name, desc, s_pnl, r_pnl, s_pnl + r_pnl))

    return results


def cds_scenario_stress(
    book: CDSBook,
    ctx: PricingContext,
    scenarios: list | None = None,
) -> list:
    """Full-reprice stress via scenario.py."""
    from pricebook.scenario import parallel_shift, credit_spread_shift, run_scenarios
    from pricebook.trade import Trade, Portfolio

    portfolio = Portfolio(name=book.name)
    for e in book.entries:
        if hasattr(e.instrument, 'pv_ctx'):
            portfolio.add(Trade(instrument=e.instrument, trade_id=e.trade_id))

    if scenarios is None:
        scenarios = [
            credit_spread_shift(0.01, name="spreads_+100bp"),
            credit_spread_shift(0.02, name="spreads_+200bp"),
            credit_spread_shift(-0.01, name="spreads_-100bp"),
            parallel_shift(0.01, "rates_+100bp"),
        ]

    return run_scenarios(portfolio, ctx, scenarios)


# ---------------------------------------------------------------------------
# Capital
# ---------------------------------------------------------------------------

@dataclass
class CDSCapitalResult:
    ead: float
    rwa: float
    capital: float
    simm_im: float

    def to_dict(self) -> dict:
        return {"ead": self.ead, "rwa": self.rwa, "capital": self.capital, "simm_im": self.simm_im}


def cds_capital(
    cds: CDS,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    counterparty_rw: float = 1.0,
) -> CDSCapitalResult:
    """SA-CCR capital for a single-name CDS. SF=0.005 for credit."""
    T = year_fraction(cds.start, cds.end, DayCountConvention.ACT_365_FIXED)
    pv = cds.pv(discount_curve, survival_curve)
    mtm = max(abs(pv), 0)

    sf = 0.005
    mf = math.sqrt(min(T, 1.0))
    ead = 1.4 * (mtm + cds.notional * sf * mf)
    rwa = ead * counterparty_rw
    capital = rwa * 0.08

    # SIMM: CS01 into CSR bucket
    from pricebook.simm import SIMMCalculator, SIMMSensitivity
    rm = cds_risk_metrics(cds, discount_curve, survival_curve)
    simm_inputs = [SIMMSensitivity(
        risk_class="CSR", bucket="IG_corporate", tenor="5Y", delta=rm.cs01)]
    simm_im = SIMMCalculator().compute(simm_inputs).total_margin

    return CDSCapitalResult(ead=ead, rwa=rwa, capital=capital, simm_im=simm_im)


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class CDSHedgeRecommendation:
    risk_type: str
    current: float
    limit: float
    breach_pct: float
    action: str

    def to_dict(self) -> dict:
        return {
            "risk": self.risk_type, "current": self.current,
            "limit": self.limit, "breach_pct": self.breach_pct,
            "action": self.action,
        }


def cds_hedge_recommendations(
    book: CDSBook,
    curve: DiscountCurve,
    cs01_limit: float = 50_000,
    jtd_limit: float = 10_000_000,
    concentration_limit_pct: float = 0.25,
) -> list[CDSHedgeRecommendation]:
    """Hedge recommendations for CDS book."""
    risk = book.aggregate_risk(curve)
    recs = []

    checks = [
        ("cs01", abs(risk["total_cs01"]), cs01_limit,
         "Buy index protection to reduce portfolio CS01"),
        ("jtd", abs(risk["total_jtd"]), jtd_limit,
         "Reduce single-name concentration or buy FTD protection"),
    ]

    for risk_type, current, limit, action in checks:
        if limit > 0 and current > limit * 0.75:
            recs.append(CDSHedgeRecommendation(
                risk_type=risk_type, current=current, limit=limit,
                breach_pct=current / limit, action=action,
            ))

    # Concentration check per name
    by_name = book.by_name()
    total = risk["total_notional"]
    for name, entries in by_name.items():
        name_notional = sum(e.instrument.notional for e in entries if hasattr(e.instrument, 'notional'))
        if total > 0 and name_notional / total > concentration_limit_pct:
            recs.append(CDSHedgeRecommendation(
                risk_type="concentration", current=name_notional / total,
                limit=concentration_limit_pct,
                breach_pct=(name_notional / total) / concentration_limit_pct,
                action=f"Reduce {name} concentration — exceeds {concentration_limit_pct:.0%} of book",
            ))

    return recs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class CDSEventType:
    CREDIT_EVENT = "credit_event"
    RESTRUCTURING = "restructuring"
    SUCCESSION = "succession"
    MATURITY = "maturity"
    SETTLEMENT = "settlement"


class CDSLifecycle:
    """Lifecycle management for a CDS position."""

    def __init__(self, cds: CDS, survival_curve: SurvivalCurve,
                 trade_id: str = "", creation_date: date | None = None):
        from pricebook.trade import Trade
        from pricebook.trade_lifecycle import ManagedTrade

        self._cds = cds
        self._surv = survival_curve
        self._trade_id = trade_id
        trade = Trade(instrument=cds, trade_id=trade_id)
        self._managed = ManagedTrade(trade, creation_date or cds.start)
        self._events: list[dict] = []

    @property
    def cds(self) -> CDS:
        return self._cds

    @property
    def history(self) -> list[dict]:
        base = [
            {"type": e.event_type.value, "date": e.event_date.isoformat(),
             "version": e.version, **e.details}
            for e in self._managed.history
        ]
        return sorted(base + self._events, key=lambda x: x.get("date", ""))

    def credit_event(self, event_date: date, curve: DiscountCurve,
                     event_type: str = "default") -> float:
        """Process credit event. Returns protection payout = (1-R) × N."""
        payout = (1 - self._cds.recovery) * self._cds.notional
        self._events.append({
            "type": CDSEventType.CREDIT_EVENT,
            "date": event_date.isoformat(),
            "event_type": event_type,
            "payout": payout,
        })
        self._managed.exercise(event_date, underlying_instrument=self._cds)
        return payout

    def maturity_alert(self, as_of: date, alert_days: int = 30) -> dict | None:
        days_to_mat = (self._cds.end - as_of).days
        if 0 < days_to_mat <= alert_days:
            return {
                "type": "maturity_alert",
                "date": self._cds.end.isoformat(),
                "days_remaining": days_to_mat,
            }
        return None

    def record_succession(self, event_date: date, new_reference: str) -> dict:
        """Record succession event (reference entity changed)."""
        event = {
            "type": CDSEventType.SUCCESSION,
            "date": event_date.isoformat(),
            "new_reference": new_reference,
        }
        self._events.append(event)
        return event
