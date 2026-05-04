"""Swaption trading desk: book, dashboard, stress, XVA, hedge, lifecycle.

Extends the existing swaption_desk.py (VolCube, combos, delta/vega hedging)
with the standard desk consolidation layer: book management, dashboard,
stress testing, XVA, hedge recommendations, and lifecycle.

    from pricebook.swaption_trading_desk import (
        swaption_risk_metrics, SwaptionRiskMetrics,
        SwaptionBook, SwaptionBookEntry,
        swaption_dashboard, SwaptionDashboard,
        swaption_stress_suite, swaption_scenario_stress,
        swaption_capital, SwaptionCapitalResult,
        swaption_hedge_recommendations,
        SwaptionLifecycle,
    )

Vol operations (VolCube, SABR, combos) remain in swaption_desk.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.swaption import Swaption, SwaptionType
from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
from pricebook.day_count import DayCountConvention, year_fraction


class _FlatVol:
    """Minimal vol surface wrapper for flat vol input."""
    def __init__(self, v: float):
        self._v = v
    def vol(self, *args, **kwargs) -> float:
        return self._v


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class SwaptionRiskMetrics:
    """Unified risk metrics for a swaption position."""
    pv: float
    delta: float               # dPV/dF (forward swap rate)
    gamma: float               # d²PV/dF²
    vega: float                # dPV/dσ (per 1% vol move)
    theta: float               # time decay
    forward_rate: float
    annuity: float
    notional: float
    swaption_type: str
    expiry: date
    swap_tenor: str            # e.g. "5Y"

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "delta": self.delta, "gamma": self.gamma,
            "vega": self.vega, "theta": self.theta,
            "forward": self.forward_rate, "annuity": self.annuity,
            "notional": self.notional, "type": self.swaption_type,
            "expiry": self.expiry.isoformat(), "tenor": self.swap_tenor,
        }


def swaption_risk_metrics(
    swaption: Swaption,
    curve: DiscountCurve,
    vol: float,
    projection_curve: DiscountCurve | None = None,
) -> SwaptionRiskMetrics:
    """Compute unified risk metrics for a swaption."""
    proj = projection_curve or curve
    vol_surf = _FlatVol(vol) if isinstance(vol, (int, float)) else vol
    pv = swaption.pv(curve, vol_surf, proj)
    fwd = swaption.forward_swap_rate(curve, proj)
    ann = swaption.annuity(curve)

    # Greeks from instrument
    g = swaption.greeks(curve, vol_surf, proj)

    # Swap tenor label
    T_swap = year_fraction(swaption.expiry, swaption.swap_end, DayCountConvention.ACT_365_FIXED)
    tenor = f"{int(round(T_swap))}Y"

    # Theta via rolldown
    from pricebook.pnl_explain import compute_rolldown
    theta = compute_rolldown(lambda c: swaption.pv(c, vol_surf, proj), curve, days=1)

    return SwaptionRiskMetrics(
        pv=pv, delta=g.delta, gamma=g.gamma, vega=g.vega,
        theta=theta, forward_rate=fwd, annuity=ann,
        notional=swaption.notional, swaption_type=swaption.swaption_type.value,
        expiry=swaption.expiry, swap_tenor=tenor,
    )


# ---------------------------------------------------------------------------
# Book
# ---------------------------------------------------------------------------

@dataclass
class SwaptionBookEntry:
    """A swaption position in the book."""
    trade_id: str
    swaption: Swaption
    vol: float
    counterparty: str = ""


class SwaptionBook:
    """Collection of swaption positions."""

    def __init__(self, name: str = "swaption_book"):
        self.name = name
        self._entries: list[SwaptionBookEntry] = []

    def add(self, entry: SwaptionBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[SwaptionBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_notional(self) -> float:
        return sum(e.swaption.notional for e in self._entries)

    def by_type(self) -> dict[str, list[SwaptionBookEntry]]:
        result: dict[str, list[SwaptionBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.swaption.swaption_type.value, []).append(e)
        return result

    def by_counterparty(self) -> dict[str, list[SwaptionBookEntry]]:
        result: dict[str, list[SwaptionBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.counterparty, []).append(e)
        return result

    def aggregate_risk(
        self, curve: DiscountCurve, projection: DiscountCurve | None = None,
    ) -> dict[str, float]:
        total_pv = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0

        for e in self._entries:
            rm = swaption_risk_metrics(e.swaption, curve, e.vol, projection)
            total_pv += rm.pv
            total_delta += rm.delta
            total_gamma += rm.gamma
            total_vega += rm.vega

        return {
            "total_pv": total_pv,
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_vega": total_vega,
            "n_positions": len(self._entries),
            "total_notional": self.total_notional(),
        }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class SwaptionDashboard:
    """Morning-meeting summary for the swaption desk."""
    date: date
    n_positions: int
    total_notional: float
    total_pv: float
    total_delta: float
    total_gamma: float
    total_vega: float
    by_type: dict[str, int]
    vega_ladder: dict[str, float]  # per expiry×tenor bucket

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "notional": self.total_notional, "pv": self.total_pv,
            "delta": self.total_delta, "gamma": self.total_gamma,
            "vega": self.total_vega, "by_type": self.by_type,
            "vega_ladder": self.vega_ladder,
        }


def swaption_dashboard(
    book: SwaptionBook,
    reference_date: date,
    curve: DiscountCurve,
    projection: DiscountCurve | None = None,
) -> SwaptionDashboard:
    """Build swaption desk morning dashboard with vega ladder."""
    risk = book.aggregate_risk(curve, projection)
    by_type = {k: len(v) for k, v in book.by_type().items()}

    # Vega ladder by expiry×tenor
    vega_ladder: dict[str, float] = {}
    for e in book.entries:
        rm = swaption_risk_metrics(e.swaption, curve, e.vol, projection)
        T_exp = year_fraction(reference_date, e.swaption.expiry, DayCountConvention.ACT_365_FIXED)
        T_swap = year_fraction(e.swaption.expiry, e.swaption.swap_end, DayCountConvention.ACT_365_FIXED)
        label = f"{int(round(T_exp))}Y×{int(round(T_swap))}Y"
        vega_ladder[label] = vega_ladder.get(label, 0.0) + rm.vega

    return SwaptionDashboard(
        date=reference_date, n_positions=risk["n_positions"],
        total_notional=risk["total_notional"], total_pv=risk["total_pv"],
        total_delta=risk["total_delta"], total_gamma=risk["total_gamma"],
        total_vega=risk["total_vega"], by_type=by_type,
        vega_ladder=vega_ladder,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class SwaptionStressResult:
    scenario: str
    description: str
    pnl: float

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "description": self.description, "pnl": self.pnl}


def swaption_stress_suite(
    book: SwaptionBook,
    curve: DiscountCurve,
    projection: DiscountCurve | None = None,
) -> list[SwaptionStressResult]:
    """Stress scenarios for swaption book."""
    risk = book.aggregate_risk(curve, projection)
    delta = risk["total_delta"]
    gamma = risk["total_gamma"]
    vega = risk["total_vega"]

    scenarios = [
        ("rates_up_100", "Rates +100bp", delta * 0.01 + 0.5 * gamma * 0.01**2),
        ("rates_dn_100", "Rates -100bp", delta * -0.01 + 0.5 * gamma * 0.01**2),
        ("vol_up_5", "Vol +5%", vega * 5),
        ("vol_dn_5", "Vol -5%", vega * -5),
        ("combined", "Rates +50bp, vol +3%", delta * 0.005 + vega * 3),
    ]
    return [SwaptionStressResult(n, d, p) for n, d, p in scenarios]


def swaption_scenario_stress(
    book: SwaptionBook,
    ctx: PricingContext,
    scenarios: list | None = None,
) -> list:
    """Full-reprice swaption stress via scenario.py."""
    from pricebook.scenario import parallel_shift, run_scenarios
    from pricebook.trade import Trade, Portfolio

    portfolio = Portfolio(name=book.name)
    for e in book.entries:
        portfolio.add(Trade(instrument=e.swaption, trade_id=e.trade_id))

    if scenarios is None:
        scenarios = [
            parallel_shift(0.01, "rates_+100bp"),
            parallel_shift(-0.01, "rates_-100bp"),
        ]

    return run_scenarios(portfolio, ctx, scenarios)


# ---------------------------------------------------------------------------
# Capital
# ---------------------------------------------------------------------------

@dataclass
class SwaptionCapitalResult:
    ead: float
    rwa: float
    capital: float
    simm_im: float

    def to_dict(self) -> dict:
        return {"ead": self.ead, "rwa": self.rwa, "capital": self.capital, "simm_im": self.simm_im}


def swaption_capital(
    swaption: Swaption,
    curve: DiscountCurve,
    vol: float,
    projection: DiscountCurve | None = None,
    counterparty_rw: float = 0.20,
) -> SwaptionCapitalResult:
    """SA-CCR capital for a swaption. SF=0.005 for IR."""
    T = year_fraction(swaption.expiry, swaption.swap_end, DayCountConvention.ACT_365_FIXED)
    vol_surf = _FlatVol(vol) if isinstance(vol, (int, float)) else vol
    pv = swaption.pv(curve, vol_surf, projection)
    mtm = max(pv, 0)

    sf = 0.005
    mf = math.sqrt(min(T, 1.0))
    ead = 1.4 * (mtm + swaption.notional * sf * mf)
    rwa = ead * counterparty_rw
    capital = rwa * 0.08

    # SIMM: vega into GIRR bucket
    from pricebook.simm import SIMMCalculator, SIMMSensitivity
    g = swaption.greeks(curve, vol_surf, projection)
    simm_inputs = [
        SIMMSensitivity(risk_class="GIRR", bucket="USD", tenor="5Y", delta=0, vega=g.vega),
    ]
    simm_im = SIMMCalculator().compute(simm_inputs).total_margin

    return SwaptionCapitalResult(ead=ead, rwa=rwa, capital=capital, simm_im=simm_im)


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class SwaptionHedgeRecommendation:
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


def swaption_hedge_recommendations(
    book: SwaptionBook,
    curve: DiscountCurve,
    projection: DiscountCurve | None = None,
    delta_limit: float = 50_000,
    gamma_limit: float = 1_000_000,
    vega_limit: float = 500_000,
) -> list[SwaptionHedgeRecommendation]:
    """Hedge recommendations for swaption book."""
    risk = book.aggregate_risk(curve, projection)
    recs = []

    checks = [
        ("delta", abs(risk["total_delta"]), delta_limit,
         "Delta-hedge via underlying swaps"),
        ("gamma", abs(risk["total_gamma"]), gamma_limit,
         "Buy/sell swaptions to flatten gamma"),
        ("vega", abs(risk["total_vega"]), vega_limit,
         "Trade straddles/strangles to reduce vega exposure"),
    ]

    for risk_type, current, limit, action in checks:
        if limit > 0 and current > limit * 0.75:
            recs.append(SwaptionHedgeRecommendation(
                risk_type=risk_type, current=current, limit=limit,
                breach_pct=current / limit, action=action,
            ))

    return recs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class SwaptionEventType:
    EXPIRY = "expiry"
    EXERCISE = "exercise"
    LAPSE = "lapse"
    NOVATION = "novation"


class SwaptionLifecycle:
    """Lifecycle management for swaption positions."""

    def __init__(self, swaption: Swaption, trade_id: str = ""):
        self._swaption = swaption
        self._trade_id = trade_id
        self._events: list[dict] = []

    @property
    def swaption(self) -> Swaption:
        return self._swaption

    @property
    def history(self) -> list[dict]:
        return sorted(self._events, key=lambda x: x.get("date", ""))

    def expiry_alert(self, as_of: date, alert_days: int = 5) -> dict | None:
        days = (self._swaption.expiry - as_of).days
        if 0 < days <= alert_days:
            return {
                "type": SwaptionEventType.EXPIRY,
                "date": self._swaption.expiry.isoformat(),
                "days_remaining": days,
                "type_": self._swaption.swaption_type.value,
            }
        return None

    def record_exercise(self, exercise_date: date, swap_rate: float) -> dict:
        """Record exercise into physical swap."""
        event = {
            "type": SwaptionEventType.EXERCISE,
            "date": exercise_date.isoformat(),
            "swap_rate": swap_rate,
            "swap_end": self._swaption.swap_end.isoformat(),
        }
        self._events.append(event)
        return event

    def record_lapse(self, expiry_date: date) -> dict:
        """Record lapse (option expires unexercised)."""
        event = {
            "type": SwaptionEventType.LAPSE,
            "date": expiry_date.isoformat(),
        }
        self._events.append(event)
        return event


# ---------------------------------------------------------------------------
# Carry decomposition (protocol compliance)
# ---------------------------------------------------------------------------

@dataclass
class SwaptionCarryDecomposition:
    """Swaption carry: theta decay is the primary carry component."""
    theta_decay: float
    net_carry: float

    def to_dict(self) -> dict:
        return {"theta": self.theta_decay, "net": self.net_carry}


# ---------------------------------------------------------------------------
# Daily P&L (protocol compliance)
# ---------------------------------------------------------------------------

@dataclass
class SwaptionDailyPnL:
    """Swaption daily P&L attribution."""
    date: date
    total: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    unexplained: float

    def to_dict(self) -> dict:
        return {"date": self.date.isoformat(), "total": self.total,
                "delta": self.delta_pnl, "gamma": self.gamma_pnl,
                "vega": self.vega_pnl, "theta": self.theta_pnl,
                "unexplained": self.unexplained}
