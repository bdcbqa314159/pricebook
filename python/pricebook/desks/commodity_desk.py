"""Commodity trading desk: unified dashboard, stress, hedge recs, lifecycle.

Consolidation layer wiring existing commodity modules (commodity_book,
commodity_daily_pnl, commodity_capital, commodity_rv, commodity_vol_surface)
into the standard desk pattern.

    from pricebook.desks.commodity_desk import (
        commodity_risk_metrics, CommodityRiskMetrics,
        commodity_dashboard, CommodityDashboard,
        commodity_stress_suite, commodity_hedge_recommendations,
        CommodityLifecycle,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
from pricebook.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class CommodityRiskMetrics:
    """Unified risk metrics for a commodity position."""
    pv: float
    delta: float               # dPV/dSpot
    dv01: float                # rate sensitivity
    vega: float                # vol sensitivity
    gamma: float               # d²PV/dS²
    theta: float               # time decay
    notional: float
    commodity: str

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "delta": self.delta, "dv01": self.dv01,
            "vega": self.vega, "gamma": self.gamma, "theta": self.theta,
            "notional": self.notional, "commodity": self.commodity,
        }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class CommodityDashboard:
    """Morning-meeting summary for the commodity desk."""
    date: date
    n_positions: int
    total_delta: float
    total_vega: float
    by_commodity: dict[str, int]
    by_sector: dict[str, int]
    limit_breaches: list[dict]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "delta": self.total_delta, "vega": self.total_vega,
            "by_commodity": self.by_commodity, "by_sector": self.by_sector,
            "breaches": self.limit_breaches,
        }


def commodity_dashboard(
    book,
    reference_date: date,
) -> CommodityDashboard:
    """Build commodity desk morning dashboard."""
    positions = book.positions_by_commodity()

    by_commodity = {p.commodity: p.trade_count for p in positions}
    total_delta = sum(abs(p.net_notional) for p in positions)

    # Sector aggregation
    sectors = {}
    try:
        for s in book.exposures_by_sector():
            sectors[s.sector] = getattr(s, 'n_positions', 1)
    except (AttributeError, TypeError):
        pass

    breaches = []
    try:
        for b in book.check_limits():
            breaches.append({
                "type": b.limit_type, "name": b.limit_name,
                "limit": b.limit_value, "actual": b.actual_value,
            })
    except (AttributeError, TypeError):
        pass

    return CommodityDashboard(
        date=reference_date,
        n_positions=sum(p.trade_count for p in positions),
        total_delta=total_delta, total_vega=0.0,
        by_commodity=by_commodity, by_sector=sectors,
        limit_breaches=breaches,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class CommodityStressResult:
    scenario: str
    description: str
    pnl: float

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "description": self.description, "pnl": self.pnl}


def commodity_stress_suite(
    total_delta: float,
    total_vega: float = 0.0,
) -> list[CommodityStressResult]:
    """Parametric commodity stress scenarios."""
    scenarios = [
        ("price_dn_10", "Prices -10%", -0.10 * total_delta),
        ("price_up_10", "Prices +10%", 0.10 * total_delta),
        ("price_dn_20", "Prices -20%", -0.20 * total_delta),
        ("price_up_20", "Prices +20%", 0.20 * total_delta),
        ("combined", "Prices -10%, vol +10%", -0.10 * total_delta + total_vega * 10),
    ]
    return [CommodityStressResult(n, d, p) for n, d, p in scenarios]


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class CommodityHedgeRecommendation:
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


def commodity_hedge_recommendations(
    book,
    delta_limit: float = 50_000_000,
    concentration_pct: float = 0.30,
) -> list[CommodityHedgeRecommendation]:
    """Hedge recommendations for commodity book."""
    recs = []
    positions = book.positions_by_commodity()

    total_delta = sum(abs(p.net_notional) for p in positions)
    total_notional = sum(abs(p.net_notional) for p in positions)

    if delta_limit > 0 and total_delta > delta_limit * 0.75:
        recs.append(CommodityHedgeRecommendation(
            "delta", total_delta, delta_limit, total_delta / delta_limit,
            "Reduce exposure via futures or swaps"))

    for p in positions:
        if total_notional > 0 and abs(p.net_notional) / total_notional > concentration_pct:
            recs.append(CommodityHedgeRecommendation(
                "concentration", abs(p.net_notional) / total_notional,
                concentration_pct,
                (abs(p.net_notional) / total_notional) / concentration_pct,
                f"Reduce {p.commodity} concentration"))

    return recs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class CommodityEventType:
    DELIVERY = "delivery"
    EXPIRY = "expiry"
    ROLL = "roll"
    NOMINATION = "nomination"


class CommodityLifecycle:
    """Lifecycle management for commodity positions."""

    def __init__(self, instrument, trade_id: str = ""):
        self._instrument = instrument
        self._trade_id = trade_id
        self._events: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return sorted(self._events, key=lambda x: x.get("date", ""))

    def delivery_alert(self, as_of: date, alert_days: int = 5) -> dict | None:
        expiry = getattr(self._instrument, 'end',
                        getattr(self._instrument, 'expiry', None))
        if expiry is None:
            return None
        days = (expiry - as_of).days
        if 0 < days <= alert_days:
            return {
                "type": CommodityEventType.DELIVERY,
                "date": expiry.isoformat(),
                "days_remaining": days,
            }
        return None

    def record_roll(self, roll_date: date, new_delivery: date, roll_cost: float = 0.0) -> dict:
        event = {
            "type": CommodityEventType.ROLL,
            "date": roll_date.isoformat(),
            "new_delivery": new_delivery.isoformat(),
            "roll_cost": roll_cost,
        }
        self._events.append(event)
        return event

    def record_nomination(self, nom_date: date, volume: float) -> dict:
        event = {
            "type": CommodityEventType.NOMINATION,
            "date": nom_date.isoformat(),
            "volume": volume,
        }
        self._events.append(event)
        return event


# ---------------------------------------------------------------------------
# Carry decomposition (protocol compliance)
# ---------------------------------------------------------------------------

@dataclass
class CommodityCarryDecomposition:
    """Commodity carry: roll yield + convenience - storage."""
    roll_yield: float
    convenience_yield: float
    storage_cost: float
    net_carry: float

    def to_dict(self) -> dict:
        return {"roll": self.roll_yield, "convenience": self.convenience_yield,
                "storage": self.storage_cost, "net": self.net_carry}


# ---------------------------------------------------------------------------
# Daily P&L (protocol compliance)
# ---------------------------------------------------------------------------

@dataclass
class CommodityDailyPnL:
    """Commodity daily P&L attribution."""
    date: date
    total: float
    spot_pnl: float
    carry_pnl: float
    roll_pnl: float
    unexplained: float

    def to_dict(self) -> dict:
        return {"date": self.date.isoformat(), "total": self.total,
                "spot": self.spot_pnl, "carry": self.carry_pnl,
                "roll": self.roll_pnl, "unexplained": self.unexplained}
