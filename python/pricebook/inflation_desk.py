"""Inflation trading desk: unified dashboard, stress, hedge recs, lifecycle.

Consolidation layer wiring existing inflation modules (inflation_book,
inflation_trading, inflation_carry, inflation_basis, inflation_smile)
into the standard desk pattern.

    from pricebook.inflation_desk import (
        inflation_dashboard, InflationDashboard,
        inflation_stress_suite, inflation_hedge_recommendations,
        InflationLifecycle,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class InflationRiskMetrics:
    """Unified risk metrics for an inflation position."""
    pv: float
    ie01: float                # inflation expectation sensitivity (1bp breakeven)
    real_dv01: float           # real rate sensitivity
    nominal_dv01: float        # nominal rate sensitivity
    notional: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "ie01": self.ie01,
            "real_dv01": self.real_dv01, "nominal_dv01": self.nominal_dv01,
            "notional": self.notional,
        }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class InflationDashboard:
    """Morning-meeting summary for the inflation desk."""
    date: date
    n_positions: int
    total_ie01: float
    total_real_dv01: float
    by_type: dict[str, int]   # linker, ZC swap, YoY swap, cap, floor
    limit_breaches: list[dict]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "ie01": self.total_ie01, "real_dv01": self.total_real_dv01,
            "by_type": self.by_type, "breaches": self.limit_breaches,
        }


def inflation_dashboard(
    book,
    reference_date: date,
) -> InflationDashboard:
    """Build inflation desk morning dashboard."""
    positions = book.positions()

    by_type: dict[str, int] = {}
    total_ie01 = 0.0
    total_real = 0.0

    for p in positions:
        ptype = getattr(p, 'instrument_type', 'other')
        by_type[ptype] = by_type.get(ptype, 0) + 1
        total_ie01 += getattr(p, 'ie01', 0.0)
        total_real += getattr(p, 'real_dv01', 0.0)

    breaches = []
    try:
        for b in book.check_limits():
            breaches.append({
                "type": getattr(b, 'limit_type', ''),
                "name": getattr(b, 'limit_name', ''),
                "limit": getattr(b, 'limit_value', 0),
                "actual": getattr(b, 'actual_value', 0),
            })
    except (AttributeError, TypeError):
        pass

    return InflationDashboard(
        date=reference_date,
        n_positions=len(positions),
        total_ie01=total_ie01, total_real_dv01=total_real,
        by_type=by_type, limit_breaches=breaches,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class InflationStressResult:
    scenario: str
    description: str
    pnl: float

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "description": self.description, "pnl": self.pnl}


def inflation_stress_suite(
    total_ie01: float,
    total_real_dv01: float = 0.0,
    total_nominal_dv01: float = 0.0,
) -> list[InflationStressResult]:
    """Parametric inflation stress scenarios."""
    scenarios = [
        ("breakeven_up_50", "Breakeven +50bp", total_ie01 * 50),
        ("breakeven_dn_50", "Breakeven -50bp", total_ie01 * -50),
        ("real_up_100", "Real rates +100bp", total_real_dv01 * 100),
        ("nominal_up_100", "Nominal rates +100bp", total_nominal_dv01 * 100),
        ("combined", "Breakeven -25bp, real +50bp",
         total_ie01 * -25 + total_real_dv01 * 50),
    ]
    return [InflationStressResult(n, d, p) for n, d, p in scenarios]


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class InflationHedgeRecommendation:
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


def inflation_hedge_recommendations(
    total_ie01: float,
    total_real_dv01: float = 0.0,
    ie01_limit: float = 50_000,
    real_dv01_limit: float = 50_000,
) -> list[InflationHedgeRecommendation]:
    """Hedge recommendations for inflation book."""
    recs = []

    if ie01_limit > 0 and abs(total_ie01) > ie01_limit * 0.75:
        recs.append(InflationHedgeRecommendation(
            "ie01", abs(total_ie01), ie01_limit, abs(total_ie01) / ie01_limit,
            "Reduce breakeven exposure via ZC inflation swap or linker"))

    if real_dv01_limit > 0 and abs(total_real_dv01) > real_dv01_limit * 0.75:
        recs.append(InflationHedgeRecommendation(
            "real_dv01", abs(total_real_dv01), real_dv01_limit,
            abs(total_real_dv01) / real_dv01_limit,
            "Hedge real rate risk via real rate swap or duration match"))

    return recs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class InflationEventType:
    CPI_FIXING = "cpi_fixing"
    COUPON = "coupon"
    MATURITY = "maturity"
    RESET = "reset"


class InflationLifecycle:
    """Lifecycle management for inflation positions."""

    def __init__(self, instrument, trade_id: str = ""):
        self._instrument = instrument
        self._trade_id = trade_id
        self._events: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return sorted(self._events, key=lambda x: x.get("date", ""))

    def fixing_alert(self, as_of: date, alert_days: int = 5) -> dict | None:
        """Alert for upcoming CPI fixing (typically 2-3 month lag)."""
        # CPI is published with a lag — next fixing date is roughly as_of + 2 months
        next_fixing = as_of + timedelta(days=60)
        if (next_fixing - as_of).days <= alert_days + 60:
            return {
                "type": InflationEventType.CPI_FIXING,
                "date": next_fixing.isoformat(),
                "note": "CPI publication expected — check for seasonal adjustment",
            }
        return None

    def maturity_alert(self, as_of: date, alert_days: int = 30) -> dict | None:
        maturity = getattr(self._instrument, 'end',
                          getattr(self._instrument, 'maturity', None))
        if maturity is None:
            return None
        days = (maturity - as_of).days
        if 0 < days <= alert_days:
            return {
                "type": InflationEventType.MATURITY,
                "date": maturity.isoformat(),
                "days_remaining": days,
            }
        return None

    def record_cpi_fixing(self, fixing_date: date, cpi_level: float) -> dict:
        event = {
            "type": InflationEventType.CPI_FIXING,
            "date": fixing_date.isoformat(),
            "cpi": cpi_level,
        }
        self._events.append(event)
        return event


# ---------------------------------------------------------------------------
# Carry decomposition (protocol compliance)
# ---------------------------------------------------------------------------

@dataclass
class InflationCarryDecomposition:
    """Inflation carry: real yield + breakeven accrual."""
    real_yield_carry: float
    breakeven_accrual: float
    net_carry: float

    def to_dict(self) -> dict:
        return {"real_yield": self.real_yield_carry,
                "breakeven": self.breakeven_accrual, "net": self.net_carry}


# ---------------------------------------------------------------------------
# Daily P&L (protocol compliance)
# ---------------------------------------------------------------------------

@dataclass
class InflationDailyPnL:
    """Inflation daily P&L attribution."""
    date: date
    total: float
    breakeven_pnl: float
    real_rate_pnl: float
    carry_pnl: float
    unexplained: float

    def to_dict(self) -> dict:
        return {"date": self.date.isoformat(), "total": self.total,
                "breakeven": self.breakeven_pnl, "real_rate": self.real_rate_pnl,
                "carry": self.carry_pnl, "unexplained": self.unexplained}


# ---------------------------------------------------------------------------
# Capital (protocol compliance)
# ---------------------------------------------------------------------------

@dataclass
class InflationCapitalResult:
    """SA-CCR capital for inflation position."""
    ead: float
    rwa: float
    capital: float
    simm_im: float

    def to_dict(self) -> dict:
        return {"ead": self.ead, "rwa": self.rwa, "capital": self.capital,
                "simm_im": self.simm_im}
