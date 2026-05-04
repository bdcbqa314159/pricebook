"""Equity trading desk: unified dashboard, stress, XVA, hedge recs, lifecycle.

Consolidation layer wiring existing equity modules (equity_book, equity_daily_pnl,
equity_capital, equity_vol_desk, options_book) into the standard desk pattern.

    from pricebook.equity_desk import (
        equity_risk_metrics, EquityRiskMetrics,
        equity_dashboard, EquityDashboard,
        equity_stress_suite, equity_scenario_stress,
        equity_hedge_recommendations, EquityHedgeRecommendation,
        EquityLifecycle,
    )

The equity pricing engine (25+ modules) is already production-grade.
This module adds the desk-level consolidation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
from pricebook.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class EquityRiskMetrics:
    """Unified risk metrics for an equity position."""
    pv: float
    delta: float               # dPV/dS (delta-adjusted exposure)
    gamma: float               # d²PV/dS²
    vega: float                # dPV/dσ
    theta: float               # dPV/dt
    rho: float                 # dPV/dr
    notional: float
    ticker: str

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "delta": self.delta, "gamma": self.gamma,
            "vega": self.vega, "theta": self.theta, "rho": self.rho,
            "notional": self.notional, "ticker": self.ticker,
        }


def equity_risk_metrics(
    instrument,
    curve: DiscountCurve,
    spot: float = 0.0,
    vol: float = 0.20,
    ticker: str = "",
) -> EquityRiskMetrics:
    """Compute unified risk metrics for any equity instrument."""
    # PV
    pv = 0.0
    if hasattr(instrument, 'pv_ctx'):
        try:
            ctx = PricingContext(valuation_date=curve.reference_date, discount_curve=curve)
            pv = instrument.pv_ctx(ctx)
        except (TypeError, AttributeError, KeyError):
            pass

    # Greeks via bump-and-reprice or instrument methods
    delta = gamma = vega = theta = rho = 0.0

    if hasattr(instrument, 'greeks'):
        try:
            g = instrument.greeks(curve) if not hasattr(instrument, 'spot') else {}
            delta = g.get("delta", 0.0)
            gamma = g.get("gamma", 0.0)
            vega = g.get("vega", 0.0)
            theta = g.get("theta", 0.0)
            rho = g.get("rho", 0.0)
        except (TypeError, AttributeError):
            pass

    notional = getattr(instrument, 'notional',
                       getattr(instrument, 'face_value', 0))

    return EquityRiskMetrics(
        pv=pv, delta=delta, gamma=gamma,
        vega=vega, theta=theta, rho=rho,
        notional=notional, ticker=ticker,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class EquityDashboard:
    """Morning-meeting summary for the equity desk."""
    date: date
    n_positions: int
    total_delta: float
    total_gamma: float
    total_vega: float
    total_theta: float
    by_sector: dict[str, int]
    by_ticker: dict[str, int]
    limit_breaches: list[dict]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "delta": self.total_delta, "gamma": self.total_gamma,
            "vega": self.total_vega, "theta": self.total_theta,
            "by_sector": self.by_sector, "by_ticker": self.by_ticker,
            "breaches": self.limit_breaches,
        }


def equity_dashboard(
    book,
    reference_date: date,
) -> EquityDashboard:
    """Build equity desk morning dashboard from EquityBook."""
    positions = book.positions_by_ticker()

    by_ticker = {p.ticker: p.trade_count for p in positions}
    # Aggregate by sector from position data
    sector_counts: dict[str, int] = {}
    for p in positions:
        sector_counts[p.sector] = sector_counts.get(p.sector, 0) + 1
    by_sector = sector_counts
    total_delta = sum(p.delta_exposure for p in positions)

    breaches = []
    try:
        for b in book.check_limits():
            breaches.append({
                "type": b.limit_type, "name": b.limit_name,
                "limit": b.limit_value, "actual": b.actual_value,
            })
    except (AttributeError, TypeError):
        pass

    return EquityDashboard(
        date=reference_date,
        n_positions=sum(p.trade_count for p in positions),
        total_delta=total_delta,
        total_gamma=0.0,  # would need per-position Greeks
        total_vega=0.0,
        total_theta=0.0,
        by_sector=by_sector, by_ticker=by_ticker,
        limit_breaches=breaches,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class EquityStressResult:
    scenario: str
    description: str
    pnl: float

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "description": self.description, "pnl": self.pnl}


def equity_stress_suite(
    total_delta: float,
    total_gamma: float = 0.0,
    total_vega: float = 0.0,
) -> list[EquityStressResult]:
    """Parametric equity stress scenarios using Greek approximation."""
    scenarios = [
        ("spot_dn_10", "Spot -10%", -0.10),
        ("spot_up_10", "Spot +10%", 0.10),
        ("spot_dn_20", "Spot -20%", -0.20),
        ("spot_up_20", "Spot +20%", 0.20),
        ("combined", "Spot -10%, vol +5%", -0.10),
    ]

    results = []
    for name, desc, shock in scenarios:
        # Delta P&L + 0.5 × gamma × shock²
        pnl = total_delta * shock + 0.5 * total_gamma * shock ** 2
        if name == "combined":
            pnl += total_vega * 5.0  # +5 vol points
        results.append(EquityStressResult(name, desc, pnl))

    return results


def equity_scenario_stress(
    book,
    ctx: PricingContext,
    scenarios: list | None = None,
) -> list:
    """Full-reprice equity stress via scenario.py."""
    from pricebook.scenario import parallel_shift, run_scenarios
    from pricebook.trade import Portfolio

    portfolio = Portfolio(name="equity_stress")
    for entry in book._entries:
        portfolio.add(entry.trade)

    if scenarios is None:
        scenarios = [
            parallel_shift(0.01, "rates_+100bp"),
            parallel_shift(-0.01, "rates_-100bp"),
        ]

    return run_scenarios(portfolio, ctx, scenarios)


# ---------------------------------------------------------------------------
# XVA
# ---------------------------------------------------------------------------

def equity_mc_xva(
    instrument,
    ctx: PricingContext,
    cpty_survival,
    own_survival,
    funding_spread: float = 0.005,
    n_paths: int = 1000,
    n_steps: int = 12,
    rate_vol: float = 0.01,
    seed: int = 42,
):
    """MC XVA for equity option — wires xva.simulate_exposures."""
    import numpy as np
    from pricebook.xva import (
        simulate_exposures, expected_positive_exposure,
        expected_negative_exposure, total_xva_decomposition,
    )

    expiry = getattr(instrument, 'expiry',
                     getattr(instrument, 'maturity', None))
    if expiry is None:
        return None

    T = year_fraction(ctx.valuation_date, expiry, DayCountConvention.ACT_365_FIXED)
    if T <= 0:
        return None

    time_grid = [(i + 1) * T / n_steps for i in range(n_steps)]
    pricer = lambda c: instrument.pv_ctx(c) if hasattr(instrument, 'pv_ctx') else 0.0

    pvs = simulate_exposures(pricer, ctx, time_grid, n_paths, rate_vol, seed)
    epe = expected_positive_exposure(pvs)
    ene = expected_negative_exposure(pvs)

    return total_xva_decomposition(
        epe=epe, ene=ene, time_grid=time_grid,
        discount_curve=ctx.discount_curve,
        cpty_survival=cpty_survival, own_survival=own_survival,
        funding_spread=funding_spread,
    )


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class EquityHedgeRecommendation:
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


def equity_hedge_recommendations(
    book,
    delta_limit: float = 50_000_000,
    gamma_limit: float = 1_000_000,
    vega_limit: float = 500_000,
    concentration_pct: float = 0.25,
) -> list[EquityHedgeRecommendation]:
    """Hedge recommendations for equity book."""
    recs = []
    positions = book.positions_by_ticker()

    total_delta = sum(abs(p.delta_exposure) for p in positions)
    total_notional = sum(abs(p.net_notional) for p in positions)

    if delta_limit > 0 and total_delta > delta_limit * 0.75:
        recs.append(EquityHedgeRecommendation(
            "delta", total_delta, delta_limit, total_delta / delta_limit,
            "Reduce delta via futures or offsetting options"))

    # Concentration
    for p in positions:
        if total_notional > 0 and abs(p.net_notional) / total_notional > concentration_pct:
            recs.append(EquityHedgeRecommendation(
                "concentration", abs(p.net_notional) / total_notional,
                concentration_pct,
                (abs(p.net_notional) / total_notional) / concentration_pct,
                f"Reduce {p.ticker} concentration — exceeds {concentration_pct:.0%}"))

    return recs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class EquityEventType:
    EXPIRY = "expiry"
    EXERCISE = "exercise"
    ASSIGNMENT = "assignment"
    DIVIDEND_EX = "dividend_ex"


class EquityLifecycle:
    """Lifecycle management for equity positions."""

    def __init__(self, instrument, trade_id: str = ""):
        self._instrument = instrument
        self._trade_id = trade_id
        self._events: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return sorted(self._events, key=lambda x: x.get("date", ""))

    def expiry_alert(self, as_of: date, alert_days: int = 5) -> dict | None:
        expiry = getattr(self._instrument, 'expiry', None)
        if expiry is None:
            return None
        days = (expiry - as_of).days
        if 0 < days <= alert_days:
            return {
                "type": EquityEventType.EXPIRY,
                "date": expiry.isoformat(),
                "days_remaining": days,
            }
        return None

    def record_exercise(self, exercise_date: date, exercise_price: float) -> dict:
        event = {
            "type": EquityEventType.EXERCISE,
            "date": exercise_date.isoformat(),
            "price": exercise_price,
        }
        self._events.append(event)
        return event

    def record_dividend_ex(self, ex_date: date, amount: float) -> dict:
        event = {
            "type": EquityEventType.DIVIDEND_EX,
            "date": ex_date.isoformat(),
            "amount": amount,
        }
        self._events.append(event)
        return event
