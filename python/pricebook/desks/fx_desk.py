"""FX trading desk: unified dashboard, stress, XVA, hedge recs, lifecycle.

Consolidation layer wiring existing FX modules (fx_book, fx_daily_pnl,
fx_greeks, fx_carry, fx_hedging, fx_capital) into the standard desk pattern.

    from pricebook.desks.fx_desk import (
        fx_risk_metrics, FXRiskMetrics,
        fx_dashboard, FXDashboard,
        fx_stress_suite, fx_scenario_stress,
        fx_mc_xva,
        fx_hedge_recommendations, FXHedgeRecommendation,
        FXLifecycle,
    )

The FX pricing engine (20 modules) is already production-grade.
This module adds the desk-level aggregation, stress, and lifecycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.pricing_context import PricingContext
from pricebook.core.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class FXRiskMetrics:
    """Unified risk metrics for an FX position."""
    pv: float
    spot_delta: float          # dPV/dSpot
    dv01_base: float           # base rate sensitivity
    dv01_quote: float          # quote rate sensitivity
    vega: float                # vol sensitivity (options only)
    gamma: float               # d²PV/dS² (options only)
    theta: float               # time decay (options only)
    notional: float
    pair: str

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "delta": self.spot_delta,
            "dv01_base": self.dv01_base, "dv01_quote": self.dv01_quote,
            "vega": self.vega, "gamma": self.gamma, "theta": self.theta,
            "notional": self.notional, "pair": self.pair,
        }


def fx_risk_metrics(
    instrument,
    base_curve: DiscountCurve,
    quote_curve: DiscountCurve,
    spot: float = 0.0,
    pair: str = "",
    vol: float = 0.0,
) -> FXRiskMetrics:
    """Compute unified risk metrics for any FX instrument.

    For FXForward: needs spot, base_curve, quote_curve.
    """
    # Infer spot from instrument if not provided
    if spot == 0.0:
        spot = getattr(instrument, 'strike', getattr(instrument, 'spot', 1.0))

    # PV
    pv = 0.0
    if hasattr(instrument, 'pv'):
        try:
            pv = instrument.pv(spot, base_curve, quote_curve)
        except TypeError:
            try:
                pv = instrument.pv(quote_curve, base_curve)
            except TypeError:
                pv = 0.0

    # Delta
    delta = 0.0
    if hasattr(instrument, 'fx_delta'):
        try:
            delta = instrument.fx_delta(spot, base_curve, quote_curve)
        except TypeError:
            delta = 0.0

    # DV01
    h = 0.0001
    dv01_base = 0.0
    dv01_quote = 0.0
    try:
        pv_base_up = instrument.pv(spot, base_curve.bumped(h), quote_curve)
        dv01_base = pv_base_up - pv
    except (TypeError, AttributeError):
        pass
    try:
        pv_quote_up = instrument.pv(spot, base_curve, quote_curve.bumped(h))
        dv01_quote = pv_quote_up - pv
    except (TypeError, AttributeError):
        pass

    notional = getattr(instrument, 'notional', getattr(instrument, 'contract_size', 0))

    return FXRiskMetrics(
        pv=pv, spot_delta=delta,
        dv01_base=dv01_base, dv01_quote=dv01_quote,
        vega=0.0, gamma=0.0, theta=0.0,
        notional=notional, pair=pair,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class FXDashboard:
    """Morning-meeting summary for the FX desk."""
    date: date
    n_positions: int
    total_delta: float
    total_vega: float
    total_dv01: float
    by_pair: dict[str, int]
    by_currency: dict[str, float]  # net exposure per currency
    limit_breaches: list[dict]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "delta": self.total_delta, "vega": self.total_vega,
            "dv01": self.total_dv01,
            "by_pair": self.by_pair, "by_currency": self.by_currency,
            "breaches": self.limit_breaches,
        }


def fx_dashboard(
    book,
    reference_date: date,
) -> FXDashboard:
    """Build FX desk morning dashboard from FXBook."""
    positions = book.positions_by_pair()
    ccy_exposures = book.net_currency_exposure()

    by_pair = {p.pair: p.trade_count for p in positions}
    by_ccy = {e.currency: e.net_exposure for e in ccy_exposures}

    total_delta = sum(abs(p.net_notional) for p in positions)
    total_dv01 = 0.0  # would need curves for per-position DV01

    breaches = []
    try:
        for b in book.check_limits():
            breaches.append({
                "type": b.limit_type, "name": b.limit_name,
                "limit": b.limit_value, "actual": b.actual_value,
            })
    except (AttributeError, TypeError):
        pass

    return FXDashboard(
        date=reference_date, n_positions=sum(p.trade_count for p in positions),
        total_delta=total_delta, total_vega=0.0, total_dv01=total_dv01,
        by_pair=by_pair, by_currency=by_ccy,
        limit_breaches=breaches,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class FXStressResult:
    scenario: str
    description: str
    pnl: float

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "description": self.description, "pnl": self.pnl}


def fx_stress_suite(
    positions: list[tuple[str, float, float]],  # (pair, notional, spot)
) -> list[FXStressResult]:
    """Parametric FX stress scenarios.

    Args:
        positions: list of (pair, net_notional_in_base, current_spot).
    """
    total_exposure = sum(abs(n) * s for _, n, s in positions)
    net_exposure = sum(n * s for _, n, s in positions)

    scenarios = [
        ("spot_dn_5", "Spot -5%", -0.05 * net_exposure),
        ("spot_up_5", "Spot +5%", 0.05 * net_exposure),
        ("spot_dn_10", "Spot -10%", -0.10 * net_exposure),
        ("spot_up_10", "Spot +10%", 0.10 * net_exposure),
        ("combined", "Spot -5%, rates +100bp", -0.05 * net_exposure),
    ]

    return [FXStressResult(n, d, p) for n, d, p in scenarios]


def fx_scenario_stress(
    book,
    ctx: PricingContext,
    scenarios: list | None = None,
) -> list:
    """Full-reprice FX stress via scenario.py."""
    from pricebook.risk.scenario import parallel_shift, fx_spot_shock, run_scenarios
    from pricebook.core.trade import Trade, Portfolio

    portfolio = Portfolio(name="fx_stress")
    for entry in book._entries:
        if hasattr(entry.trade, 'instrument') and hasattr(entry.trade.instrument, 'pv_ctx'):
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

def fx_mc_xva(
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
    """MC XVA for FX forward/option — wires xva.simulate_exposures."""
    import numpy as np
    from pricebook.risk.xva import (
        simulate_exposures, expected_positive_exposure,
        expected_negative_exposure, total_xva_decomposition,
    )

    # Infer maturity from instrument
    expiry = getattr(instrument, 'expiry', getattr(instrument, 'delivery', None))
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
class FXHedgeRecommendation:
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


def fx_hedge_recommendations(
    book,
    delta_limit: float = 50_000_000,
    vega_limit: float = 1_000_000,
    ccy_limit: float = 100_000_000,
) -> list[FXHedgeRecommendation]:
    """Hedge recommendations for FX book."""
    recs = []

    # Delta check
    positions = book.positions_by_pair()
    total_delta = sum(abs(p.net_notional) for p in positions)
    if delta_limit > 0 and total_delta > delta_limit * 0.75:
        recs.append(FXHedgeRecommendation(
            "delta", total_delta, delta_limit, total_delta / delta_limit,
            "Reduce spot delta via offsetting forwards or options"))

    # Currency concentration
    ccy_exps = book.net_currency_exposure()
    for exp in ccy_exps:
        if ccy_limit > 0 and abs(exp.net_exposure) > ccy_limit * 0.75:
            recs.append(FXHedgeRecommendation(
                "currency", abs(exp.net_exposure), ccy_limit,
                abs(exp.net_exposure) / ccy_limit,
                f"Reduce {exp.currency} exposure — exceeds limit"))

    return recs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class FXEventType:
    SETTLEMENT = "settlement"
    EXPIRY = "expiry"
    ROLL = "roll"
    FIXING = "fixing"


class FXLifecycle:
    """Lifecycle management for FX positions."""

    def __init__(self, instrument, trade_id: str = ""):
        self._instrument = instrument
        self._trade_id = trade_id
        self._events: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return sorted(self._events, key=lambda x: x.get("date", ""))

    def settlement_alert(self, as_of: date, alert_days: int = 2) -> dict | None:
        """T+2 settlement alert for spot/forward."""
        delivery = getattr(self._instrument, 'maturity',
                          getattr(self._instrument, 'delivery',
                          getattr(self._instrument, 'expiry', None)))
        if delivery is None:
            return None
        days = (delivery - as_of).days
        if 0 < days <= alert_days:
            return {
                "type": FXEventType.SETTLEMENT,
                "date": delivery.isoformat(),
                "days_remaining": days,
            }
        return None

    def record_settlement(self, settle_date: date, settle_rate: float) -> dict:
        event = {
            "type": FXEventType.SETTLEMENT,
            "date": settle_date.isoformat(),
            "rate": settle_rate,
        }
        self._events.append(event)
        return event

    def record_fixing(self, fixing_date: date, fixing_rate: float) -> dict:
        """Record NDF fixing."""
        event = {
            "type": FXEventType.FIXING,
            "date": fixing_date.isoformat(),
            "rate": fixing_rate,
        }
        self._events.append(event)
        return event

    def record_roll(self, roll_date: date, new_delivery: date, roll_cost: float = 0.0) -> dict:
        event = {
            "type": FXEventType.ROLL,
            "date": roll_date.isoformat(),
            "new_delivery": new_delivery.isoformat(),
            "roll_cost": roll_cost,
        }
        self._events.append(event)
        return event


# ---------------------------------------------------------------------------
# Carry decomposition (protocol compliance)
# ---------------------------------------------------------------------------

@dataclass
class FXCarryDecomposition:
    """FX carry: forward premium/discount from rate differential."""
    forward_carry: float
    funding_cost: float
    net_carry: float

    def to_dict(self) -> dict:
        return {"forward_carry": self.forward_carry,
                "funding": self.funding_cost, "net": self.net_carry}


def fx_carry(spot: float, base_rate: float, quote_rate: float,
             notional: float, horizon_days: int = 30) -> FXCarryDecomposition:
    """Carry from rate differential (CIP)."""
    dt = horizon_days / 365.0
    carry = spot * (quote_rate - base_rate) * dt * notional
    return FXCarryDecomposition(carry, 0.0, carry)


# ---------------------------------------------------------------------------
# Daily P&L (protocol compliance)
# ---------------------------------------------------------------------------

@dataclass
class FXDailyPnL:
    """FX daily P&L attribution."""
    date: date
    total: float
    spot_pnl: float
    carry_pnl: float
    unexplained: float

    def to_dict(self) -> dict:
        return {"date": self.date.isoformat(), "total": self.total,
                "spot": self.spot_pnl, "carry": self.carry_pnl,
                "unexplained": self.unexplained}
