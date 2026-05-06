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
    gamma: float               # d²PV/d(breakeven)²
    notional: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "ie01": self.ie01,
            "real_dv01": self.real_dv01, "nominal_dv01": self.nominal_dv01,
            "gamma": self.gamma, "notional": self.notional,
        }


def _price_inflation(instrument, discount_curve, cpi_curve) -> float:
    """Uniform pricer dispatching on instrument type."""
    from pricebook.inflation import InflationLinkedBond, ZCInflationSwap, YoYInflationSwap

    if isinstance(instrument, InflationLinkedBond):
        return instrument.dirty_price(discount_curve, cpi_curve) * instrument.notional / 100.0
    elif isinstance(instrument, (ZCInflationSwap, YoYInflationSwap)):
        return instrument.pv(discount_curve, cpi_curve)
    elif hasattr(instrument, 'pv'):
        # Generic fallback for any instrument with .pv(disc, cpi)
        return instrument.pv(discount_curve, cpi_curve)
    else:
        raise TypeError(f"Cannot price inflation instrument of type {type(instrument).__name__}")


def inflation_risk_metrics(
    instrument,
    discount_curve: DiscountCurve,
    cpi_curve,
    bump: float = 0.0001,
) -> InflationRiskMetrics:
    """Compute inflation risk metrics via multi-curve bump-and-reprice.

    Args:
        instrument: InflationLinkedBond, ZCInflationSwap, YoYInflationSwap, or
            any object with .pv(discount_curve, cpi_curve) and .notional.
        discount_curve: nominal discount curve.
        cpi_curve: CPICurve (expected CPI index levels).
        bump: shift size for sensitivities (default 1bp).

    Returns:
        InflationRiskMetrics with IE01, real DV01, nominal DV01, gamma.
    """
    base_pv = _price_inflation(instrument, discount_curve, cpi_curve)

    # IE01: bump breakeven (CPI curve) by 1bp
    cpi_up = cpi_curve.bumped(bump)
    cpi_dn = cpi_curve.bumped(-bump)
    pv_cpi_up = _price_inflation(instrument, discount_curve, cpi_up)
    pv_cpi_dn = _price_inflation(instrument, discount_curve, cpi_dn)
    ie01 = (pv_cpi_up - pv_cpi_dn) / 2

    # Gamma: second-order breakeven sensitivity
    gamma = (pv_cpi_up - 2 * base_pv + pv_cpi_dn) / (bump ** 2)

    # Real DV01: bump discount curve only (centred)
    pv_disc_up = _price_inflation(instrument, discount_curve.bumped(bump), cpi_curve)
    pv_disc_dn = _price_inflation(instrument, discount_curve.bumped(-bump), cpi_curve)
    real_dv01 = (pv_disc_up - pv_disc_dn) / 2

    # Nominal DV01: bump both curves simultaneously
    pv_both_up = _price_inflation(instrument, discount_curve.bumped(bump), cpi_up)
    nominal_dv01 = pv_both_up - base_pv

    notional = getattr(instrument, 'notional', 0.0)

    return InflationRiskMetrics(
        pv=base_pv, ie01=ie01, real_dv01=real_dv01,
        nominal_dv01=nominal_dv01, gamma=gamma, notional=notional,
    )


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
    total_ie01: float = 0.0,
    total_real_dv01: float = 0.0,
    total_nominal_dv01: float = 0.0,
    *,
    book=None,
    discount_curve: DiscountCurve | None = None,
    cpi_curve=None,
) -> list[InflationStressResult]:
    """Parametric inflation stress scenarios.

    Can be called with explicit risk numbers or with book + curves
    (aggregate_risk is called automatically).
    """
    if book is not None and discount_curve is not None and cpi_curve is not None:
        risk = book.aggregate_risk(discount_curve, cpi_curve)
        total_ie01 = risk.get("total_ie01", total_ie01)
        total_real_dv01 = risk.get("total_real_dv01", total_real_dv01)
        total_nominal_dv01 = risk.get("total_nominal_dv01", total_nominal_dv01)

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


def inflation_carry_decomposition(
    instrument,
    discount_curve: DiscountCurve,
    cpi_curve,
    horizon_days: int = 1,
) -> InflationCarryDecomposition:
    """Decompose inflation carry into real yield and breakeven accrual.

    Real yield carry: the real coupon/rate income over the horizon.
    Breakeven accrual: the CPI indexation (inflation expectation) over the horizon.
    """
    dt = horizon_days / 365.0

    # Real yield component: coupon rate (for linker/swap fixed rate)
    real_rate = getattr(instrument, 'coupon_rate',
                        getattr(instrument, 'fixed_rate', 0.0))
    notional = getattr(instrument, 'notional', 0.0)
    real_yield_carry = real_rate * notional * dt

    # Breakeven accrual: CPI forward rate * notional * dt
    maturity = getattr(instrument, 'end',
                       getattr(instrument, 'maturity', None))
    if maturity is not None:
        be = cpi_curve.breakeven_rate(maturity)
    else:
        be = 0.0
    breakeven_accrual = be * notional * dt

    net = real_yield_carry + breakeven_accrual

    return InflationCarryDecomposition(
        real_yield_carry=real_yield_carry,
        breakeven_accrual=breakeven_accrual,
        net_carry=net,
    )


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


def inflation_daily_pnl(
    instrument,
    disc_t0: DiscountCurve,
    cpi_t0,
    disc_t1: DiscountCurve,
    cpi_t1,
    date_t1: date,
) -> InflationDailyPnL:
    """Daily P&L attribution for an inflation position.

    Decomposes total P&L into breakeven, real rate, carry, and unexplained.

    Args:
        instrument: inflation instrument.
        disc_t0, cpi_t0: yesterday's curves.
        disc_t1, cpi_t1: today's curves.
        date_t1: today's date.
    """
    pv_t0 = _price_inflation(instrument, disc_t0, cpi_t0)
    pv_t1 = _price_inflation(instrument, disc_t1, cpi_t1)
    total = pv_t1 - pv_t0

    # Breakeven P&L: reprice with new CPI curve, old discount curve
    pv_cpi_only = _price_inflation(instrument, disc_t0, cpi_t1)
    breakeven_pnl = pv_cpi_only - pv_t0

    # Real rate P&L: reprice with new discount curve, old CPI curve
    pv_disc_only = _price_inflation(instrument, disc_t1, cpi_t0)
    real_rate_pnl = pv_disc_only - pv_t0

    # Carry: 1-day carry from yesterday's curves
    carry = inflation_carry_decomposition(instrument, disc_t0, cpi_t0, horizon_days=1)
    carry_pnl = carry.net_carry

    # Unexplained: residual (cross-gamma, curve shape, etc.)
    unexplained = total - breakeven_pnl - real_rate_pnl - carry_pnl

    return InflationDailyPnL(
        date=date_t1, total=total,
        breakeven_pnl=breakeven_pnl, real_rate_pnl=real_rate_pnl,
        carry_pnl=carry_pnl, unexplained=unexplained,
    )


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


def inflation_capital(
    instrument,
    discount_curve: DiscountCurve,
    cpi_curve,
    counterparty_rw: float = 0.20,
) -> InflationCapitalResult:
    """SA-CCR capital for an inflation position.

    Uses GIRR supervisory factor (SF=0.005) for inflation products.
    SIMM: IE01 mapped to GIRR inflation bucket.

    Args:
        instrument: inflation instrument with .notional.
        discount_curve: nominal discount curve.
        cpi_curve: CPICurve.
        counterparty_rw: counterparty risk weight (default 20%).
    """
    from pricebook.day_count import year_fraction as _yf, DayCountConvention as _DC

    rm = inflation_risk_metrics(instrument, discount_curve, cpi_curve)
    pv = rm.pv
    mtm = max(pv, 0)
    notional = rm.notional

    # Time to maturity
    maturity = getattr(instrument, 'end',
                       getattr(instrument, 'maturity', None))
    if maturity is not None:
        T = _yf(discount_curve.reference_date, maturity, _DC.ACT_365_FIXED)
    else:
        T = 1.0

    # SA-CCR: SF=0.005 for GIRR
    sf = 0.005
    mf = math.sqrt(min(T, 1.0))
    ead = 1.4 * (mtm + notional * sf * mf)
    rwa = ead * counterparty_rw
    capital = rwa * 0.08

    # SIMM: IE01 into GIRR inflation bucket
    # GIRR inflation risk weight = 16bp (ISDA SIMM 2.6)
    girr_inflation_rw = 0.0016
    simm_im = abs(rm.ie01) / girr_inflation_rw if girr_inflation_rw > 0 else 0.0

    return InflationCapitalResult(ead=ead, rwa=rwa, capital=capital, simm_im=simm_im)
