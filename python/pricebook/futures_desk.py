"""Futures trading desk: multi-asset (bond, IR, equity, commodity, FX).

Unified desk layer for all futures types: position management, margin,
daily settlement, carry/roll, stress, and lifecycle.

    from pricebook.futures_desk import (
        FuturesBook, FuturesBookEntry,
        futures_risk_metrics, FuturesRiskMetrics,
        futures_daily_settlement, FuturesMargin,
        futures_dashboard, FuturesDashboard,
        futures_stress_suite, futures_hedge_recommendations,
        FuturesLifecycle,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Asset classes
# ---------------------------------------------------------------------------

class FuturesAssetClass(Enum):
    BOND = "bond"
    IR = "ir"
    EQUITY = "equity"
    COMMODITY = "commodity"
    FX = "fx"


# ---------------------------------------------------------------------------
# BondFuture wrapper (stateful class wrapping bond_futures.py)
# ---------------------------------------------------------------------------

class BondFuture:
    """Stateful bond future for desk integration.

    Wraps bond_futures.py CTD analytics into a priceable object.
    PV = (market_price - trade_price) × multiplier × contracts.
    """

    def __init__(
        self,
        trade_price: float,
        market_price: float,
        expiry: date,
        multiplier: float = 1000.0,  # $/point (UST: $1000 per point)
        ctd_dv01: float = 0.0,       # CTD bond DV01 (per 100 face)
        ctd_cf: float = 1.0,         # conversion factor
        notional: float = 100_000,   # contract notional
    ):
        self.trade_price = trade_price
        self.market_price = market_price
        self.expiry = expiry
        self.multiplier = multiplier
        self.ctd_dv01 = ctd_dv01
        self.ctd_cf = ctd_cf
        self.notional = notional

    def pv(self, contracts: int = 1) -> float:
        """Mark-to-market P&L per position."""
        return (self.market_price - self.trade_price) * self.multiplier * contracts

    def pv_ctx(self, ctx) -> float:
        return self.pv()

    def dv01(self, contracts: int = 1) -> float:
        """Futures DV01 = CTD_DV01 / CF × multiplier × contracts."""
        if self.ctd_cf > 0:
            return self.ctd_dv01 / self.ctd_cf * self.multiplier / 100 * contracts
        return 0.0

    def basis(self) -> float:
        return self.market_price - self.trade_price


# ---------------------------------------------------------------------------
# FXFuture (new)
# ---------------------------------------------------------------------------

class FXFuture:
    """FX future: F = spot × df_domestic / df_foreign.

    Daily cash settlement in domestic currency.
    """

    def __init__(
        self,
        base_ccy: str,
        quote_ccy: str,
        spot: float,
        expiry: date,
        contract_size: float = 125_000,  # standard EUR/USD contract
    ):
        self.base_ccy = base_ccy
        self.quote_ccy = quote_ccy
        self.spot = spot
        self.expiry = expiry
        self.contract_size = contract_size

    def fair_price(self, domestic_curve: DiscountCurve, foreign_curve: DiscountCurve,
                   valuation_date: date) -> float:
        """Forward = spot × df_foreign / df_domestic (covered IRP)."""
        df_d = domestic_curve.df(self.expiry)
        df_f = foreign_curve.df(self.expiry)
        return self.spot * df_f / df_d

    def pv(self, trade_price: float, market_price: float, contracts: int = 1) -> float:
        return (market_price - trade_price) * self.contract_size * contracts

    def pv_ctx(self, ctx) -> float:
        return 0.0  # needs trade price context


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class FuturesRiskMetrics:
    """Unified risk metrics for a futures position."""
    pv: float
    dv01: float
    contracts: int
    asset_class: str
    expiry: date
    margin_required: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "dv01": self.dv01, "contracts": self.contracts,
            "asset_class": self.asset_class, "expiry": self.expiry.isoformat(),
            "margin": self.margin_required,
        }


def futures_risk_metrics(
    instrument,
    contracts: int = 1,
    margin_per_contract: float = 0.0,
) -> FuturesRiskMetrics:
    """Compute risk metrics for any futures position."""
    if isinstance(instrument, BondFuture):
        pv = instrument.pv(contracts)
        dv01 = instrument.dv01(contracts)
        ac = "bond"
        expiry = instrument.expiry
    elif hasattr(instrument, 'accrual_start'):  # IRFuture
        pv = 0.0  # IR futures need trade price
        dv01 = getattr(instrument, 'tick_value', 0) * contracts
        ac = "ir"
        expiry = instrument.accrual_end
    elif hasattr(instrument, 'notional_per_point'):  # EquityFuture
        pv = 0.0
        dv01 = 0.0  # equity futures: delta, not DV01
        ac = "equity"
        expiry = instrument.expiry
    elif isinstance(instrument, FXFuture):
        pv = 0.0
        dv01 = 0.0
        ac = "fx"
        expiry = instrument.expiry
    else:
        pv = 0.0
        dv01 = 0.0
        ac = "commodity"
        expiry = getattr(instrument, 'expiry', date.today())

    return FuturesRiskMetrics(
        pv=pv, dv01=dv01, contracts=contracts,
        asset_class=ac, expiry=expiry,
        margin_required=margin_per_contract * abs(contracts),
    )


# ---------------------------------------------------------------------------
# Futures Book
# ---------------------------------------------------------------------------

@dataclass
class FuturesBookEntry:
    """A futures position in the book."""
    trade_id: str
    instrument: object
    contracts: int = 1
    asset_class: FuturesAssetClass = FuturesAssetClass.BOND
    counterparty: str = ""
    exchange: str = ""
    margin_per_contract: float = 0.0

    @property
    def margin_required(self) -> float:
        return self.margin_per_contract * abs(self.contracts)


class FuturesBook:
    """Multi-asset futures book."""

    def __init__(self, name: str = "futures_book"):
        self.name = name
        self._entries: list[FuturesBookEntry] = []

    def add(self, entry: FuturesBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[FuturesBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_contracts(self) -> int:
        return sum(abs(e.contracts) for e in self._entries)

    def total_margin(self) -> float:
        return sum(e.margin_required for e in self._entries)

    def by_asset_class(self) -> dict[str, list[FuturesBookEntry]]:
        result: dict[str, list[FuturesBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.asset_class.value, []).append(e)
        return result

    def by_exchange(self) -> dict[str, list[FuturesBookEntry]]:
        result: dict[str, list[FuturesBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.exchange, []).append(e)
        return result

    def aggregate_risk(self) -> dict[str, float]:
        total_pv = 0.0
        total_dv01 = 0.0
        total_margin = 0.0
        total_contracts = 0

        for e in self._entries:
            rm = futures_risk_metrics(e.instrument, e.contracts, e.margin_per_contract)
            total_pv += rm.pv
            total_dv01 += rm.dv01
            total_margin += rm.margin_required
            total_contracts += abs(e.contracts)

        return {
            "total_pv": total_pv,
            "total_dv01": total_dv01,
            "total_margin": total_margin,
            "total_contracts": total_contracts,
            "n_positions": len(self._entries),
        }


# ---------------------------------------------------------------------------
# Margin framework
# ---------------------------------------------------------------------------

@dataclass
class FuturesMarginState:
    """Margin state for a position."""
    initial_margin: float
    variation_margin: float
    total_margin: float
    margin_call: float  # positive if more margin needed

    def to_dict(self) -> dict:
        return {
            "initial": self.initial_margin, "variation": self.variation_margin,
            "total": self.total_margin, "margin_call": self.margin_call,
        }


def futures_daily_settlement(
    prev_price: float,
    curr_price: float,
    multiplier: float,
    contracts: int,
) -> float:
    """Daily variation margin = (curr - prev) × multiplier × contracts."""
    return (curr_price - prev_price) * multiplier * contracts


def futures_margin_check(
    position_value: float,
    initial_margin: float,
    maintenance_margin: float,
) -> FuturesMarginState:
    """Check margin adequacy and compute margin call if needed."""
    variation = position_value  # cumulative MTM
    total = initial_margin + variation
    call = max(initial_margin - total, 0) if total < maintenance_margin else 0.0

    return FuturesMarginState(
        initial_margin=initial_margin,
        variation_margin=variation,
        total_margin=total,
        margin_call=call,
    )


# ---------------------------------------------------------------------------
# Carry / Roll
# ---------------------------------------------------------------------------

@dataclass
class FuturesCarryRoll:
    """Carry and roll analysis for a futures position."""
    basis_carry: float        # income from convergence
    roll_yield: float         # cost/benefit of rolling
    financing_cost: float     # margin opportunity cost
    net_carry: float

    def to_dict(self) -> dict:
        return {
            "basis_carry": self.basis_carry, "roll_yield": self.roll_yield,
            "financing": self.financing_cost, "net": self.net_carry,
        }


def futures_carry_roll(
    instrument,
    contracts: int = 1,
    days: int = 30,
    financing_rate: float = 0.04,
    margin_per_contract: float = 0.0,
) -> FuturesCarryRoll:
    """Compute carry and roll for a futures position."""
    # Basis carry: for bond futures, convergence toward CTD
    if isinstance(instrument, BondFuture):
        basis = instrument.basis()
        basis_carry = basis * instrument.multiplier * contracts * days / 365
    else:
        basis_carry = 0.0

    # Roll yield: cost of rolling from front to back month
    roll_yield = 0.0  # would need two contracts to compute

    # Financing: opportunity cost of margin
    margin = margin_per_contract * abs(contracts)
    financing = margin * financing_rate * days / 365

    net = basis_carry + roll_yield - financing

    return FuturesCarryRoll(basis_carry, roll_yield, financing, net)


# ---------------------------------------------------------------------------
# Daily P&L
# ---------------------------------------------------------------------------

@dataclass
class FuturesDailyPnL:
    """Daily P&L for a futures position."""
    date: date
    settlement_pnl: float    # daily mark variation
    carry: float
    total: float

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "settlement": self.settlement_pnl,
            "carry": self.carry, "total": self.total,
        }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class FuturesDashboard:
    """Morning summary for the futures desk."""
    date: date
    n_positions: int
    total_contracts: int
    total_pv: float
    total_dv01: float
    total_margin: float
    by_asset_class: dict[str, int]
    by_exchange: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "contracts": self.total_contracts, "pv": self.total_pv,
            "dv01": self.total_dv01, "margin": self.total_margin,
            "by_asset_class": self.by_asset_class,
            "by_exchange": self.by_exchange,
        }


def futures_dashboard(
    book: FuturesBook,
    reference_date: date,
) -> FuturesDashboard:
    """Build futures desk morning dashboard."""
    risk = book.aggregate_risk()
    by_ac = {k: len(v) for k, v in book.by_asset_class().items()}
    by_ex = {k: len(v) for k, v in book.by_exchange().items()}

    return FuturesDashboard(
        date=reference_date, n_positions=risk["n_positions"],
        total_contracts=risk["total_contracts"],
        total_pv=risk["total_pv"], total_dv01=risk["total_dv01"],
        total_margin=risk["total_margin"],
        by_asset_class=by_ac, by_exchange=by_ex,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class FuturesStressResult:
    scenario: str
    description: str
    pnl: float

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "description": self.description, "pnl": self.pnl}


def futures_stress_suite(
    book: FuturesBook,
) -> list[FuturesStressResult]:
    """Parametric stress scenarios for futures book."""
    risk = book.aggregate_risk()
    dv01 = risk["total_dv01"]

    scenarios = [
        ("rates_up_100", "Rates +100bp", dv01 * 100),
        ("rates_dn_100", "Rates -100bp", dv01 * -100),
        ("rates_up_200", "Rates +200bp", dv01 * 200),
    ]

    # Equity stress: estimate from total PV
    eq_entries = [e for e in book.entries if e.asset_class == FuturesAssetClass.EQUITY]
    eq_pv = sum(
        e.instrument.spot * e.instrument.notional_per_point * e.contracts
        for e in eq_entries if hasattr(e.instrument, 'spot')
    )
    scenarios.append(("equity_dn_10", "Equity -10%", -0.10 * eq_pv))
    scenarios.append(("combined", "Rates +100bp, Equity -5%", dv01 * 100 - 0.05 * eq_pv))

    return [FuturesStressResult(n, d, p) for n, d, p in scenarios]


# ---------------------------------------------------------------------------
# Capital (SA-CCR for exchange-traded via CCP)
# ---------------------------------------------------------------------------

@dataclass
class FuturesCapitalResult:
    """SA-CCR capital for a futures position via CCP."""
    ead: float
    rwa: float
    capital: float
    margin_required: float

    def to_dict(self) -> dict:
        return {"ead": self.ead, "rwa": self.rwa, "capital": self.capital,
                "margin": self.margin_required}


# CCP risk weights by asset class (Basel CRE54)
_CCP_RISK_WEIGHTS = {
    "bond": 0.02, "ir": 0.02, "fx": 0.02,
    "equity": 0.04, "commodity": 0.04,  # higher for some CCPs
}


def futures_capital(
    instrument,
    contracts: int = 1,
    margin_per_contract: float = 0.0,
    asset_class: str = "bond",
) -> FuturesCapitalResult:
    """SA-CCR capital for exchange-traded futures.

    Futures cleared via CCP: risk weight per asset class (Basel CRE54).
    Bond/IR/FX: 2%. Equity/Commodity: 4%.
    EAD = margin + |MTM|.
    """
    ccp_rw = _CCP_RISK_WEIGHTS.get(asset_class, 0.02)
    margin = margin_per_contract * abs(contracts)

    # EAD: margin + MTM (guard against missing pv)
    pv = 0.0
    if isinstance(instrument, BondFuture):
        try:
            pv = abs(instrument.pv(contracts))
        except (TypeError, ValueError):
            pv = 0.0

    ead = max(margin + abs(pv), margin)  # floor at margin
    rwa = ead * ccp_rw
    capital = rwa * 0.08

    return FuturesCapitalResult(ead=ead, rwa=rwa, capital=capital,
                                margin_required=margin)


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class FuturesHedgeRecommendation:
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


def futures_hedge_recommendations(
    book: FuturesBook,
    dv01_limit: float = 50_000,
    margin_limit: float = 10_000_000,
    concentration_limit: int = 500,
) -> list[FuturesHedgeRecommendation]:
    """Hedge recommendations for futures book."""
    risk = book.aggregate_risk()
    recs = []

    checks = [
        ("dv01", abs(risk["total_dv01"]), dv01_limit,
         "Reduce rate exposure via offsetting bond or IR futures"),
        ("margin", risk["total_margin"], margin_limit,
         "Reduce positions to free margin — approaching limit"),
        ("contracts", risk["total_contracts"], concentration_limit,
         "Reduce total contract count — operational risk"),
    ]

    for risk_type, current, limit, action in checks:
        if limit > 0 and current > limit * 0.75:
            recs.append(FuturesHedgeRecommendation(
                risk_type=risk_type, current=current, limit=limit,
                breach_pct=current / limit, action=action,
            ))

    return recs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class FuturesEventType:
    EXPIRY = "expiry"
    ROLL = "roll"
    DELIVERY = "delivery"
    FIRST_NOTICE = "first_notice"


class FuturesLifecycle:
    """Lifecycle management for a futures position."""

    def __init__(self, instrument, trade_id: str = "", contracts: int = 1):
        self._instrument = instrument
        self._trade_id = trade_id
        self._contracts = contracts
        self._events: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return sorted(self._events, key=lambda x: x.get("date", ""))

    def expiry_alert(self, as_of: date, alert_days: int = 10) -> dict | None:
        """Alert if expiry within alert_days."""
        expiry = getattr(self._instrument, 'expiry',
                         getattr(self._instrument, 'accrual_end', None))
        if expiry is None:
            return None
        days_to = (expiry - as_of).days
        if 0 < days_to <= alert_days:
            return {
                "type": FuturesEventType.EXPIRY,
                "date": expiry.isoformat(),
                "days_remaining": days_to,
                "contracts": self._contracts,
            }
        return None

    def record_roll(self, roll_date: date, new_expiry: date, roll_cost: float = 0.0) -> dict:
        """Record roll from current to next contract."""
        event = {
            "type": FuturesEventType.ROLL,
            "date": roll_date.isoformat(),
            "new_expiry": new_expiry.isoformat(),
            "roll_cost": roll_cost,
        }
        self._events.append(event)
        return event

    def record_delivery(self, delivery_date: date, settlement_price: float) -> dict:
        """Record physical or cash delivery."""
        event = {
            "type": FuturesEventType.DELIVERY,
            "date": delivery_date.isoformat(),
            "settlement_price": settlement_price,
            "contracts": self._contracts,
        }
        self._events.append(event)
        return event
