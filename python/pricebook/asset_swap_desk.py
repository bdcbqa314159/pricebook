"""Asset swap trading desk: book, risk, carry, stress, capital, lifecycle.

Consolidates par_asset_swap.py, cmasw.py, and risky_bond.py primitives
into the standard 9-component desk protocol.

    from pricebook.asset_swap_desk import (
        asw_risk_metrics, ASWRiskMetrics,
        ASWBook, ASWBookEntry,
        asw_carry_decomposition, ASWCarryDecomposition,
        asw_daily_pnl, ASWDailyPnL,
        asw_dashboard, ASWDashboard,
        asw_stress_suite, ASWStressResult,
        asw_capital, ASWCapitalResult,
        asw_hedge_recommendations, ASWHedgeRecommendation,
        ASWLifecycle,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.par_asset_swap import ParAssetSwap, ProceedsAssetSwap


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class ASWRiskMetrics:
    """Unified risk metrics for an asset swap position."""
    pv: float               # net PV of the ASW package
    asw_spread: float       # current ASW spread (decimal, e.g. 0.005 = 50bp)
    asw01: float            # PV change per 1bp ASW spread move
    dv01: float             # net rate sensitivity (bond + swap)
    bond_dv01: float        # bond leg DV01
    swap_dv01: float        # swap leg DV01 (opposite sign to bond)
    basis_risk: float       # bond_dv01 + swap_dv01 residual
    notional: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "asw_spread": self.asw_spread,
            "asw01": self.asw01, "dv01": self.dv01,
            "bond_dv01": self.bond_dv01, "swap_dv01": self.swap_dv01,
            "basis_risk": self.basis_risk, "notional": self.notional,
        }


def _price_asw(instrument, curve: DiscountCurve):
    """Price an ASW instrument, returning (result, bond_pv_per100)."""
    result = instrument.price(curve)
    return result


def asw_risk_metrics(
    instrument: ParAssetSwap | ProceedsAssetSwap,
    curve: DiscountCurve,
    bump: float = 0.0001,
) -> ASWRiskMetrics:
    """Compute asset swap risk metrics via bump-and-reprice.

    Args:
        instrument: ParAssetSwap or ProceedsAssetSwap.
        curve: OIS discount curve.
        bump: shift size (default 1bp).

    Returns:
        ASWRiskMetrics with spread, ASW01, DV01, basis risk.
    """
    bond = instrument.bond
    base = instrument.price(curve)

    # Bond DV01: bump curve, reprice bond
    bond_pv_base = bond.dirty_price(curve)
    bond_pv_up = bond.dirty_price(curve.bumped(bump))
    bond_pv_dn = bond.dirty_price(curve.bumped(-bump))
    bond_dv01 = (bond_pv_up - bond_pv_dn) / 2  # per 100 face

    # Swap DV01: annuity changes with rates
    # For an ASW, the swap leg PV = ASW_spread × annuity × face
    # So swap_dv01 ≈ -bond_dv01 (by construction, ASW nets to near zero)
    result_up = instrument.price(curve.bumped(bump))
    result_dn = instrument.price(curve.bumped(-bump))
    # The ASW spread changes when the curve moves — the net DV01 is the
    # residual after bond and swap legs offset
    spread_up = result_up.asw_spread
    spread_dn = result_dn.asw_spread

    # Net DV01: the residual rate exposure of the ASW package
    # ASW PV = (bond_rf_pv - market_price) - asw_spread × annuity × face
    # Bumping the curve changes both sides — net DV01 is the residual
    annuity = base.annuity
    face = bond.face_value

    # Approximate swap DV01 from annuity change
    annuity_up = result_up.annuity
    annuity_dn = result_dn.annuity
    swap_dv01 = -base.asw_spread * (annuity_up - annuity_dn) / 2 * face

    # Net DV01 = bond_dv01 + swap_dv01 (should be small for par ASW)
    net_dv01 = bond_dv01 * face / 100.0 + swap_dv01
    basis_risk = abs(net_dv01)

    # ASW01: PV change per 1bp spread move
    # ASW01 = annuity × face × 0.0001
    asw01 = annuity * face * bump

    # Net PV: bond PV - market price (par convention)
    notional = face
    pv = (base.bond_pv - base.market_price) / 100.0 * face

    return ASWRiskMetrics(
        pv=pv,
        asw_spread=base.asw_spread,
        asw01=asw01,
        dv01=net_dv01,
        bond_dv01=bond_dv01 * face / 100.0,
        swap_dv01=swap_dv01,
        basis_risk=basis_risk,
        notional=notional,
    )


# ---------------------------------------------------------------------------
# Book management
# ---------------------------------------------------------------------------

@dataclass
class ASWBookEntry:
    """A single asset swap position."""
    trade_id: str
    instrument: ParAssetSwap | ProceedsAssetSwap
    convention: str = "par"       # "par" or "proceeds"
    counterparty: str = ""
    bond_issuer: str = ""
    sector: str = ""
    currency: str = "USD"
    direction: int = 1            # +1 long, -1 short


class ASWBook:
    """Asset swap position book."""

    def __init__(self, name: str = "asw_book"):
        self.name = name
        self._entries: list[ASWBookEntry] = []

    def add(self, entry: ASWBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[ASWBookEntry]:
        return list(self._entries)

    def positions(self) -> list[ASWBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_notional(self) -> float:
        return sum(e.instrument.bond.face_value for e in self._entries)

    def by_issuer(self) -> dict[str, list[ASWBookEntry]]:
        result: dict[str, list[ASWBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.bond_issuer, []).append(e)
        return result

    def by_sector(self) -> dict[str, list[ASWBookEntry]]:
        result: dict[str, list[ASWBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.sector, []).append(e)
        return result

    def by_convention(self) -> dict[str, list[ASWBookEntry]]:
        result: dict[str, list[ASWBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.convention, []).append(e)
        return result

    def aggregate_risk(self, curve: DiscountCurve) -> dict:
        """Aggregate risk across all positions."""
        total_pv = 0.0
        total_asw01 = 0.0
        total_dv01 = 0.0
        total_notional = 0.0

        for e in self._entries:
            rm = asw_risk_metrics(e.instrument, curve)
            total_pv += e.direction * rm.pv
            total_asw01 += e.direction * rm.asw01
            total_dv01 += e.direction * rm.dv01
            total_notional += rm.notional

        return {
            "total_pv": total_pv,
            "total_asw01": total_asw01,
            "total_dv01": total_dv01,
            "n_positions": len(self._entries),
            "total_notional": total_notional,
        }


# ---------------------------------------------------------------------------
# Carry decomposition
# ---------------------------------------------------------------------------

@dataclass
class ASWCarryDecomposition:
    """Asset swap carry attribution."""
    coupon_income: float       # bond coupon received
    floating_payment: float    # SOFR + ASW spread paid
    funding_cost: float        # repo financing of bond leg
    net_carry: float

    def to_dict(self) -> dict:
        return {"coupon": self.coupon_income, "floating": self.floating_payment,
                "funding": self.funding_cost, "net": self.net_carry}


def asw_carry_decomposition(
    instrument: ParAssetSwap | ProceedsAssetSwap,
    curve: DiscountCurve,
    repo_rate: float = 0.04,
    horizon_days: int = 1,
) -> ASWCarryDecomposition:
    """Decompose ASW carry into coupon, floating, and funding."""
    dt = horizon_days / 365.0
    bond = instrument.bond
    face = bond.face_value
    result = instrument.price(curve)

    # Coupon income: bond coupon accrual
    coupon_income = bond.coupon_rate * face * dt

    # Floating payment: (forward rate + ASW spread) × face × dt
    fwd = curve.forward_rate(curve.reference_date,
                              curve.reference_date + timedelta(days=max(horizon_days, 1)))
    floating_payment = (fwd + result.asw_spread) * face * dt

    # Funding cost: repo rate × dirty price / 100 × face × dt
    dirty = bond.dirty_price(curve)
    funding_cost = repo_rate * (dirty / 100.0) * face * dt

    net = coupon_income - floating_payment - funding_cost

    return ASWCarryDecomposition(
        coupon_income=coupon_income,
        floating_payment=floating_payment,
        funding_cost=funding_cost,
        net_carry=net,
    )


# ---------------------------------------------------------------------------
# Daily P&L
# ---------------------------------------------------------------------------

@dataclass
class ASWDailyPnL:
    """Asset swap daily P&L attribution."""
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


def asw_daily_pnl(
    instrument: ParAssetSwap | ProceedsAssetSwap,
    curve_t0: DiscountCurve,
    curve_t1: DiscountCurve,
    date_t1: date,
    repo_rate: float = 0.04,
) -> ASWDailyPnL:
    """Daily P&L attribution for an ASW position."""
    r0 = instrument.price(curve_t0)
    r1 = instrument.price(curve_t1)
    face = instrument.bond.face_value

    pv_t0 = (r0.bond_pv - r0.market_price) / 100.0 * face
    pv_t1 = (r1.bond_pv - r1.market_price) / 100.0 * face
    total = pv_t1 - pv_t0

    # Spread P&L: ASW spread change × ASW01
    asw01 = r0.annuity * face * 0.0001
    spread_change_bp = (r1.asw_spread - r0.asw_spread) * 10_000
    spread_pnl = asw01 * spread_change_bp

    # Rate P&L: bond PV change net of spread
    rate_pnl = (r1.bond_pv - r0.bond_pv) / 100.0 * face - spread_pnl

    # Carry
    carry = asw_carry_decomposition(instrument, curve_t0, repo_rate, horizon_days=1)
    carry_pnl = carry.net_carry

    unexplained = total - spread_pnl - rate_pnl - carry_pnl

    return ASWDailyPnL(
        date=date_t1, total=total,
        spread_pnl=spread_pnl, rate_pnl=rate_pnl,
        carry_pnl=carry_pnl, unexplained=unexplained,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class ASWDashboard:
    """Asset swap desk morning summary."""
    date: date
    n_positions: int
    total_notional: float
    total_asw01: float
    total_dv01: float
    average_spread_bp: float
    by_sector: dict[str, int]
    by_convention: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "notional": self.total_notional, "asw01": self.total_asw01,
            "dv01": self.total_dv01, "avg_spread_bp": self.average_spread_bp,
            "by_sector": self.by_sector, "by_convention": self.by_convention,
        }


def asw_dashboard(
    book: ASWBook,
    reference_date: date,
    curve: DiscountCurve,
) -> ASWDashboard:
    """Build asset swap desk morning dashboard."""
    risk = book.aggregate_risk(curve)

    by_sector = {k: len(v) for k, v in book.by_sector().items()}
    by_convention = {k: len(v) for k, v in book.by_convention().items()}

    # Average spread
    spreads = []
    for e in book.entries:
        r = e.instrument.price(curve)
        spreads.append(r.asw_spread)
    avg_spread_bp = (sum(spreads) / len(spreads) * 10_000) if spreads else 0.0

    return ASWDashboard(
        date=reference_date,
        n_positions=risk["n_positions"],
        total_notional=risk["total_notional"],
        total_asw01=risk["total_asw01"],
        total_dv01=risk["total_dv01"],
        average_spread_bp=avg_spread_bp,
        by_sector=by_sector,
        by_convention=by_convention,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class ASWStressResult:
    scenario: str
    description: str
    pnl: float

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "description": self.description, "pnl": self.pnl}


def asw_stress_suite(
    book: ASWBook,
    curve: DiscountCurve,
) -> list[ASWStressResult]:
    """Parametric stress scenarios for the ASW book."""
    risk = book.aggregate_risk(curve)
    asw01 = risk["total_asw01"]
    dv01 = risk["total_dv01"]

    scenarios = [
        ("rates_up_100", "Rates +100bp", dv01 * 100),
        ("rates_dn_100", "Rates -100bp", dv01 * -100),
        ("spread_up_50", "ASW spreads +50bp", asw01 * 50),
        ("spread_dn_50", "ASW spreads -50bp", asw01 * -50),
        ("combined", "Rates +50bp, spreads +25bp",
         dv01 * 50 + asw01 * 25),
    ]
    return [ASWStressResult(n, d, p) for n, d, p in scenarios]


# ---------------------------------------------------------------------------
# Capital
# ---------------------------------------------------------------------------

@dataclass
class ASWCapitalResult:
    ead: float
    rwa: float
    capital: float
    simm_im: float

    def to_dict(self) -> dict:
        return {"ead": self.ead, "rwa": self.rwa, "capital": self.capital,
                "simm_im": self.simm_im}


def asw_capital(
    instrument: ParAssetSwap | ProceedsAssetSwap,
    curve: DiscountCurve,
    counterparty_rw: float = 0.20,
) -> ASWCapitalResult:
    """SA-CCR capital for an ASW position."""
    rm = asw_risk_metrics(instrument, curve)
    mtm = max(rm.pv, 0)
    notional = rm.notional

    maturity = instrument.bond.maturity
    T = year_fraction(curve.reference_date, maturity, DayCountConvention.ACT_365_FIXED)

    sf = 0.005  # GIRR supervisory factor
    mf = math.sqrt(min(T, 1.0))
    ead = 1.4 * (mtm + notional * sf * mf)
    rwa = ead * counterparty_rw
    capital = rwa * 0.08

    # SIMM: ASW01 into CSR non-securitisation bucket
    csr_rw = 0.005  # CSR risk weight ~50bp
    simm_im = abs(rm.asw01) * csr_rw * math.sqrt(10.0 / 252.0)

    return ASWCapitalResult(ead=ead, rwa=rwa, capital=capital, simm_im=simm_im)


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class ASWHedgeRecommendation:
    risk_type: str
    current: float
    limit: float
    breach_pct: float
    action: str

    def to_dict(self) -> dict:
        return {"risk": self.risk_type, "current": self.current,
                "limit": self.limit, "breach_pct": self.breach_pct,
                "action": self.action}


def asw_hedge_recommendations(
    book: ASWBook,
    curve: DiscountCurve,
    dv01_limit: float = 50_000,
    asw01_limit: float = 100_000,
) -> list[ASWHedgeRecommendation]:
    """Hedge recommendations for ASW book."""
    risk = book.aggregate_risk(curve)
    recs = []

    if dv01_limit > 0 and abs(risk["total_dv01"]) > dv01_limit * 0.75:
        recs.append(ASWHedgeRecommendation(
            "dv01", abs(risk["total_dv01"]), dv01_limit,
            abs(risk["total_dv01"]) / dv01_limit,
            "Hedge rate exposure via IRS or futures"))

    if asw01_limit > 0 and abs(risk["total_asw01"]) > asw01_limit * 0.75:
        recs.append(ASWHedgeRecommendation(
            "asw01", abs(risk["total_asw01"]), asw01_limit,
            abs(risk["total_asw01"]) / asw01_limit,
            "Reduce spread exposure via offsetting ASW or CDS"))

    # Concentration by issuer
    by_issuer = book.by_issuer()
    total = risk["total_notional"]
    for issuer, entries in by_issuer.items():
        issuer_notional = sum(e.instrument.bond.face_value for e in entries)
        if total > 0 and issuer_notional / total > 0.25:
            recs.append(ASWHedgeRecommendation(
                "concentration", issuer_notional / total, 0.25,
                (issuer_notional / total) / 0.25,
                f"Reduce concentration in {issuer}"))

    return recs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class ASWEventType:
    BOND_MATURITY = "bond_maturity"
    COUPON = "coupon"
    SWAP_RESET = "swap_reset"
    BASIS_ALERT = "basis_alert"


class ASWLifecycle:
    """Lifecycle management for asset swap positions."""

    def __init__(self, instrument, trade_id: str = ""):
        self._instrument = instrument
        self._trade_id = trade_id
        self._events: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return sorted(self._events, key=lambda x: x.get("date", ""))

    def maturity_alert(self, as_of: date, alert_days: int = 30) -> dict | None:
        maturity = self._instrument.bond.maturity
        days = (maturity - as_of).days
        if 0 < days <= alert_days:
            return {
                "type": ASWEventType.BOND_MATURITY,
                "date": maturity.isoformat(),
                "days_remaining": days,
            }
        return None

    def record_coupon(self, coupon_date: date, amount: float) -> dict:
        event = {
            "type": ASWEventType.COUPON,
            "date": coupon_date.isoformat(),
            "amount": amount,
        }
        self._events.append(event)
        return event

    def record_basis_alert(self, alert_date: date, basis_bp: float,
                            threshold_bp: float) -> dict:
        event = {
            "type": ASWEventType.BASIS_ALERT,
            "date": alert_date.isoformat(),
            "basis_bp": basis_bp,
            "threshold_bp": threshold_bp,
        }
        self._events.append(event)
        return event
