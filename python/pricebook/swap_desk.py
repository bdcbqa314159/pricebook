"""Swap trading desk: risk metrics, carry, P&L, book, dashboard, stress, XVA, capital.

Consolidates swap pricing (swap.py, swaption.py) into a unified desk layer
matching trs_desk.py / cln_desk.py / bond_trading_desk.py pattern.

    from pricebook.swap_desk import (
        swap_risk_metrics, SwapRiskMetrics,
        SwapBook, SwapBookEntry,
        swap_carry_decomposition, SwapCarryDecomposition,
        swap_daily_pnl, SwapDailyPnL,
        swap_dashboard, SwapDashboard,
        swap_stress_suite, swap_scenario_stress,
        swap_capital, SwapCapitalResult,
        swap_hedge_recommendations, SwapHedgeRecommendation,
        SwapLifecycle,
    )

Known out-of-scope:
- Real-time curve streaming: pricing library, not trading system.
- Compression/netting optimisation: separate workflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
from pricebook.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class SwapRiskMetrics:
    """Unified risk metrics for an IRS position."""
    pv: float
    par_rate: float
    annuity: float
    dv01: float                     # centred parallel DV01
    key_rate_dv01: dict[str, float] # per-pillar
    gamma: float                    # d²PV/dr²
    theta: float                    # 1-day rolldown
    notional: float
    direction: str

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "par_rate": self.par_rate, "annuity": self.annuity,
            "dv01": self.dv01, "key_rate_dv01": self.key_rate_dv01,
            "gamma": self.gamma, "theta": self.theta,
            "notional": self.notional, "direction": self.direction,
        }


def swap_risk_metrics(
    swap: InterestRateSwap,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
) -> SwapRiskMetrics:
    """Compute unified risk metrics for an IRS via bump-and-reprice."""
    proj = projection_curve or curve
    base_pv = swap.pv(curve, proj)
    par = swap.par_rate(curve, proj)
    ann = swap.annuity(curve)

    h = 0.0001  # 1bp

    # Centred DV01 (bump both discount and projection)
    pv_up = swap.pv(curve.bumped(h), proj.bumped(h) if proj is not curve else None)
    pv_dn = swap.pv(curve.bumped(-h), proj.bumped(-h) if proj is not curve else None)
    dv01 = (pv_up - pv_dn) / 2

    # Gamma: d²PV/dr²
    gamma = (pv_up - 2 * base_pv + pv_dn) / (h ** 2)

    # Key-rate DV01: per-pillar sensitivity
    # For single-curve (proj is curve): bump the shared curve at each pillar
    # For dual-curve: bump both discount and projection at each pillar
    pillar_times = [t for t in curve.pillar_times if t > 0]
    n_pillars = len(pillar_times)
    key_rate = {}
    for i in range(n_pillars):
        bumped_disc = curve.bumped_at(i, h)
        if proj is curve:
            # Single-curve: bumped disc IS the bumped projection
            kr = swap.pv(bumped_disc, None) - base_pv
        else:
            # Dual-curve: bump projection at same pillar index if it has enough pillars
            proj_times = [t2 for t2 in proj.pillar_times if t2 > 0]
            if i < len(proj_times):
                bumped_proj = proj.bumped_at(i, h)
            else:
                bumped_proj = proj
            kr = swap.pv(bumped_disc, bumped_proj) - base_pv
        label = _time_to_tenor(pillar_times[i])
        key_rate[label] = kr

    # Theta: rolldown via curve roll_down
    from pricebook.pnl_explain import compute_rolldown
    theta = compute_rolldown(
        lambda c: swap.pv(c, proj if proj is curve else proj), curve, days=1,
    )

    return SwapRiskMetrics(
        pv=base_pv, par_rate=par, annuity=ann,
        dv01=dv01, key_rate_dv01=key_rate,
        gamma=gamma, theta=theta,
        notional=swap.notional, direction=swap.direction.value,
    )


def _time_to_tenor(t: float) -> str:
    if t < 1/12:
        return f"{int(t*52)}W"
    if t < 1.0:
        return f"{int(t*12)}M"
    return f"{int(round(t))}Y"


# ---------------------------------------------------------------------------
# Swap Book
# ---------------------------------------------------------------------------

@dataclass
class SwapBookEntry:
    """A swap position in the book."""
    trade_id: str
    swap: InterestRateSwap
    counterparty: str = ""
    currency: str = "USD"


class SwapBook:
    """Collection of swap positions with aggregation."""

    def __init__(self, name: str = "swap_book"):
        self.name = name
        self._entries: list[SwapBookEntry] = []

    def add(self, entry: SwapBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[SwapBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_notional(self) -> float:
        return sum(e.swap.notional for e in self._entries)

    def by_direction(self) -> dict[str, list[SwapBookEntry]]:
        result: dict[str, list[SwapBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.swap.direction.value, []).append(e)
        return result

    def by_counterparty(self) -> dict[str, list[SwapBookEntry]]:
        result: dict[str, list[SwapBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.counterparty, []).append(e)
        return result

    def by_tenor(self, ref: date) -> dict[str, list[SwapBookEntry]]:
        """Bucket by remaining maturity."""
        result: dict[str, list[SwapBookEntry]] = {}
        for e in self._entries:
            T = year_fraction(ref, e.swap.end, DayCountConvention.ACT_365_FIXED)
            label = _time_to_tenor(T)
            result.setdefault(label, []).append(e)
        return result

    def aggregate_risk(
        self, curve: DiscountCurve, projection: DiscountCurve | None = None,
    ) -> dict[str, float]:
        total_pv = 0.0
        total_dv01 = 0.0
        total_gamma = 0.0
        total_notional = 0.0
        payer_dv01 = 0.0
        receiver_dv01 = 0.0

        for e in self._entries:
            rm = swap_risk_metrics(e.swap, curve, projection)
            total_pv += rm.pv
            total_dv01 += rm.dv01
            total_gamma += rm.gamma
            total_notional += rm.notional
            if e.swap.direction == SwapDirection.PAYER:
                payer_dv01 += rm.dv01
            else:
                receiver_dv01 += rm.dv01

        return {
            "total_pv": total_pv,
            "total_dv01": total_dv01,
            "net_dv01": payer_dv01 + receiver_dv01,
            "payer_dv01": payer_dv01,
            "receiver_dv01": receiver_dv01,
            "total_gamma": total_gamma,
            "n_positions": len(self._entries),
            "total_notional": total_notional,
        }


# ---------------------------------------------------------------------------
# Carry decomposition
# ---------------------------------------------------------------------------

@dataclass
class SwapCarryDecomposition:
    """Prospective carry for a swap position."""
    fixed_accrual: float      # fixed rate × notional × dt
    floating_accrual: float   # forward rate × notional × dt
    net_carry: float          # what the position earns per day
    direction: str

    def to_dict(self) -> dict:
        return {
            "fixed": self.fixed_accrual, "floating": self.floating_accrual,
            "net": self.net_carry, "direction": self.direction,
        }


def swap_carry_decomposition(
    swap: InterestRateSwap,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
    horizon_days: int = 1,
) -> SwapCarryDecomposition:
    """Decompose swap carry into fixed and floating accrual."""
    proj = projection_curve or curve
    dt = horizon_days / 365.0

    fixed = swap.fixed_rate * swap.notional * dt
    # Floating: use short-end forward rate as proxy for next reset
    fwd = proj.forward_rate(swap.start, swap.end) if swap.start < swap.end else 0.0
    floating = (fwd + swap.spread) * swap.notional * dt

    if swap.direction == SwapDirection.PAYER:
        net = floating - fixed  # pay fixed, receive floating
    else:
        net = fixed - floating  # receive fixed, pay floating

    return SwapCarryDecomposition(
        fixed_accrual=fixed, floating_accrual=floating,
        net_carry=net, direction=swap.direction.value,
    )


# ---------------------------------------------------------------------------
# Daily P&L
# ---------------------------------------------------------------------------

@dataclass
class SwapDailyPnL:
    """Daily P&L decomposition."""
    date: date
    total: float
    curve_pnl: float
    carry: float
    theta: float
    unexplained: float

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "total": self.total,
            "curve": self.curve_pnl, "carry": self.carry,
            "theta": self.theta, "unexplained": self.unexplained,
        }


def swap_daily_pnl(
    swap: InterestRateSwap,
    curve_t0: DiscountCurve,
    curve_t1: DiscountCurve,
    date_t1: date,
    projection_t0: DiscountCurve | None = None,
    projection_t1: DiscountCurve | None = None,
) -> SwapDailyPnL:
    """Daily P&L with curve/carry/theta attribution."""
    from pricebook.pnl_explain import compute_rolldown

    proj_t0 = projection_t0 or curve_t0
    proj_t1 = projection_t1 or curve_t1

    pv_t0 = swap.pv(curve_t0, proj_t0)
    pv_t1 = swap.pv(curve_t1, proj_t1)
    total = pv_t1 - pv_t0

    # Carry: 1-day accrual
    carry_d = swap_carry_decomposition(swap, curve_t0, proj_t0, horizon_days=1)
    carry = carry_d.net_carry

    # Theta: rolldown on unchanged curve
    theta = compute_rolldown(
        lambda c: swap.pv(c, proj_t0 if proj_t0 is curve_t0 else proj_t0),
        curve_t0, days=1,
    )

    # Curve P&L: total - carry - theta
    curve_pnl = total - carry - theta
    unexplained = 0.0  # fully attributed

    return SwapDailyPnL(date_t1, total, curve_pnl, carry, theta, unexplained)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class SwapDashboard:
    """Morning-meeting summary for the swap desk."""
    date: date
    n_positions: int
    total_notional: float
    total_pv: float
    total_dv01: float
    net_dv01: float
    total_gamma: float
    dv01_ladder: dict[str, float]  # per-tenor DV01
    by_direction: dict[str, int]
    by_counterparty: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "notional": self.total_notional, "pv": self.total_pv,
            "dv01": self.total_dv01, "net_dv01": self.net_dv01,
            "gamma": self.total_gamma, "dv01_ladder": self.dv01_ladder,
            "by_direction": self.by_direction, "by_counterparty": self.by_counterparty,
        }


def swap_dashboard(
    book: SwapBook,
    reference_date: date,
    curve: DiscountCurve,
    projection: DiscountCurve | None = None,
) -> SwapDashboard:
    """Build swap desk morning dashboard with DV01 ladder."""
    risk = book.aggregate_risk(curve, projection)

    # Build DV01 ladder by tenor bucket
    dv01_ladder: dict[str, float] = {}
    for e in book.entries:
        rm = swap_risk_metrics(e.swap, curve, projection)
        T = year_fraction(reference_date, e.swap.end, DayCountConvention.ACT_365_FIXED)
        label = _time_to_tenor(T)
        dv01_ladder[label] = dv01_ladder.get(label, 0.0) + rm.dv01

    by_dir = {k: len(v) for k, v in book.by_direction().items()}
    by_cpty = {k: len(v) for k, v in book.by_counterparty().items()}

    return SwapDashboard(
        date=reference_date, n_positions=risk["n_positions"],
        total_notional=risk["total_notional"], total_pv=risk["total_pv"],
        total_dv01=risk["total_dv01"], net_dv01=risk["net_dv01"],
        total_gamma=risk["total_gamma"], dv01_ladder=dv01_ladder,
        by_direction=by_dir, by_counterparty=by_cpty,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class SwapStressResult:
    """One stress scenario result."""
    scenario: str
    description: str
    pnl: float

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "description": self.description, "pnl": self.pnl}


def swap_stress_suite(
    book: SwapBook,
    curve: DiscountCurve,
    projection: DiscountCurve | None = None,
) -> list[SwapStressResult]:
    """Five parametric stress scenarios."""
    risk = book.aggregate_risk(curve, projection)
    dv01 = risk["total_dv01"]
    gamma = risk["total_gamma"]

    scenarios = [
        ("rates_up_100", "Parallel +100bp", 100.0),
        ("rates_dn_100", "Parallel -100bp", -100.0),
        ("rates_up_200", "Parallel +200bp", 200.0),
        ("rates_dn_200", "Parallel -200bp", -200.0),
        ("rates_up_50", "Parallel +50bp", 50.0),
    ]

    results = []
    for name, desc, bp in scenarios:
        pnl = dv01 * bp + 0.5 * gamma * (bp * h_bp) ** 2
        results.append(SwapStressResult(name, desc, pnl))

    return results


# Avoid computing h_bp in the loop
h_bp = 0.0001  # 1bp in rate terms


def swap_scenario_stress(
    book: SwapBook,
    ctx: PricingContext,
    scenarios: list | None = None,
) -> list:
    """Full-reprice stress via scenario.py."""
    from pricebook.scenario import parallel_shift, run_scenarios
    from pricebook.trade import Trade, Portfolio

    portfolio = Portfolio(name=book.name)
    for e in book.entries:
        portfolio.add(Trade(instrument=e.swap, trade_id=e.trade_id))

    if scenarios is None:
        scenarios = [
            parallel_shift(0.01, "rates_+100bp"),
            parallel_shift(-0.01, "rates_-100bp"),
            parallel_shift(0.02, "rates_+200bp"),
        ]

    return run_scenarios(portfolio, ctx, scenarios)


# ---------------------------------------------------------------------------
# XVA + Capital
# ---------------------------------------------------------------------------

@dataclass
class SwapCapitalResult:
    """SA-CCR capital for a swap position."""
    ead: float
    rwa: float
    capital: float
    simm_im: float

    def to_dict(self) -> dict:
        return {"ead": self.ead, "rwa": self.rwa, "capital": self.capital, "simm_im": self.simm_im}


def swap_capital(
    swap: InterestRateSwap,
    curve: DiscountCurve,
    projection: DiscountCurve | None = None,
    counterparty_rw: float = 0.20,
    hurdle_rate: float = 0.10,
) -> SwapCapitalResult:
    """SA-CCR capital for an IRS. SF=0.005 for rates."""
    T = year_fraction(swap.start, swap.end, DayCountConvention.ACT_365_FIXED)
    pv = swap.pv(curve, projection)
    mtm = max(pv, 0)

    sf = 0.005  # IR supervisory factor
    mf = math.sqrt(min(T, 1.0))
    ead = 1.4 * (mtm + swap.notional * sf * mf)
    rwa = ead * counterparty_rw
    capital = rwa * 0.08

    # SIMM: wire DV01 to GIRR
    from pricebook.simm import SIMMCalculator, SIMMSensitivity
    rm = swap_risk_metrics(swap, curve, projection)
    simm_inputs = []
    for tenor, kr_dv01 in rm.key_rate_dv01.items():
        simm_inputs.append(SIMMSensitivity(
            risk_class="GIRR", bucket="USD", tenor=tenor, delta=kr_dv01))
    simm_im = SIMMCalculator().compute(simm_inputs).total_margin if simm_inputs else 0.0

    return SwapCapitalResult(ead=ead, rwa=rwa, capital=capital, simm_im=simm_im)


def swap_mc_xva(
    swap: InterestRateSwap,
    ctx: PricingContext,
    cpty_survival,
    own_survival,
    cpty_recovery: float = 0.4,
    own_recovery: float = 0.4,
    funding_spread: float = 0.005,
    n_paths: int = 1000,
    n_steps: int = 12,
    rate_vol: float = 0.01,
    seed: int = 42,
):
    """MC XVA for a swap — wires xva.simulate_exposures."""
    import numpy as np
    from pricebook.xva import (
        simulate_exposures, expected_positive_exposure,
        expected_negative_exposure, total_xva_decomposition,
    )

    T = year_fraction(swap.start, swap.end, DayCountConvention.ACT_365_FIXED)
    time_grid = [(i + 1) * T / n_steps for i in range(n_steps)]

    pricer = lambda c: swap.pv_ctx(c)
    pvs = simulate_exposures(pricer, ctx, time_grid, n_paths, rate_vol, seed)
    epe = expected_positive_exposure(pvs)
    ene = expected_negative_exposure(pvs)

    return total_xva_decomposition(
        epe=epe, ene=ene, time_grid=time_grid,
        discount_curve=ctx.discount_curve,
        cpty_survival=cpty_survival, own_survival=own_survival,
        cpty_recovery=cpty_recovery, own_recovery=own_recovery,
        funding_spread=funding_spread,
    )


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class SwapHedgeRecommendation:
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


def swap_hedge_recommendations(
    book: SwapBook,
    curve: DiscountCurve,
    projection: DiscountCurve | None = None,
    dv01_limit: float = 50_000,
    gamma_limit: float = 1e9,
    net_dv01_limit: float = 20_000,
) -> list[SwapHedgeRecommendation]:
    """Hedge recommendations when risk exceeds limits."""
    risk = book.aggregate_risk(curve, projection)
    recs = []

    checks = [
        ("dv01", abs(risk["total_dv01"]), dv01_limit,
         "Enter offsetting swap to reduce parallel rate exposure"),
        ("net_dv01", abs(risk["net_dv01"]), net_dv01_limit,
         "Net DV01 imbalanced — add payer/receiver to flatten"),
        ("gamma", abs(risk["total_gamma"]), gamma_limit,
         "Buy swaption straddle to reduce gamma exposure"),
    ]

    for risk_type, current, limit, action in checks:
        if limit > 0 and current > limit * 0.75:
            recs.append(SwapHedgeRecommendation(
                risk_type=risk_type, current=current, limit=limit,
                breach_pct=current / limit, action=action,
            ))

    return recs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class SwapEventType:
    RESET = "reset"
    MATURITY = "maturity"
    NOVATION = "novation"


class SwapLifecycle:
    """Lifecycle management for a swap position."""

    def __init__(self, swap: InterestRateSwap, trade_id: str = "",
                 creation_date: date | None = None):
        from pricebook.trade import Trade
        from pricebook.trade_lifecycle import ManagedTrade

        self._swap = swap
        self._trade_id = trade_id
        trade = Trade(instrument=swap, trade_id=trade_id)
        self._managed = ManagedTrade(trade, creation_date or swap.start)
        self._events: list[dict] = []

    @property
    def swap(self) -> InterestRateSwap:
        return self._swap

    @property
    def history(self) -> list[dict]:
        base = [
            {"type": e.event_type.value, "date": e.event_date.isoformat(),
             "version": e.version, **e.details}
            for e in self._managed.history
        ]
        return sorted(base + self._events, key=lambda x: x.get("date", ""))

    def upcoming_resets(self, as_of: date, horizon_days: int = 90) -> list[dict]:
        """List upcoming floating rate reset dates."""
        horizon = as_of + timedelta(days=horizon_days)
        events = []
        for cf in self._swap.floating_leg.cashflows:
            if as_of < cf.accrual_start <= horizon:
                events.append({
                    "type": SwapEventType.RESET,
                    "date": cf.accrual_start.isoformat(),
                    "accrual_end": cf.accrual_end.isoformat(),
                })
        return events

    def maturity_alert(self, as_of: date, alert_days: int = 30) -> dict | None:
        days_to_mat = (self._swap.end - as_of).days
        if 0 < days_to_mat <= alert_days:
            return {
                "type": "maturity_alert",
                "date": self._swap.end.isoformat(),
                "days_remaining": days_to_mat,
            }
        return None

    def record_novation(self, novation_date: date, new_counterparty: str) -> dict:
        """Record novation to a new counterparty."""
        event = {
            "type": SwapEventType.NOVATION,
            "date": novation_date.isoformat(),
            "new_counterparty": new_counterparty,
        }
        self._events.append(event)
        return event
