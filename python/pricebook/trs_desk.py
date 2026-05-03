"""TRS trading desk: risk metrics, carry, P&L attribution, book management.

Same depth as repo_desk.py — production desk infrastructure for TRS.

    from pricebook.trs_desk import (
        trs_risk_metrics, trs_carry_decomposition,
        trs_daily_pnl, TRSBook, trs_dashboard,
        trs_all_in_cost, trs_stress_suite,
        trs_capital_summary, TRSCapitalSummary,
        trs_hedge_recommendations, TRSHedgeRecommendation,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.trs import TotalReturnSwap, TRSResult, FundingLegSpec
from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
from pricebook.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class TRSRiskMetrics:
    """Complete risk metrics for a TRS position."""
    pv: float
    delta: float            # dPV/dS (equity) or dPV/dy (bond)
    gamma: float            # d²PV/dS² or d²PV/dy²
    dv01: float             # parallel rate shift
    funding_dv01: float     # funding spread shift
    repo_dv01: float        # repo spread shift
    vega: float             # vol sensitivity (equity only)
    cs01: float             # credit spread sensitivity
    notional: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "delta": self.delta, "gamma": self.gamma,
            "dv01": self.dv01, "funding_dv01": self.funding_dv01,
            "repo_dv01": self.repo_dv01, "vega": self.vega,
            "cs01": self.cs01, "notional": self.notional,
        }


def trs_risk_metrics(
    trs: TotalReturnSwap,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
) -> TRSRiskMetrics:
    """Compute all risk metrics for a TRS via bump-and-reprice."""
    base = trs.price(curve, projection_curve)
    h = 0.0001  # 1bp

    # DV01: parallel OIS shift
    pv_up = trs.price(curve.bumped(h), projection_curve).value
    dv01 = pv_up - base.value

    # Funding DV01: bump funding spread
    old_spread = trs.funding.spread
    trs.funding = FundingLegSpec(
        spread=old_spread + h,
        day_count=trs.funding.day_count,
        compounding=trs.funding.compounding,
    )
    pv_fund_up = trs.price(curve, projection_curve).value
    trs.funding = FundingLegSpec(
        spread=old_spread,
        day_count=trs.funding.day_count,
        compounding=trs.funding.compounding,
    )
    funding_dv01 = pv_fund_up - base.value

    # Repo DV01: bump repo spread
    old_repo = trs.repo_spread
    trs.repo_spread = old_repo + h
    pv_repo_up = trs.price(curve, projection_curve).value
    trs.repo_spread = old_repo
    repo_dv01 = pv_repo_up - base.value

    # Delta + Gamma
    delta = 0.0
    gamma = 0.0
    if trs._underlying_type == "equity":
        old_spot = float(trs.underlying)
        bump = old_spot * 0.01  # 1% of spot
        trs.underlying = old_spot + bump
        pv_up_s = trs.price(curve, projection_curve).value
        trs.underlying = old_spot - bump
        pv_dn_s = trs.price(curve, projection_curve).value
        trs.underlying = old_spot
        # Delta per unit spot move (not per %)
        delta = (pv_up_s - pv_dn_s) / (2 * bump)
        gamma = (pv_up_s - 2 * base.value + pv_dn_s) / (bump ** 2)

    # Vega (equity only)
    vega = 0.0
    if trs._underlying_type == "equity" and trs.sigma > 0:
        old_sigma = trs.sigma
        trs.sigma = old_sigma + 0.01
        pv_vol_up = trs.price(curve, projection_curve).value
        trs.sigma = old_sigma
        vega = (pv_vol_up - base.value)

    # CS01: credit spread sensitivity (bond/loan/CLN only)
    cs01 = 0.0
    if trs.survival_curve is not None:
        from pricebook.credit_risk import _bump_survival_curve
        bumped_sc = _bump_survival_curve(trs.survival_curve, h)
        old_sc = trs.survival_curve
        trs.survival_curve = bumped_sc
        pv_cs_up = trs.price(curve, projection_curve).value
        trs.survival_curve = old_sc
        cs01 = pv_cs_up - base.value

    return TRSRiskMetrics(
        pv=base.value, delta=delta, gamma=gamma,
        dv01=dv01, funding_dv01=funding_dv01, repo_dv01=repo_dv01,
        vega=vega, cs01=cs01, notional=trs.notional,
    )


# ---------------------------------------------------------------------------
# Carry decomposition
# ---------------------------------------------------------------------------

@dataclass
class TRSCarryDecomposition:
    """Carry P&L split for a TRS position."""
    total_carry: float
    income: float          # coupon / dividend income
    funding_cost: float    # funding leg cost
    repo_cost: float       # repo financing (FVA)
    net_carry: float       # income - funding - repo

    def to_dict(self) -> dict:
        return {
            "total": self.total_carry, "income": self.income,
            "funding": self.funding_cost, "repo": self.repo_cost,
            "net": self.net_carry,
        }


def trs_carry_decomposition(
    trs: TotalReturnSwap,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
) -> TRSCarryDecomposition:
    """Decompose TRS carry into income, funding, and repo components."""
    result = trs.price(curve, projection_curve)

    income = result.income_return
    funding = result.funding_leg
    repo = result.fva
    net = income - funding - repo

    return TRSCarryDecomposition(
        total_carry=result.value,
        income=income, funding_cost=funding,
        repo_cost=repo, net_carry=net,
    )


# ---------------------------------------------------------------------------
# Daily P&L attribution
# ---------------------------------------------------------------------------

@dataclass
class TRSDailyPnL:
    """Daily P&L decomposition with full Greek attribution."""
    date: date
    total: float
    delta_pnl: float
    gamma_pnl: float
    carry_pnl: float
    theta_pnl: float
    funding_pnl: float
    spread_pnl: float
    vega_pnl: float
    unexplained: float

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "total": self.total,
            "delta": self.delta_pnl, "gamma": self.gamma_pnl,
            "carry": self.carry_pnl, "theta": self.theta_pnl,
            "funding": self.funding_pnl, "spread": self.spread_pnl,
            "vega": self.vega_pnl, "unexplained": self.unexplained,
        }


def trs_daily_pnl(
    trs: TotalReturnSwap,
    curve_t0: DiscountCurve,
    curve_t1: DiscountCurve,
    date_t1: date,
    projection_curve: DiscountCurve | None = None,
    spot_t0: float | None = None,
    spot_t1: float | None = None,
    vol_change: float = 0.0,
    spread_change: float = 0.0,
) -> TRSDailyPnL:
    """Daily P&L with full Greek attribution.

    Components:
    - delta: first-order price sensitivity × underlying move
    - gamma: 0.5 × gamma × (underlying move)²
    - carry: 1-day income accrual
    - theta: time-decay (PV loss from 1 day passing, same curves)
    - funding: funding leg change
    - spread: CS01 × spread_change
    - vega: vega × vol_change
    - unexplained: total minus all explained
    """
    from pricebook.pnl_explain import greek_pnl

    pv_t0 = trs.price(curve_t0, projection_curve).value
    pv_t1 = trs.price(curve_t1, projection_curve).value
    total = pv_t1 - pv_t0

    # Risk metrics at t0
    rm = trs_risk_metrics(trs, curve_t0, projection_curve)

    # Carry: 1-day accrual
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    carry = trs.price(curve_t0, projection_curve).income_return / max(T * 365, 1)

    # Delta + Gamma P&L
    delta_pnl = 0.0
    gamma_pnl = 0.0
    if trs._underlying_type == "equity" and spot_t0 is not None and spot_t1 is not None:
        dx = spot_t1 - spot_t0
        delta_pnl = rm.delta * dx
        gamma_pnl = 0.5 * rm.gamma * dx ** 2
    else:
        # Rate-driven: use DV01 × rate change
        rate_change = 0.0
        if hasattr(curve_t0, '_rates') and hasattr(curve_t1, '_rates'):
            r0 = curve_t0._rates[0] if curve_t0._rates else 0
            r1 = curve_t1._rates[0] if curve_t1._rates else 0
            rate_change = r1 - r0
        delta_pnl = greek_pnl(rm.dv01, rate_change * 10_000)  # DV01 is per 1bp

    # Theta: time decay (advance 1 day, same curve)
    theta_pnl = 0.0
    old_start = trs.start
    trs.start = old_start + timedelta(days=1)
    try:
        pv_t0_shifted = trs.price(curve_t0, projection_curve).value
    except (ValueError, ZeroDivisionError):
        pv_t0_shifted = pv_t0
    trs.start = old_start
    theta_pnl = pv_t0_shifted - pv_t0

    # Funding P&L
    funding_t0 = trs.price(curve_t0, projection_curve).funding_leg
    funding_t1 = trs.price(curve_t1, projection_curve).funding_leg
    funding_pnl = -(funding_t1 - funding_t0)

    # Spread P&L
    spread_pnl = rm.cs01 * spread_change / 0.0001 if spread_change != 0 else 0.0

    # Vega P&L
    vega_pnl = rm.vega * vol_change if vol_change != 0 else 0.0

    explained = delta_pnl + gamma_pnl + carry + theta_pnl + funding_pnl + spread_pnl + vega_pnl
    unexplained = total - explained

    return TRSDailyPnL(
        date_t1, total, delta_pnl, gamma_pnl, carry, theta_pnl,
        funding_pnl, spread_pnl, vega_pnl, unexplained,
    )


# ---------------------------------------------------------------------------
# TRS Book
# ---------------------------------------------------------------------------

@dataclass
class TRSBookEntry:
    """A TRS position in the book."""
    trade_id: str
    trs: TotalReturnSwap
    counterparty: str = ""
    underlying_type: str = ""
    independent_amount: float = 0.0  # initial margin / IA

    def __post_init__(self):
        if not self.underlying_type:
            self.underlying_type = self.trs._underlying_type


class TRSBook:
    """Collection of TRS positions with aggregation."""

    def __init__(self, name: str = "trs_book"):
        self.name = name
        self._entries: list[TRSBookEntry] = []

    def add(self, entry: TRSBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[TRSBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_pv(self, curve: DiscountCurve, projection: DiscountCurve | None = None) -> float:
        return sum(e.trs.price(curve, projection).value for e in self._entries)

    def total_notional(self) -> float:
        return sum(e.trs.notional for e in self._entries)

    def total_independent_amount(self) -> float:
        """Total Independent Amount (IA) posted across all trades."""
        return sum(e.independent_amount for e in self._entries)

    def by_type(self) -> dict[str, list[TRSBookEntry]]:
        result: dict[str, list[TRSBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.underlying_type, []).append(e)
        return result

    def by_counterparty(self) -> dict[str, list[TRSBookEntry]]:
        result: dict[str, list[TRSBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.counterparty, []).append(e)
        return result

    def aggregate_risk(
        self, curve: DiscountCurve, projection: DiscountCurve | None = None,
    ) -> dict[str, float]:
        """Aggregate risk metrics across the book."""
        total_pv = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_dv01 = 0.0
        total_funding_dv01 = 0.0
        total_vega = 0.0

        for e in self._entries:
            rm = trs_risk_metrics(e.trs, curve, projection)
            total_pv += rm.pv
            total_delta += rm.delta
            total_gamma += rm.gamma
            total_dv01 += rm.dv01
            total_funding_dv01 += rm.funding_dv01
            total_vega += rm.vega

        return {
            "total_pv": total_pv,
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_dv01": total_dv01,
            "total_funding_dv01": total_funding_dv01,
            "total_vega": total_vega,
            "n_positions": len(self._entries),
            "total_notional": self.total_notional(),
            "total_ia": self.total_independent_amount(),
        }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class TRSDashboard:
    """Morning-meeting summary for the TRS desk."""
    date: date
    n_positions: int
    total_pv: float
    total_notional: float
    total_delta: float
    total_dv01: float
    total_funding_dv01: float
    total_vega: float
    total_ia: float
    by_type: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "pv": self.total_pv, "notional": self.total_notional,
            "delta": self.total_delta, "dv01": self.total_dv01,
            "funding_dv01": self.total_funding_dv01, "vega": self.total_vega,
            "ia": self.total_ia, "by_type": self.by_type,
        }


def trs_dashboard(
    book: TRSBook,
    reference_date: date,
    curve: DiscountCurve,
    projection: DiscountCurve | None = None,
) -> TRSDashboard:
    """Build the TRS desk morning dashboard."""
    risk = book.aggregate_risk(curve, projection)
    by_type = {k: len(v) for k, v in book.by_type().items()}

    return TRSDashboard(
        date=reference_date,
        n_positions=risk["n_positions"],
        total_pv=risk["total_pv"],
        total_notional=risk["total_notional"],
        total_delta=risk["total_delta"],
        total_dv01=risk["total_dv01"],
        total_funding_dv01=risk["total_funding_dv01"],
        total_vega=risk["total_vega"],
        total_ia=risk["total_ia"],
        by_type=by_type,
    )


# ---------------------------------------------------------------------------
# TRS XVA: all-in cost
# ---------------------------------------------------------------------------

@dataclass
class TRSAllInCost:
    """True cost of a TRS beyond headline funding spread."""
    funding_cost: float
    fva: float
    kva: float
    mva: float
    total_cost: float
    headline_spread_bps: float
    all_in_spread_bps: float
    hidden_cost_bps: float

    def to_dict(self) -> dict:
        return {
            "funding": self.funding_cost, "fva": self.fva,
            "kva": self.kva, "mva": self.mva,
            "total": self.total_cost,
            "headline_bps": self.headline_spread_bps,
            "all_in_bps": self.all_in_spread_bps,
            "hidden_bps": self.hidden_cost_bps,
        }


def trs_all_in_cost(
    trs: TotalReturnSwap,
    curve: DiscountCurve,
    funding_spread: float = 0.002,
    capital_charge: float = 0.0,
    hurdle_rate: float = 0.10,
    initial_margin: float = 0.0,
) -> TRSAllInCost:
    """Total cost of a TRS including XVA.

    Shows the hidden cost in bps above the headline funding spread.
    """
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    result = trs.price(curve)

    funding = result.funding_leg
    fva = result.fva
    kva = capital_charge * hurdle_rate * T
    mva = initial_margin * funding_spread * T

    total = funding + fva + kva + mva

    # Convert to spread
    annuity = trs.notional * T
    headline = trs.funding.spread * 10_000
    all_in = total / annuity * 10_000 if annuity > 0 else headline
    hidden = all_in - headline

    return TRSAllInCost(
        funding_cost=funding, fva=fva, kva=kva, mva=mva,
        total_cost=total, headline_spread_bps=headline,
        all_in_spread_bps=all_in, hidden_cost_bps=hidden,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class TRSStressResult:
    """One stress scenario result."""
    scenario: str
    description: str
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    funding_pnl: float
    total_pnl: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario, "description": self.description,
            "delta": self.delta_pnl, "gamma": self.gamma_pnl,
            "vega": self.vega_pnl, "funding": self.funding_pnl,
            "total": self.total_pnl,
        }


def trs_stress_suite(
    book: TRSBook,
    curve: DiscountCurve,
    projection: DiscountCurve | None = None,
) -> list[TRSStressResult]:
    """Pre-built stress scenarios for the TRS book."""
    risk = book.aggregate_risk(curve, projection)
    delta = risk["total_delta"]
    gamma = risk["total_gamma"]
    vega = risk["total_vega"]
    fdv01 = risk["total_funding_dv01"]

    scenarios = [
        ("equity_crash", "Equity -20%", -0.20, 0.0, 0.0, 0.0),
        ("rates_up", "Rates +100bp", 0.0, 100.0, 0.0, 0.0),
        ("credit_wide", "Credit spreads +200bp", 0.0, 0.0, 0.0, 200.0),
        ("vol_spike", "Vol +10%", 0.0, 0.0, 10.0, 0.0),
        ("combined", "Equity -10%, rates +50bp, vol +5%", -0.10, 50.0, 5.0, 100.0),
    ]

    results = []
    for name, desc, eq_shock, rate_bp, vol_pct, fund_bp in scenarios:
        d_pnl = delta * eq_shock * risk["total_notional"] / risk["n_positions"] if risk["n_positions"] > 0 else 0
        g_pnl = 0.5 * gamma * (eq_shock * risk["total_notional"] / max(risk["n_positions"], 1)) ** 2
        v_pnl = vega * vol_pct
        f_pnl = fdv01 * fund_bp
        total = d_pnl + g_pnl + v_pnl + f_pnl

        results.append(TRSStressResult(name, desc, d_pnl, g_pnl, v_pnl, f_pnl, total))

    return results


# ---------------------------------------------------------------------------
# Scenario full-reprice stress (wires scenario.py)
# ---------------------------------------------------------------------------

def trs_scenario_stress(
    book: TRSBook,
    ctx: PricingContext,
    scenarios: list | None = None,
) -> list:
    """Full-reprice stress testing using scenario.py engine.

    Wraps TRS book into a Portfolio, then runs scenarios via
    run_scenarios() for exact PV recomputation (no Greek approx).
    """
    from pricebook.scenario import parallel_shift, run_scenarios, ScenarioResult
    from pricebook.trade import Trade, Portfolio

    portfolio = Portfolio(name=book.name)
    for e in book.entries:
        portfolio.add(Trade(instrument=e.trs, trade_id=e.trade_id))

    if scenarios is None:
        from pricebook.scenario import parallel_shift
        scenarios = [
            parallel_shift(0.01, "rates_+100bp"),
            parallel_shift(-0.01, "rates_-100bp"),
            parallel_shift(0.02, "rates_+200bp"),
            parallel_shift(-0.02, "rates_-200bp"),
        ]

    return run_scenarios(portfolio, ctx, scenarios)


def trs_dv01_ladder(
    book: TRSBook,
    ctx: PricingContext,
) -> list:
    """Per-pillar DV01 ladder using scenario.py pillar_bump."""
    from pricebook.scenario import pillar_bump, run_scenarios
    from pricebook.trade import Trade, Portfolio

    portfolio = Portfolio(name=book.name)
    for e in book.entries:
        portfolio.add(Trade(instrument=e.trs, trade_id=e.trade_id))

    n_pillars = len(ctx.discount_curve._times) if ctx.discount_curve else 0
    scenarios = [pillar_bump(i) for i in range(n_pillars)]
    return run_scenarios(portfolio, ctx, scenarios)


# ---------------------------------------------------------------------------
# Regulatory capital summary
# ---------------------------------------------------------------------------

# SA-CCR supervisory factors by asset class
_SA_CCR_SF = {"equity": 0.32, "bond": 0.005, "loan": 0.005, "cln": 0.005,
              "commodity": 0.18, "fx": 0.04}

# SA risk weights by counterparty type
_SA_RISK_WEIGHTS = {
    "sovereign": 0.0, "central_bank": 0.0, "bank": 0.20,
    "broker_dealer": 0.20, "corporate": 1.00, "hedge_fund": 1.50,
    "ccp": 0.02, "other": 1.00,
}


@dataclass
class TRSCapitalEntry:
    """Capital for one TRS position."""
    trade_id: str
    underlying_type: str
    notional: float
    mtm: float
    ead: float
    rwa: float
    capital: float
    simm_im: float


@dataclass
class TRSCapitalSummary:
    """Regulatory capital summary for the TRS book."""
    entries: list[TRSCapitalEntry]
    total_ead: float
    total_rwa: float
    total_capital: float
    total_simm_im: float
    total_notional: float

    def to_dict(self) -> dict:
        return {
            "n_positions": len(self.entries),
            "total_ead": self.total_ead,
            "total_rwa": self.total_rwa,
            "total_capital": self.total_capital,
            "total_simm_im": self.total_simm_im,
            "total_notional": self.total_notional,
            "by_trade": [
                {"id": e.trade_id, "type": e.underlying_type,
                 "notional": e.notional, "ead": e.ead,
                 "rwa": e.rwa, "capital": e.capital}
                for e in self.entries
            ],
        }


def trs_capital_summary(
    book: TRSBook,
    curve: DiscountCurve,
    counterparty_type: str = "corporate",
    projection: DiscountCurve | None = None,
) -> TRSCapitalSummary:
    """Compute SA-CCR based regulatory capital for the TRS book.

    EAD = 1.4 x (max(MTM, 0) + notional x SF x sqrt(min(T, 1)))
    RWA = EAD x RW
    Capital = RWA x 8%
    SIMM IM via ISDA SIMM v2.6 (trs_simm_im)
    """
    from pricebook.trs_xva import trs_simm_im

    rw = _SA_RISK_WEIGHTS.get(counterparty_type, 1.0)
    entries = []

    for e in book.entries:
        trs = e.trs
        T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
        result = trs.price(curve, projection)
        mtm = max(result.value, 0)

        sf = _SA_CCR_SF.get(trs._underlying_type, 0.10)
        ead = 1.4 * (mtm + trs.notional * sf * math.sqrt(min(T, 1.0)))
        rwa = ead * rw
        capital = rwa * 0.08
        simm_im = trs_simm_im(trs, curve, projection)

        entries.append(TRSCapitalEntry(
            trade_id=e.trade_id, underlying_type=trs._underlying_type,
            notional=trs.notional, mtm=mtm, ead=ead, rwa=rwa,
            capital=capital, simm_im=simm_im,
        ))

    return TRSCapitalSummary(
        entries=entries,
        total_ead=sum(e.ead for e in entries),
        total_rwa=sum(e.rwa for e in entries),
        total_capital=sum(e.capital for e in entries),
        total_simm_im=sum(e.simm_im for e in entries),
        total_notional=sum(e.notional for e in entries),
    )


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class TRSHedgeRecommendation:
    """A hedge recommendation for the TRS book."""
    risk_type: str      # "delta", "dv01", "vega", "funding"
    current: float      # current exposure
    limit: float        # limit
    breach_pct: float   # current / limit
    action: str         # suggested hedge action

    def to_dict(self) -> dict:
        return {
            "risk": self.risk_type, "current": self.current,
            "limit": self.limit, "breach_pct": self.breach_pct,
            "action": self.action,
        }


def trs_hedge_recommendations(
    book: TRSBook,
    curve: DiscountCurve,
    delta_limit: float = 1_000_000,
    dv01_limit: float = 50_000,
    vega_limit: float = 100_000,
    funding_dv01_limit: float = 10_000,
    projection: DiscountCurve | None = None,
) -> list[TRSHedgeRecommendation]:
    """Generate hedge recommendations when risk exceeds limits.

    Returns a list of recommendations for any breached limits.
    """
    risk = book.aggregate_risk(curve, projection)
    recs = []

    checks = [
        ("delta", abs(risk["total_delta"]), delta_limit,
         "Reduce equity delta via futures or offsetting TRS"),
        ("dv01", abs(risk["total_dv01"]), dv01_limit,
         "Hedge rate risk via interest rate swaps"),
        ("vega", abs(risk["total_vega"]), vega_limit,
         "Reduce vol exposure via variance swaps or options"),
        ("funding_dv01", abs(risk["total_funding_dv01"]), funding_dv01_limit,
         "Reduce funding exposure or term out funding"),
    ]

    for risk_type, current, limit, action in checks:
        if limit > 0 and current > limit * 0.75:  # warn at 75% of limit
            recs.append(TRSHedgeRecommendation(
                risk_type=risk_type, current=current, limit=limit,
                breach_pct=current / limit if limit > 0 else 0,
                action=action,
            ))

    return recs


# ---------------------------------------------------------------------------
# TRS Lifecycle (resets, margin calls, early termination)
# ---------------------------------------------------------------------------

class TRSEventType:
    RESET = "reset"
    MARGIN_CALL = "margin_call"
    EARLY_TERMINATION = "early_termination"


@dataclass
class TRSMarginCall:
    """Result of a margin call computation."""
    date: date
    mtm: float
    threshold: float
    required_transfer: float
    direction: str  # "receive" or "pay"

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "mtm": self.mtm,
            "threshold": self.threshold, "transfer": self.required_transfer,
            "direction": self.direction,
        }


class TRSLifecycle:
    """Lifecycle management for a TRS position.

    Wraps trade_lifecycle.ManagedTrade for versioned history + TRS-specific
    events (resets, margin calls, early termination).
    """

    def __init__(self, trs: TotalReturnSwap, trade_id: str = "", creation_date: date | None = None):
        from pricebook.trade import Trade
        from pricebook.trade_lifecycle import ManagedTrade

        self._trs = trs
        self._trade_id = trade_id
        trade = Trade(instrument=trs, trade_id=trade_id)
        self._managed = ManagedTrade(trade, creation_date or trs.start)
        self._events: list[dict] = []

    @property
    def trs(self) -> TotalReturnSwap:
        return self._trs

    @property
    def history(self) -> list[dict]:
        """Combined history: lifecycle events + TRS events."""
        base = [
            {"type": e.event_type.value, "date": e.event_date.isoformat(),
             "version": e.version, **e.details}
            for e in self._managed.history
        ]
        return sorted(base + self._events, key=lambda x: x.get("date", ""))

    def process_reset(
        self, reset_date: date, curve: DiscountCurve,
        new_price: float | None = None,
    ) -> float:
        """Process a periodic reset: update initial_price, record event.

        Returns the new initial price.
        """
        if new_price is None:
            result = self._trs.price(curve)
            if self._trs._underlying_type == "equity":
                new_price = float(self._trs.underlying)
            else:
                new_price = result.value / self._trs.notional * 100 + (self._trs.initial_price or 100)

        old_price = self._trs.initial_price
        self._trs.initial_price = new_price

        self._events.append({
            "type": TRSEventType.RESET,
            "date": reset_date.isoformat(),
            "old_price": old_price,
            "new_price": new_price,
        })

        # Record as amendment in managed trade
        self._managed.amend(reset_date, instrument=self._trs)

        return new_price

    def margin_call(
        self, as_of: date, curve: DiscountCurve,
        threshold: float = 0.0,
        min_transfer: float = 250_000,
    ) -> TRSMarginCall:
        """Compute margin call based on MTM vs threshold."""
        result = self._trs.price(curve)
        mtm = result.value
        net = abs(mtm) - threshold
        required = max(net, 0)
        direction = "receive" if mtm > 0 else "pay"

        if required < min_transfer:
            required = 0.0

        if required > 0:
            self._events.append({
                "type": TRSEventType.MARGIN_CALL,
                "date": as_of.isoformat(),
                "mtm": mtm,
                "transfer": required,
                "direction": direction,
            })

        return TRSMarginCall(as_of, mtm, threshold, required, direction)

    def early_terminate(self, as_of: date, curve: DiscountCurve) -> float:
        """Early termination: compute breakage cost (= current MTM)."""
        result = self._trs.price(curve)
        breakage = result.value

        self._events.append({
            "type": TRSEventType.EARLY_TERMINATION,
            "date": as_of.isoformat(),
            "breakage": breakage,
        })

        # Record as exercise in managed trade
        self._managed.exercise(as_of, underlying_instrument=self._trs)

        return breakage


# ---------------------------------------------------------------------------
# Collateral evolution
# ---------------------------------------------------------------------------

@dataclass
class CollateralState:
    """Collateral state at a point in time."""
    date: date
    mtm: float
    collateral_posted: float
    net_exposure: float
    margin_call: float

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "mtm": self.mtm,
            "collateral": self.collateral_posted,
            "net_exposure": self.net_exposure,
            "margin_call": self.margin_call,
        }


def trs_collateral_evolution(
    trs: TotalReturnSwap,
    dates: list[date],
    curves: list[DiscountCurve],
    initial_collateral: float = 0.0,
    threshold: float = 0.0,
    min_transfer: float = 250_000,
) -> list[CollateralState]:
    """Simulate collateral evolution over a series of dates.

    At each date: price TRS, compute net exposure, trigger margin
    call if exposure exceeds threshold and amount exceeds min_transfer.
    """
    if len(dates) != len(curves):
        raise ValueError("dates and curves must have same length")

    states = []
    collateral = initial_collateral

    for d, curve in zip(dates, curves):
        mtm = trs.price(curve).value
        net_exp = mtm - collateral
        call = 0.0

        if abs(net_exp) > threshold:
            call_amount = abs(net_exp) - threshold
            if call_amount >= min_transfer:
                call = call_amount if net_exp > 0 else -call_amount
                collateral += call

        states.append(CollateralState(d, mtm, collateral, mtm - collateral, call))

    return states
