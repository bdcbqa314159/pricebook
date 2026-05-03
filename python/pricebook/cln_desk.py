"""CLN trading desk: risk metrics, carry, P&L attribution, book management.

Same depth as trs_desk.py / repo_desk.py — production desk infrastructure
for Credit-Linked Notes.

    from pricebook.cln_desk import (
        cln_risk_metrics, CLNRiskMetrics,
        cln_carry_decomposition, CLNCarryDecomposition,
        cln_daily_pnl, CLNDailyPnL,
        CLNBook, CLNBookEntry,
        cln_dashboard, CLNDashboard,
    )

Known out-of-scope:
- Real-time Greeks refresh: pricing library, not trading system.
- Corporate actions on reference entity: edge case, per-issuer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.cln import CreditLinkedNote, CLNResult
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class CLNRiskMetrics:
    """Complete risk metrics for a CLN position."""
    pv: float
    dv01: float                    # parallel rate shift +1bp (centred)
    cs01: float                    # credit spread (hazard) shift +1bp
    recovery_sensitivity: float    # recovery +1%
    jump_to_default_pnl: float     # immediate default: R×N - PV
    notional: float
    leverage: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "dv01": self.dv01, "cs01": self.cs01,
            "recovery_sensitivity": self.recovery_sensitivity,
            "jtd": self.jump_to_default_pnl,
            "notional": self.notional, "leverage": self.leverage,
        }


def cln_risk_metrics(
    cln: CreditLinkedNote,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> CLNRiskMetrics:
    """Compute all risk metrics for a CLN via bump-and-reprice.

    DV01: centred difference on discount curve (O(h²)).
    CS01: +1bp hazard rate shift.
    Recovery: +1% recovery bump.
    JTD: loss on immediate default = R × N - PV.
    """
    base = cln.dirty_price(discount_curve, survival_curve)
    greeks = cln.greeks(discount_curve, survival_curve)

    # Centred DV01 (override one-sided from greeks)
    h = 0.0001
    pv_up = cln.dirty_price(discount_curve.bumped(h), survival_curve)
    pv_dn = cln.dirty_price(discount_curve.bumped(-h), survival_curve)
    dv01 = (pv_up - pv_dn) / 2

    # JTD: on immediate default, investor receives recovery × notional
    # but loses their investment (current PV)
    jtd = cln.recovery * cln.notional - base

    return CLNRiskMetrics(
        pv=base,
        dv01=dv01,
        cs01=greeks["cs01"],
        recovery_sensitivity=greeks["recovery_sensitivity"],
        jump_to_default_pnl=jtd,
        notional=cln.notional,
        leverage=cln.leverage,
    )


# ---------------------------------------------------------------------------
# Carry decomposition
# ---------------------------------------------------------------------------

@dataclass
class CLNCarryDecomposition:
    """Carry P&L split for a CLN position.

    CLN carry is fundamentally different from TRS:
    - Coupon income: periodic coupon accrual
    - Default drag: expected loss accrual = h × (1-R) × N per year
    - Funding cost: risk-free opportunity cost
    """
    total_carry: float
    coupon_income: float     # annualised coupon accrual
    default_drag: float      # -h × (1-R) × N (expected loss per year)
    funding_cost: float      # risk-free rate × N (opportunity cost)
    net_carry: float         # coupon - |default_drag| - funding

    def to_dict(self) -> dict:
        return {
            "total": self.total_carry, "coupon": self.coupon_income,
            "default_drag": self.default_drag, "funding": self.funding_cost,
            "net": self.net_carry,
        }


def cln_carry_decomposition(
    cln: CreditLinkedNote,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> CLNCarryDecomposition:
    """Decompose CLN carry into coupon, default drag, and funding."""
    T = year_fraction(cln.start, cln.end, DayCountConvention.ACT_365_FIXED)
    base = cln.dirty_price(discount_curve, survival_curve)

    # Annualised coupon income
    coupon = cln.coupon_rate * cln.notional

    # Default drag: hazard × (1-R) × notional per year
    base_hazard = -math.log(max(survival_curve.survival(cln.end), 1e-15)) / max(T, 1e-10)
    lgd = (1 - cln.recovery) * cln.leverage
    default_drag = -base_hazard * lgd * cln.notional

    # Funding cost: risk-free rate × notional
    r = -math.log(discount_curve.df(cln.end)) / max(T, 1e-10)
    funding = r * cln.notional

    net = coupon + default_drag - funding

    return CLNCarryDecomposition(
        total_carry=base, coupon_income=coupon,
        default_drag=default_drag, funding_cost=funding,
        net_carry=net,
    )


# ---------------------------------------------------------------------------
# Daily P&L attribution
# ---------------------------------------------------------------------------

@dataclass
class CLNDailyPnL:
    """Daily P&L decomposition with credit-specific attribution."""
    date: date
    total: float
    spread_pnl: float       # CS01 × spread change
    rate_pnl: float          # DV01 × rate change
    recovery_pnl: float      # recovery_sens × recovery change
    carry_pnl: float
    theta_pnl: float         # time decay via curve roll-down
    unexplained: float

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "total": self.total,
            "spread": self.spread_pnl, "rate": self.rate_pnl,
            "recovery": self.recovery_pnl, "carry": self.carry_pnl,
            "theta": self.theta_pnl, "unexplained": self.unexplained,
        }


def cln_daily_pnl(
    cln: CreditLinkedNote,
    curve_t0: DiscountCurve,
    curve_t1: DiscountCurve,
    surv_t0: SurvivalCurve,
    surv_t1: SurvivalCurve,
    date_t1: date,
    recovery_change: float = 0.0,
) -> CLNDailyPnL:
    """Daily P&L with credit-specific attribution.

    Spread change inferred from survival curve shift.
    Rate change inferred from discount factor shift.
    """
    from pricebook.pnl_explain import greek_pnl, compute_rolldown

    pv_t0 = cln.dirty_price(curve_t0, surv_t0)
    pv_t1 = cln.dirty_price(curve_t1, surv_t1)
    total = pv_t1 - pv_t0

    rm = cln_risk_metrics(cln, curve_t0, surv_t0)
    T = year_fraction(cln.start, cln.end, DayCountConvention.ACT_365_FIXED)

    # Carry: 1-day accrual
    carry = rm.notional * cln.coupon_rate / 365

    # Spread P&L: infer spread change from survival curves
    h0 = -math.log(max(surv_t0.survival(cln.end), 1e-15)) / max(T, 1e-10)
    h1 = -math.log(max(surv_t1.survival(cln.end), 1e-15)) / max(T, 1e-10)
    spread_change = h1 - h0
    spread_pnl = rm.cs01 * spread_change / 0.0001 if abs(spread_change) > 1e-12 else 0.0

    # Rate P&L: infer from DF change
    try:
        t1y = cln.start + timedelta(days=365)
        r0 = -math.log(curve_t0.df(t1y))
        r1 = -math.log(curve_t1.df(t1y))
        rate_change = r1 - r0
    except (ValueError, AttributeError):
        rate_change = 0.0
    rate_pnl = greek_pnl(rm.dv01, rate_change * 10_000)

    # Recovery P&L
    recovery_pnl = rm.recovery_sensitivity * recovery_change / 0.01 if recovery_change != 0 else 0.0

    # Theta via curve roll-down
    theta_pnl = compute_rolldown(
        lambda c: cln.dirty_price(c, surv_t0), curve_t0, days=1,
    )

    explained = spread_pnl + rate_pnl + recovery_pnl + carry + theta_pnl
    unexplained = total - explained

    return CLNDailyPnL(
        date_t1, total, spread_pnl, rate_pnl, recovery_pnl,
        carry, theta_pnl, unexplained,
    )


# ---------------------------------------------------------------------------
# CLN Book
# ---------------------------------------------------------------------------

@dataclass
class CLNBookEntry:
    """A CLN position in the book."""
    trade_id: str
    cln: CreditLinkedNote
    survival_curve: SurvivalCurve
    counterparty: str = ""
    issuer: str = ""
    seniority: str = "senior_unsecured"
    independent_amount: float = 0.0

    @property
    def underlying_type(self) -> str:
        return "leveraged" if self.cln.leverage > 1.0 else "vanilla"


class CLNBook:
    """Collection of CLN positions with aggregation."""

    def __init__(self, name: str = "cln_book"):
        self.name = name
        self._entries: list[CLNBookEntry] = []

    def add(self, entry: CLNBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[CLNBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_pv(self, curve: DiscountCurve) -> float:
        return sum(
            e.cln.dirty_price(curve, e.survival_curve) for e in self._entries
        )

    def total_notional(self) -> float:
        return sum(e.cln.notional for e in self._entries)

    def total_independent_amount(self) -> float:
        return sum(e.independent_amount for e in self._entries)

    def by_issuer(self) -> dict[str, list[CLNBookEntry]]:
        result: dict[str, list[CLNBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.issuer, []).append(e)
        return result

    def by_seniority(self) -> dict[str, list[CLNBookEntry]]:
        result: dict[str, list[CLNBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.seniority, []).append(e)
        return result

    def by_counterparty(self) -> dict[str, list[CLNBookEntry]]:
        result: dict[str, list[CLNBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.counterparty, []).append(e)
        return result

    def aggregate_risk(self, curve: DiscountCurve) -> dict[str, float]:
        """Aggregate risk metrics across the book."""
        total_pv = 0.0
        total_dv01 = 0.0
        total_cs01 = 0.0
        total_jtd = 0.0
        total_recovery_sens = 0.0

        for e in self._entries:
            rm = cln_risk_metrics(e.cln, curve, e.survival_curve)
            total_pv += rm.pv
            total_dv01 += rm.dv01
            total_cs01 += rm.cs01
            total_jtd += rm.jump_to_default_pnl
            total_recovery_sens += rm.recovery_sensitivity

        return {
            "total_pv": total_pv,
            "total_dv01": total_dv01,
            "total_cs01": total_cs01,
            "total_jtd": total_jtd,
            "total_recovery_sens": total_recovery_sens,
            "n_positions": len(self._entries),
            "total_notional": self.total_notional(),
            "total_ia": self.total_independent_amount(),
        }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class CLNDashboard:
    """Morning-meeting summary for the CLN desk."""
    date: date
    n_positions: int
    total_pv: float
    total_notional: float
    total_cs01: float
    total_dv01: float
    total_jtd: float
    total_recovery_sens: float
    total_ia: float
    by_issuer: dict[str, int]
    by_seniority: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "pv": self.total_pv, "notional": self.total_notional,
            "cs01": self.total_cs01, "dv01": self.total_dv01,
            "jtd": self.total_jtd, "recovery_sens": self.total_recovery_sens,
            "ia": self.total_ia, "by_issuer": self.by_issuer,
            "by_seniority": self.by_seniority,
        }


def cln_dashboard(
    book: CLNBook,
    reference_date: date,
    curve: DiscountCurve,
) -> CLNDashboard:
    """Build the CLN desk morning dashboard."""
    risk = book.aggregate_risk(curve)
    by_issuer = {k: len(v) for k, v in book.by_issuer().items()}
    by_seniority = {k: len(v) for k, v in book.by_seniority().items()}

    return CLNDashboard(
        date=reference_date,
        n_positions=risk["n_positions"],
        total_pv=risk["total_pv"],
        total_notional=risk["total_notional"],
        total_cs01=risk["total_cs01"],
        total_dv01=risk["total_dv01"],
        total_jtd=risk["total_jtd"],
        total_recovery_sens=risk["total_recovery_sens"],
        total_ia=risk["total_ia"],
        by_issuer=by_issuer,
        by_seniority=by_seniority,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class CLNStressResult:
    """One stress scenario result."""
    scenario: str
    description: str
    spread_pnl: float
    rate_pnl: float
    recovery_pnl: float
    total_pnl: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario, "description": self.description,
            "spread": self.spread_pnl, "rate": self.rate_pnl,
            "recovery": self.recovery_pnl, "total": self.total_pnl,
        }


def cln_stress_suite(
    book: CLNBook,
    curve: DiscountCurve,
) -> list[CLNStressResult]:
    """Five CLN-specific stress scenarios using parametric Greeks."""
    risk = book.aggregate_risk(curve)
    cs01 = risk["total_cs01"]
    dv01 = risk["total_dv01"]
    rec_sens = risk["total_recovery_sens"]

    scenarios = [
        ("spread_wide", "Spreads +200bp", 200.0, 0.0, 0.0),
        ("spread_tight", "Spreads -100bp", -100.0, 0.0, 0.0),
        ("recovery_down", "Recovery -20%", 0.0, 0.0, -0.20),
        ("rates_up", "Rates +100bp", 0.0, 100.0, 0.0),
        ("combined", "Spreads +200bp, recovery -20%, rates +100bp", 200.0, 100.0, -0.20),
    ]

    results = []
    for name, desc, spread_bp, rate_bp, rec_pct in scenarios:
        s_pnl = cs01 * spread_bp
        r_pnl = dv01 * rate_bp
        # Recovery sens is per +1% (0.01), so scale by pct/0.01
        rv_pnl = rec_sens * rec_pct / 0.01 if rec_pct != 0 else 0.0
        total = s_pnl + r_pnl + rv_pnl
        results.append(CLNStressResult(name, desc, s_pnl, r_pnl, rv_pnl, total))

    return results


def cln_scenario_stress(
    book: CLNBook,
    ctx: PricingContext,
    scenarios: list | None = None,
) -> list:
    """Full-reprice stress via scenario.py run_scenarios."""
    from pricebook.scenario import parallel_shift, run_scenarios
    from pricebook.trade import Trade, Portfolio
    from pricebook.pricing_context import PricingContext

    portfolio = Portfolio(name=book.name)
    for e in book.entries:
        portfolio.add(Trade(instrument=e.cln, trade_id=e.trade_id))

    if scenarios is None:
        scenarios = [
            parallel_shift(0.01, "rates_+100bp"),
            parallel_shift(-0.01, "rates_-100bp"),
            parallel_shift(0.02, "rates_+200bp"),
        ]

    return run_scenarios(portfolio, ctx, scenarios)


# ---------------------------------------------------------------------------
# Regulatory capital
# ---------------------------------------------------------------------------

@dataclass
class CLNCapitalEntry:
    """Capital for one CLN position."""
    trade_id: str
    issuer: str
    seniority: str
    notional: float
    mtm: float
    ead: float
    rwa: float
    capital: float
    simm_im: float


@dataclass
class CLNCapitalSummary:
    """Regulatory capital summary for the CLN book."""
    entries: list[CLNCapitalEntry]
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
        }


_CLN_SA_RISK_WEIGHTS = {
    "sovereign": 0.0, "bank": 0.20, "corporate": 1.00,
    "hedge_fund": 1.50, "other": 1.00,
}


def cln_capital_summary(
    book: CLNBook,
    curve: DiscountCurve,
    counterparty_type: str = "corporate",
) -> CLNCapitalSummary:
    """SA-CCR capital for CLN book. SF=0.005 for credit."""
    from pricebook.cln_xva import cln_simm_im

    rw = _CLN_SA_RISK_WEIGHTS.get(counterparty_type, 1.0)
    entries = []

    for e in book.entries:
        c = e.cln
        T = year_fraction(c.start, c.end, DayCountConvention.ACT_365_FIXED)
        pv = c.dirty_price(curve, e.survival_curve)
        mtm = max(pv, 0)

        sf = 0.005
        ead = 1.4 * (mtm + c.notional * sf * math.sqrt(min(T, 1.0)))
        rwa = ead * rw
        capital = rwa * 0.08
        simm_im = cln_simm_im(c, curve, e.survival_curve)

        entries.append(CLNCapitalEntry(
            trade_id=e.trade_id, issuer=e.issuer, seniority=e.seniority,
            notional=c.notional, mtm=mtm, ead=ead, rwa=rwa,
            capital=capital, simm_im=simm_im,
        ))

    return CLNCapitalSummary(
        entries=entries,
        total_ead=sum(e.ead for e in entries),
        total_rwa=sum(e.rwa for e in entries),
        total_capital=sum(e.capital for e in entries),
        total_simm_im=sum(e.simm_im for e in entries),
        total_notional=sum(e.notional for e in entries),
    )


# ---------------------------------------------------------------------------
# Hedge recommendations + basis monitor
# ---------------------------------------------------------------------------

@dataclass
class CLNHedgeRecommendation:
    """A hedge recommendation for the CLN book."""
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


def cln_hedge_recommendations(
    book: CLNBook,
    curve: DiscountCurve,
    cs01_limit: float = 50_000,
    jtd_limit: float = 5_000_000,
    recovery_limit: float = 100_000,
    dv01_limit: float = 50_000,
) -> list[CLNHedgeRecommendation]:
    """Hedge recommendations with credit-specific actions."""
    risk = book.aggregate_risk(curve)
    recs = []

    checks = [
        ("cs01", abs(risk["total_cs01"]), cs01_limit,
         "Buy CDS protection to reduce credit spread exposure"),
        ("jtd", abs(risk["total_jtd"]), jtd_limit,
         "Sell CLN or buy CDS to reduce jump-to-default risk"),
        ("recovery", abs(risk["total_recovery_sens"]), recovery_limit,
         "Diversify seniority mix or buy recovery swaps"),
        ("dv01", abs(risk["total_dv01"]), dv01_limit,
         "Hedge rate risk via interest rate swaps"),
    ]

    for risk_type, current, limit, action in checks:
        if limit > 0 and current > limit * 0.75:
            recs.append(CLNHedgeRecommendation(
                risk_type=risk_type, current=current, limit=limit,
                breach_pct=current / limit if limit > 0 else 0,
                action=action,
            ))

    return recs


@dataclass
class CLNBasisPoint:
    """CLN vs CDS basis for one issuer."""
    issuer: str
    cln_yield: float      # par coupon from CLN
    cds_spread: float
    basis: float           # cln_yield - cds_spread

    def to_dict(self) -> dict:
        return {
            "issuer": self.issuer, "cln_yield": self.cln_yield,
            "cds_spread": self.cds_spread, "basis": self.basis,
        }


def cln_basis_monitor(
    book: CLNBook,
    curve: DiscountCurve,
    cds_spreads: dict[str, float],
) -> list[CLNBasisPoint]:
    """CLN vs CDS basis tracking.

    Basis = CLN par_coupon - CDS spread. Positive = CLN cheap.
    """
    results = []
    seen = set()
    for e in book.entries:
        if e.issuer in seen or e.issuer not in cds_spreads:
            continue
        seen.add(e.issuer)
        par = e.cln._par_coupon(curve, e.survival_curve)
        cds = cds_spreads[e.issuer]
        results.append(CLNBasisPoint(e.issuer, par, cds, par - cds))
    return results


# ---------------------------------------------------------------------------
# Lifecycle (credit events, margin, early redemption)
# ---------------------------------------------------------------------------

class CLNEventType:
    CREDIT_EVENT = "credit_event"
    MARGIN_CALL = "margin_call"
    EARLY_REDEMPTION = "early_redemption"
    RESTRUCTURING = "restructuring"


@dataclass
class CLNMarginCall:
    """Result of a margin call computation."""
    date: date
    mtm: float
    threshold: float
    required_transfer: float
    direction: str

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "mtm": self.mtm,
            "threshold": self.threshold, "transfer": self.required_transfer,
            "direction": self.direction,
        }


class CLNLifecycle:
    """Lifecycle management for a CLN position.

    Wraps trade_lifecycle.ManagedTrade + CLN-specific credit events.
    """

    def __init__(self, cln: CreditLinkedNote, survival_curve: SurvivalCurve,
                 trade_id: str = "", creation_date: date | None = None):
        from pricebook.trade import Trade
        from pricebook.trade_lifecycle import ManagedTrade

        self._cln = cln
        self._surv = survival_curve
        self._trade_id = trade_id
        trade = Trade(instrument=cln, trade_id=trade_id)
        self._managed = ManagedTrade(trade, creation_date or cln.start)
        self._events: list[dict] = []

    @property
    def cln(self) -> CreditLinkedNote:
        return self._cln

    @property
    def history(self) -> list[dict]:
        base = [
            {"type": e.event_type.value, "date": e.event_date.isoformat(),
             "version": e.version, **e.details}
            for e in self._managed.history
        ]
        return sorted(base + self._events, key=lambda x: x.get("date", ""))

    def credit_event(
        self, event_date: date, curve: DiscountCurve,
        event_type: str = "default",
    ) -> float:
        """Process credit event. Returns recovery payout = R × N."""
        payout = self._cln.recovery * self._cln.notional

        self._events.append({
            "type": CLNEventType.CREDIT_EVENT,
            "date": event_date.isoformat(),
            "event_type": event_type,
            "recovery_payout": payout,
        })

        self._managed.exercise(event_date, underlying_instrument=self._cln)
        return payout

    def restructuring(
        self, event_date: date, curve: DiscountCurve,
        new_coupon: float | None = None,
        new_recovery: float | None = None,
    ) -> float:
        """Process restructuring: adjust terms."""
        if new_coupon is not None:
            self._cln.coupon_rate = new_coupon
        if new_recovery is not None:
            self._cln.recovery = new_recovery

        new_pv = self._cln.dirty_price(curve, self._surv)

        self._events.append({
            "type": CLNEventType.RESTRUCTURING,
            "date": event_date.isoformat(),
            "new_coupon": new_coupon,
            "new_recovery": new_recovery,
            "new_pv": new_pv,
        })

        self._managed.amend(event_date, instrument=self._cln)
        return new_pv

    def margin_call(
        self, as_of: date, curve: DiscountCurve,
        threshold: float = 0.0,
        min_transfer: float = 250_000,
    ) -> CLNMarginCall:
        """Compute margin call based on MTM vs threshold."""
        mtm = self._cln.dirty_price(curve, self._surv)
        net = abs(mtm) - threshold
        required = max(net, 0)
        direction = "receive" if mtm > 0 else "pay"

        if required < min_transfer:
            required = 0.0

        if required > 0:
            self._events.append({
                "type": CLNEventType.MARGIN_CALL,
                "date": as_of.isoformat(),
                "mtm": mtm, "transfer": required,
            })

        return CLNMarginCall(as_of, mtm, threshold, required, direction)

    def early_redeem(self, as_of: date, curve: DiscountCurve) -> float:
        """Early redemption at current MTM."""
        pv = self._cln.dirty_price(curve, self._surv)

        self._events.append({
            "type": CLNEventType.EARLY_REDEMPTION,
            "date": as_of.isoformat(),
            "redemption_pv": pv,
        })

        return pv


# ---------------------------------------------------------------------------
# Collateral evolution
# ---------------------------------------------------------------------------

@dataclass
class CLNCollateralState:
    """Collateral state at a point in time."""
    date: date
    mtm: float
    collateral_posted: float
    net_exposure: float
    margin_call: float
    spread_level: float

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "mtm": self.mtm,
            "collateral": self.collateral_posted,
            "net_exposure": self.net_exposure,
            "margin_call": self.margin_call,
            "spread": self.spread_level,
        }


def cln_collateral_evolution(
    cln: CreditLinkedNote,
    dates: list[date],
    curves: list[DiscountCurve],
    survival_curves: list[SurvivalCurve],
    initial_collateral: float = 0.0,
    threshold: float = 0.0,
    min_transfer: float = 250_000,
) -> list[CLNCollateralState]:
    """Collateral dynamics driven by spread moves."""
    if len(dates) != len(curves) or len(dates) != len(survival_curves):
        raise ValueError("dates, curves, and survival_curves must have same length")

    states = []
    collateral = initial_collateral

    for d, curve, surv in zip(dates, curves, survival_curves):
        mtm = cln.dirty_price(curve, surv)
        T = year_fraction(cln.start, cln.end, DayCountConvention.ACT_365_FIXED)
        spread = -math.log(max(surv.survival(cln.end), 1e-15)) / max(T, 1e-10)

        net_exp = mtm - collateral
        call = 0.0

        if abs(net_exp) > threshold:
            call_amount = abs(net_exp) - threshold
            if call_amount >= min_transfer:
                call = call_amount if net_exp > 0 else -call_amount
                collateral += call

        states.append(CLNCollateralState(d, mtm, collateral, mtm - collateral, call, spread))

    return states
