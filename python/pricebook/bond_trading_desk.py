"""Bond trading desk: unified risk metrics, carry, dashboard, stress, lifecycle.

Consolidates existing bond infrastructure (bond_book, bond_daily_pnl,
bond_capital, bond_rv, bond_desk) into a single desk layer matching
the trs_desk.py / cln_desk.py pattern.

    from pricebook.bond_trading_desk import (
        bond_risk_metrics, BondRiskMetrics,
        bond_carry_roll, BondCarryRollDecomposition,
        bond_dashboard, BondDashboard,
        bond_stress_suite, bond_scenario_stress,
        bond_hedge_recommendations, BondHedgeRecommendation,
        bond_funding_cost, BondFundingCost,
        BondLifecycle,
    )

Known out-of-scope:
- MBS pool factors / prepayment tracking: use amortising_bond.py directly.
- Real-time market data feed: pricing library, not trading system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.bond import FixedRateBond
from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class BondRiskMetrics:
    """Unified risk metrics for a bond position."""
    pv: float                           # dirty price (per 100 face)
    clean_price: float
    ytm: float
    macaulay_duration: float
    modified_duration: float
    effective_duration: float           # bump-and-reprice on curve
    convexity: float
    dv01: float                         # parallel DV01 (yield-based)
    dv01_curve: float                   # parallel DV01 (curve-based, centred)
    key_rate_dv01: dict[str, float]     # per-pillar sensitivities
    face_value: float
    accrued_interest: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "clean": self.clean_price, "ytm": self.ytm,
            "mac_dur": self.macaulay_duration, "mod_dur": self.modified_duration,
            "eff_dur": self.effective_duration, "convexity": self.convexity,
            "dv01": self.dv01, "dv01_curve": self.dv01_curve,
            "key_rate_dv01": self.key_rate_dv01,
            "face": self.face_value, "accrued": self.accrued_interest,
        }


def bond_risk_metrics(
    bond: FixedRateBond,
    curve: DiscountCurve,
    settlement: date | None = None,
) -> BondRiskMetrics:
    """Compute unified risk metrics for a bond.

    Combines existing bond methods with bump-and-reprice for
    effective duration and per-pillar key-rate DV01.
    """
    settle = settlement or curve.reference_date
    dp = bond.dirty_price(curve)
    cp = bond.clean_price(curve, settle)
    ai = bond.accrued_interest(settle)

    ytm = bond.yield_to_maturity(dp, settle)
    mac_dur = bond.macaulay_duration(ytm, settle)
    mod_dur = bond.modified_duration(ytm, settle)
    conv = bond.convexity(ytm, settle)
    dv01 = bond.dv01_yield(ytm, settle)

    # Effective duration: centred bump-and-reprice on curve (O(h²))
    h = 0.0001
    pv_up = bond.dirty_price(curve.bumped(h))
    pv_dn = bond.dirty_price(curve.bumped(-h))
    eff_dur = -(pv_up - pv_dn) / (2 * h * dp) if dp > 0 else 0.0

    # Curve-based DV01 (centred, in price units per 100 face)
    dv01_curve = (pv_up - pv_dn) / 2

    # Key-rate DV01: per-pillar sensitivity
    pillar_times = [t for t in curve.pillar_times if t > 0]
    key_rate = {}
    for i, t in enumerate(pillar_times):
        bumped_i = curve.bumped_at(i, h)
        kr_dv01 = bond.dirty_price(bumped_i) - dp
        label = _time_to_tenor_label(t)
        key_rate[label] = kr_dv01

    return BondRiskMetrics(
        pv=dp, clean_price=cp, ytm=ytm,
        macaulay_duration=mac_dur, modified_duration=mod_dur,
        effective_duration=eff_dur, convexity=conv,
        dv01=dv01, dv01_curve=dv01_curve,
        key_rate_dv01=key_rate,
        face_value=bond.face_value, accrued_interest=ai,
    )


def _time_to_tenor_label(t: float) -> str:
    """Convert year fraction to human-readable tenor label."""
    if t < 1/12:
        return f"{int(t*52)}W"
    if t < 1.0:
        return f"{int(t*12)}M"
    return f"{int(round(t))}Y"


# ---------------------------------------------------------------------------
# Carry-and-roll decomposition
# ---------------------------------------------------------------------------

@dataclass
class BondCarryRollDecomposition:
    """Prospective carry-and-roll forecast.

    What a bond earns if markets don't move over the horizon.
    """
    coupon_carry: float          # coupon income over horizon
    funding_cost: float          # repo financing cost
    net_carry: float             # coupon - funding
    roll_down_return: float      # price gain from aging on unchanged curve
    pull_to_par: float           # convergence toward par
    total_carry_and_roll: float  # net_carry + roll_down
    horizon_days: int

    def to_dict(self) -> dict:
        return {
            "coupon": self.coupon_carry, "funding": self.funding_cost,
            "net_carry": self.net_carry, "roll_down": self.roll_down_return,
            "pull_to_par": self.pull_to_par, "total": self.total_carry_and_roll,
            "horizon": self.horizon_days,
        }


def bond_carry_roll(
    bond: FixedRateBond,
    curve: DiscountCurve,
    repo_rate: float = 0.04,
    horizon_days: int = 30,
    settlement: date | None = None,
) -> BondCarryRollDecomposition:
    """Prospective carry-and-roll forecast.

    What the bond earns over `horizon_days` if the curve doesn't move.
    """
    from pricebook.pnl_explain import compute_rolldown

    settle = settlement or curve.reference_date
    dp = bond.dirty_price(curve)
    cp = bond.clean_price(curve, settle)

    # Coupon carry: annualised coupon × horizon fraction
    coupon = bond.coupon_rate * bond.face_value * horizon_days / 365.0

    # Funding cost: repo rate × dirty price × horizon fraction
    funding = repo_rate * dp / 100.0 * bond.face_value * horizon_days / 365.0

    net_carry = coupon - funding

    # Roll-down: price change from aging on unchanged curve
    roll_down = compute_rolldown(
        lambda c: bond.dirty_price(c), curve, days=horizon_days,
    ) / 100.0 * bond.face_value

    # Pull-to-par: convergence toward 100
    remaining_days = max((bond.maturity - settle).days, 1)
    pull = (100.0 - cp) / 100.0 * bond.face_value * horizon_days / remaining_days

    total = net_carry + roll_down

    return BondCarryRollDecomposition(
        coupon_carry=coupon, funding_cost=funding, net_carry=net_carry,
        roll_down_return=roll_down, pull_to_par=pull,
        total_carry_and_roll=total, horizon_days=horizon_days,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class BondDashboardEntry:
    """Risk summary for one bond position."""
    trade_id: str
    issuer: str
    face: float
    pv: float
    dv01: float
    duration: float


@dataclass
class BondDashboard:
    """Morning-meeting summary for the bond desk."""
    date: date
    n_positions: int
    total_face: float
    total_pv: float
    total_dv01: float
    weighted_duration: float
    weighted_convexity: float
    by_tenor: dict[str, int]
    top_positions: list[BondDashboardEntry]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "face": self.total_face, "pv": self.total_pv,
            "dv01": self.total_dv01, "duration": self.weighted_duration,
            "convexity": self.weighted_convexity,
            "by_tenor": self.by_tenor,
            "top_positions": [
                {"id": e.trade_id, "issuer": e.issuer, "dv01": e.dv01}
                for e in self.top_positions
            ],
        }


def bond_dashboard(
    positions: list[tuple[str, str, FixedRateBond, float]],
    reference_date: date,
    curve: DiscountCurve,
) -> BondDashboard:
    """Build bond desk morning dashboard.

    Args:
        positions: list of (trade_id, issuer, bond, face_amount).
    """
    entries = []
    total_face = 0.0
    total_pv = 0.0
    total_dv01_weight = 0.0
    total_dur_weight = 0.0
    total_conv_weight = 0.0

    for trade_id, issuer, bond, face in positions:
        rm = bond_risk_metrics(bond, curve, reference_date)
        pv_scaled = rm.pv / 100.0 * face
        dv01_scaled = rm.dv01_curve * face / rm.face_value

        entries.append(BondDashboardEntry(
            trade_id, issuer, face, pv_scaled, dv01_scaled, rm.modified_duration,
        ))
        total_face += face
        total_pv += pv_scaled
        total_dv01_weight += dv01_scaled
        total_dur_weight += rm.modified_duration * pv_scaled
        total_conv_weight += rm.convexity * pv_scaled

    w_dur = total_dur_weight / total_pv if total_pv > 0 else 0.0
    w_conv = total_conv_weight / total_pv if total_pv > 0 else 0.0

    # Tenor bucketing
    tenors: dict[str, int] = {}
    for _, _, bond, _ in positions:
        T = year_fraction(reference_date, bond.maturity, DayCountConvention.ACT_365_FIXED)
        bucket = _time_to_tenor_label(T)
        tenors[bucket] = tenors.get(bucket, 0) + 1

    # Top positions by abs DV01
    top = sorted(entries, key=lambda e: abs(e.dv01), reverse=True)[:5]

    return BondDashboard(
        date=reference_date, n_positions=len(positions),
        total_face=total_face, total_pv=total_pv,
        total_dv01=total_dv01_weight,
        weighted_duration=w_dur, weighted_convexity=w_conv,
        by_tenor=tenors, top_positions=top,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class BondStressResult:
    """One stress scenario result."""
    scenario: str
    description: str
    rate_pnl: float
    total_pnl: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario, "description": self.description,
            "rate": self.rate_pnl, "total": self.total_pnl,
        }


def bond_stress_suite(
    positions: list[tuple[str, FixedRateBond, float]],
    curve: DiscountCurve,
) -> list[BondStressResult]:
    """Five parametric stress scenarios using DV01 + convexity."""
    # Aggregate DV01 and convexity
    total_dv01 = 0.0
    total_conv = 0.0
    for _, bond, face in positions:
        rm = bond_risk_metrics(bond, curve)
        total_dv01 += rm.dv01_curve * face / rm.face_value
        total_conv += rm.convexity * rm.pv / 100.0 * face / rm.face_value

    scenarios = [
        ("rates_up_100", "Rates +100bp", 100.0),
        ("rates_dn_100", "Rates -100bp", -100.0),
        ("rates_up_200", "Rates +200bp", 200.0),
        ("rates_dn_200", "Rates -200bp", -200.0),
        ("rates_up_50", "Rates +50bp", 50.0),
    ]

    results = []
    for name, desc, bp in scenarios:
        # Taylor: PnL ≈ DV01 × Δbp + 0.5 × Convexity × (Δy)²
        rate_pnl = total_dv01 * bp
        conv_adj = 0.5 * total_conv * (bp / 10_000) ** 2
        total = rate_pnl + conv_adj
        results.append(BondStressResult(name, desc, rate_pnl, total))

    return results


def bond_scenario_stress(
    positions: list[tuple[str, FixedRateBond, float]],
    ctx,
    scenarios: list | None = None,
) -> list:
    """Full-reprice stress via scenario.py run_scenarios."""
    from pricebook.scenario import parallel_shift, run_scenarios
    from pricebook.trade import Trade, Portfolio

    portfolio = Portfolio(name="bond_stress")
    for trade_id, bond, face in positions:
        t = Trade(instrument=bond, trade_id=trade_id, notional_scale=face / bond.face_value)
        portfolio.add(t)

    if scenarios is None:
        scenarios = [
            parallel_shift(0.01, "rates_+100bp"),
            parallel_shift(-0.01, "rates_-100bp"),
            parallel_shift(0.02, "rates_+200bp"),
        ]

    return run_scenarios(portfolio, ctx, scenarios)


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class BondHedgeRecommendation:
    """A hedge recommendation for the bond book."""
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


def bond_hedge_recommendations(
    positions: list[tuple[str, FixedRateBond, float]],
    curve: DiscountCurve,
    dv01_limit: float = 50_000,
    duration_limit: float = 10.0,
    concentration_limit_pct: float = 0.25,
) -> list[BondHedgeRecommendation]:
    """Generate hedge recommendations when risk exceeds limits."""
    total_dv01 = 0.0
    total_pv = 0.0
    total_dur_weight = 0.0

    for _, bond, face in positions:
        rm = bond_risk_metrics(bond, curve)
        pv_scaled = rm.pv / 100.0 * face
        total_dv01 += abs(rm.dv01_curve * face / rm.face_value)
        total_pv += pv_scaled
        total_dur_weight += rm.modified_duration * pv_scaled

    w_dur = total_dur_weight / total_pv if total_pv > 0 else 0.0

    recs = []
    checks = [
        ("dv01", total_dv01, dv01_limit,
         "Hedge rate risk via Treasury futures or IRS"),
        ("duration", w_dur, duration_limit,
         "Reduce duration by selling long-end or buying short-end"),
    ]

    for risk_type, current, limit, action in checks:
        if limit > 0 and current > limit * 0.75:
            recs.append(BondHedgeRecommendation(
                risk_type=risk_type, current=current, limit=limit,
                breach_pct=current / limit if limit > 0 else 0,
                action=action,
            ))

    return recs


# ---------------------------------------------------------------------------
# Funding cost
# ---------------------------------------------------------------------------

@dataclass
class BondFundingCost:
    """Funding cost analysis for a bond position."""
    coupon_income: float         # annualised coupon
    repo_cost: float             # annualised repo financing cost
    balance_sheet_cost: float    # capital charge × hurdle
    net_income: float            # coupon - repo - balance_sheet
    all_in_yield: float          # YTM adjusted for costs
    breakeven_repo: float        # repo rate where carry = 0

    def to_dict(self) -> dict:
        return {
            "coupon": self.coupon_income, "repo": self.repo_cost,
            "bs_cost": self.balance_sheet_cost, "net": self.net_income,
            "all_in_yield": self.all_in_yield, "breakeven_repo": self.breakeven_repo,
        }


def bond_funding_cost(
    bond: FixedRateBond,
    market_price: float,
    repo_rate: float,
    capital_charge: float = 0.0,
    hurdle_rate: float = 0.10,
    settlement: date | None = None,
) -> BondFundingCost:
    """Compute funding cost breakdown for a bond position.

    breakeven_repo: the repo rate at which carry = 0.
    all_in_yield: YTM minus annualised funding cost per notional.
    """
    settle = settlement or bond.issue_date

    coupon = bond.coupon_rate * bond.face_value
    repo_cost = repo_rate * market_price / 100.0 * bond.face_value
    bs_cost = capital_charge * hurdle_rate

    net = coupon - repo_cost - bs_cost

    # All-in yield: approximate
    T = year_fraction(settle, bond.maturity, DayCountConvention.ACT_365_FIXED)
    annualised_cost = (repo_cost + bs_cost) / bond.face_value
    ytm_approx = bond.coupon_rate  # approximate
    all_in = ytm_approx - annualised_cost

    # Breakeven repo: coupon / (price/100) = rate where carry = 0
    breakeven = coupon / (market_price / 100.0 * bond.face_value) if market_price > 0 else 0.0

    return BondFundingCost(
        coupon_income=coupon, repo_cost=repo_cost,
        balance_sheet_cost=bs_cost, net_income=net,
        all_in_yield=all_in, breakeven_repo=breakeven,
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class BondEventType:
    COUPON = "coupon"
    MATURITY = "maturity"
    CALL_EXERCISE = "call_exercise"


class BondLifecycle:
    """Lifecycle management for a bond position."""

    def __init__(self, bond: FixedRateBond, trade_id: str = "",
                 creation_date: date | None = None):
        from pricebook.trade import Trade
        from pricebook.trade_lifecycle import ManagedTrade

        self._bond = bond
        self._trade_id = trade_id
        trade = Trade(instrument=bond, trade_id=trade_id)
        self._managed = ManagedTrade(trade, creation_date or bond.issue_date)
        self._events: list[dict] = []

    @property
    def bond(self) -> FixedRateBond:
        return self._bond

    @property
    def history(self) -> list[dict]:
        base = [
            {"type": e.event_type.value, "date": e.event_date.isoformat(),
             "version": e.version, **e.details}
            for e in self._managed.history
        ]
        return sorted(base + self._events, key=lambda x: x.get("date", ""))

    def upcoming_events(self, as_of: date, horizon_days: int = 90) -> list[dict]:
        """List upcoming coupon payments and maturity within horizon."""
        horizon = as_of + timedelta(days=horizon_days)
        events = []

        for cf in self._bond.coupon_leg.cashflows:
            if as_of < cf.payment_date <= horizon:
                events.append({
                    "type": BondEventType.COUPON,
                    "date": cf.payment_date.isoformat(),
                    "amount": cf.amount,
                })

        if as_of < self._bond.maturity <= horizon:
            events.append({
                "type": BondEventType.MATURITY,
                "date": self._bond.maturity.isoformat(),
                "amount": self._bond.face_value,
            })

        return events

    def maturity_alert(self, as_of: date, alert_days: int = 30) -> dict | None:
        """Return alert if maturity within alert_days."""
        days_to_mat = (self._bond.maturity - as_of).days
        if 0 < days_to_mat <= alert_days:
            return {
                "type": "maturity_alert",
                "date": self._bond.maturity.isoformat(),
                "days_remaining": days_to_mat,
            }
        return None

    def process_coupon(self, coupon_date: date, amount: float) -> dict:
        """Record coupon receipt."""
        event = {
            "type": BondEventType.COUPON,
            "date": coupon_date.isoformat(),
            "amount": amount,
        }
        self._events.append(event)
        return event
