"""Repo analytics: CTD, term structure, specialness, balance sheet, DV01 ladder.

Analytical functions split from repo_desk.py for maintainability.
All symbols are re-exported from repo_desk.py for backward compatibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.statistics.zscore import zscore as _zscore, ZScoreSignal


def repo_rate_monitor(
    current_rate: float,
    history: list[float],
    threshold: float = 2.0,
) -> ZScoreSignal:
    """Z-score the current repo rate vs history."""
    return _zscore(current_rate, history, threshold)


@dataclass
class CTDRepoCandidate:
    """A bond candidate for repo financing."""
    issuer: str
    bond_price: float
    repo_rate: float
    coupon_rate: float
    term_days: int

    @property
    def financing_cost(self) -> float:
        return self.bond_price * self.repo_rate * self.term_days / 360.0  # ACT/360

    @property
    def carry(self) -> float:
        dt_coupon = self.term_days / 365.0  # ACT/365 for coupon
        return self.coupon_rate * 100.0 * dt_coupon - self.financing_cost


def cheapest_to_deliver_repo(
    candidates: list[CTDRepoCandidate],
) -> CTDRepoCandidate | None:
    """Select the bond that minimises financing cost.

    Among all candidates, picks the one with the lowest financing
    cost per 100 face for the given term.
    """
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.financing_cost)


@dataclass
class TermVsOvernightResult:
    """Comparison of term repo vs rolling overnight."""
    term_rate: float
    overnight_rate: float
    term_days: int
    term_cost: float
    overnight_cost: float
    savings: float
    recommendation: str


def term_vs_overnight(
    face_amount: float,
    bond_price: float,
    term_rate: float,
    overnight_rate: float,
    term_days: int,
) -> TermVsOvernightResult:
    """Compare locking in a term repo vs rolling overnight.

    Assumes the overnight rate is constant over the term for simplicity.

    Args:
        face_amount: face value of collateral.
        bond_price: dirty price per 100 face.
        term_rate: annualised term repo rate.
        overnight_rate: annualised overnight repo rate.
        term_days: number of days for the term repo.

    Returns:
        :class:`TermVsOvernightResult` with costs and recommendation.
    """
    cash = face_amount * bond_price / 100.0
    dt = term_days / 360.0  # ACT/360 repo convention
    term_cost = cash * term_rate * dt
    overnight_cost = cash * overnight_rate * dt

    savings = overnight_cost - term_cost
    if term_cost < overnight_cost:
        recommendation = "term"
    elif term_cost > overnight_cost:
        recommendation = "overnight"
    else:
        recommendation = "indifferent"

    return TermVsOvernightResult(
        term_rate=term_rate,
        overnight_rate=overnight_rate,
        term_days=term_days,
        term_cost=term_cost,
        overnight_cost=overnight_cost,
        savings=savings,
        recommendation=recommendation,
    )


def repo_rate_dv01(
    book: RepoBook,
    shift_bps: float = 1.0,
) -> dict[str, float]:
    """Carry sensitivity to a parallel 1bp repo rate shift.

    Returns:
        total_dv01: change in total carry for +1bp repo shift.
        per_trade: list of per-trade carry changes.
    """
    shift = shift_bps / 10_000.0
    base_carry = book.net_carry()

    # Bump all repo rates and recompute
    bumped_carry = 0.0
    per_trade = []
    for e in book.entries:
        base_c = e.carry
        dt_coupon = e.term_days / 365.0    # ACT/365 for coupon
        dt_fin = e.term_days / 360.0       # ACT/360 for financing
        # Carry = sign × (coupon_income - cash × (repo_rate + shift) × dt_fin)
        sign = 1.0 if e.direction == "repo" else -1.0
        coupon = e.face_amount * e.coupon_rate * dt_coupon
        financing_bumped = e.cash_amount * (e.repo_rate + shift) * dt_fin
        bumped_c = sign * (coupon - financing_bumped)
        bumped_carry += bumped_c
        per_trade.append(bumped_c - base_c)

    return {
        "total_dv01": bumped_carry - base_carry,
        "base_carry": base_carry,
        "bumped_carry": bumped_carry,
        "per_trade_dv01": per_trade,
    }


@dataclass
class RolloverScenario:
    """Cost of rolling O/N repo under a rate spike."""
    scenario_name: str
    on_rate_spike_bps: float
    spike_duration_days: int
    additional_cost: float
    annualised_impact_bps: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "spike_bps": self.on_rate_spike_bps,
            "days": self.spike_duration_days,
            "cost": self.additional_cost,
            "impact_bps": self.annualised_impact_bps,
        }


def rollover_risk(
    book: RepoBook,
    scenarios: list[tuple[str, float, int]] | None = None,
) -> list[RolloverScenario]:
    """Quantify cost of O/N repo rate spikes when rolling forward.

    Computes: for each scenario, how much extra financing cost
    on the O/N portion of the book during the spike.

    Default scenarios: mild (+25bp, 3d), moderate (+100bp, 5d),
    severe (+300bp, 10d), crisis (+500bp, 30d).

    Args:
        scenarios: list of (name, spike_bps, duration_days).
    """
    if scenarios is None:
        scenarios = [
            ("mild", 25, 3),
            ("moderate", 100, 5),
            ("severe", 300, 10),
            ("crisis", 500, 30),
        ]

    # O/N and short-term positions vulnerable to rollover
    on_trades = [e for e in book.entries if e.term_days <= 7]
    on_cash = sum(e.cash_amount for e in on_trades)

    results = []
    for name, spike_bps, days in scenarios:
        spike = spike_bps / 10_000.0
        extra_cost = on_cash * spike * days / 360.0  # ACT/360
        annualised = spike_bps * (days / 365.0) if on_cash > 0 else 0.0
        results.append(RolloverScenario(
            scenario_name=name,
            on_rate_spike_bps=spike_bps,
            spike_duration_days=days,
            additional_cost=extra_cost,
            annualised_impact_bps=annualised,
        ))

    return results


@dataclass
class CounterpartyLimit:
    """Counterparty limit and utilisation."""
    counterparty: str
    limit: float
    current_exposure: float
    utilisation_pct: float
    breached: bool
    headroom: float

    def to_dict(self) -> dict:
        return {
            "counterparty": self.counterparty,
            "limit": self.limit,
            "exposure": self.current_exposure,
            "utilisation_pct": self.utilisation_pct,
            "breached": self.breached,
            "headroom": self.headroom,
        }


def counterparty_exposure_monitor(
    book: RepoBook,
    limits: dict[str, float] | None = None,
    default_limit: float = 500_000_000.0,
) -> list[CounterpartyLimit]:
    """Monitor counterparty exposure against limits.

    Args:
        limits: {counterparty: max_exposure}. Missing CPs get default_limit.
        default_limit: default exposure limit per CP.

    Returns:
        List of CounterpartyLimit, sorted by utilisation (highest first).
    """
    exposures = book.by_counterparty()
    if limits is None:
        limits = {}

    results = []
    for cp_exp in exposures:
        limit = limits.get(cp_exp.counterparty, default_limit)
        exposure = abs(cp_exp.total_cash)
        util = (exposure / limit * 100.0) if limit > 0 else 0.0
        results.append(CounterpartyLimit(
            counterparty=cp_exp.counterparty,
            limit=limit,
            current_exposure=exposure,
            utilisation_pct=util,
            breached=exposure > limit,
            headroom=max(0, limit - exposure),
        ))

    return sorted(results, key=lambda r: -r.utilisation_pct)


@dataclass
class SpecialnessForecast:
    """Forecast of specialness for a bond."""
    bond_id: str
    current_specialness_bps: float
    days_to_auction: int | None
    forecast_specialness_bps: float
    trend: str  # "widening", "stable", "collapsing"
    confidence: str  # "high", "medium", "low"

    def to_dict(self) -> dict:
        return {
            "bond": self.bond_id,
            "current_bps": self.current_specialness_bps,
            "forecast_bps": self.forecast_specialness_bps,
            "days_to_auction": self.days_to_auction,
            "trend": self.trend, "confidence": self.confidence,
        }


def forecast_specialness(
    bond_id: str,
    current_specialness_bps: float,
    days_to_auction: int | None = None,
    borrowing_demand_pct: float = 0.5,
    supply_pct: float = 0.5,
) -> SpecialnessForecast:
    """Forecast specialness using supply-demand rules.

    Rules:
    - Close to auction (< 14 days): specialness widens (supply about to increase).
    - High borrowing demand (> 70%): specialness widens.
    - Post-auction (just happened): specialness collapses.
    - Low demand (< 30%): specialness narrows.

    Args:
        borrowing_demand_pct: fraction of outstanding on loan (0-1).
        supply_pct: available supply relative to demand (0-1).
    """
    forecast = current_specialness_bps
    confidence = "medium"

    # Auction proximity effect
    if days_to_auction is not None:
        if days_to_auction <= 3:
            # Just before auction — specialness at peak, about to collapse
            forecast *= 1.2
            trend = "collapsing"
            confidence = "high"
        elif days_to_auction <= 14:
            # Approaching auction — widening
            forecast *= 1.1
            trend = "widening"
        elif days_to_auction <= 30:
            trend = "stable"
        else:
            # Far from auction — demand builds slowly
            forecast *= 0.9
            trend = "stable"
    else:
        trend = "stable"

    # Demand/supply
    if borrowing_demand_pct > 0.7:
        forecast *= 1.3
        if trend == "stable":
            trend = "widening"
        confidence = "high"
    elif borrowing_demand_pct < 0.3:
        forecast *= 0.7
        if trend == "stable":
            trend = "collapsing"

    if supply_pct < 0.3:
        forecast *= 1.2  # scarce supply widens
    elif supply_pct > 0.8:
        forecast *= 0.8  # ample supply narrows

    return SpecialnessForecast(
        bond_id=bond_id,
        current_specialness_bps=current_specialness_bps,
        days_to_auction=days_to_auction,
        forecast_specialness_bps=max(0, forecast),
        trend=trend,
        confidence=confidence,
    )


@dataclass
class RepoCurveStress:
    """Repo book P&L under a curve stress scenario."""
    scenario_name: str
    on_shift_bps: float
    term_shift_bps: float
    carry_impact: float
    financing_impact: float
    total_impact: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "on_shift": self.on_shift_bps, "term_shift": self.term_shift_bps,
            "carry_impact": self.carry_impact,
            "financing_impact": self.financing_impact,
            "total": self.total_impact,
        }


def repo_curve_stress(
    book: RepoBook,
    scenarios: list[tuple[str, float, float]] | None = None,
) -> list[RepoCurveStress]:
    """Stress the repo book under curve scenarios.

    Each scenario specifies O/N and term rate shifts independently,
    capturing curve flattening/steepening as well as parallel moves.

    Default scenarios:
    - parallel_up: +50bp across all tenors
    - parallel_down: -50bp
    - steepener: O/N -25bp, term +50bp
    - flattener: O/N +50bp, term -25bp
    - inversion: O/N +100bp, term -50bp
    """
    if scenarios is None:
        scenarios = [
            ("parallel_up", 50, 50),
            ("parallel_down", -50, -50),
            ("steepener", -25, 50),
            ("flattener", 50, -25),
            ("inversion", 100, -50),
        ]

    base_carry = book.net_carry()

    results = []
    for name, on_shift, term_shift in scenarios:
        stressed_carry = 0.0
        for e in book.entries:
            # O/N positions get on_shift, longer-term get term_shift
            if e.term_days <= 7:
                shift = on_shift / 10_000.0
            else:
                shift = term_shift / 10_000.0

            dt_coupon = e.term_days / 365.0   # ACT/365 for coupon
            dt_fin = e.term_days / 360.0      # ACT/360 for financing
            sign = 1.0 if e.direction == "repo" else -1.0
            coupon = e.face_amount * e.coupon_rate * dt_coupon
            financing = e.cash_amount * (e.repo_rate + shift) * dt_fin
            stressed_carry += sign * (coupon - financing)

        carry_impact = stressed_carry - base_carry
        # Financing impact: extra cost from the shift (ACT/360)
        financing_impact = sum(
            e.cash_amount * (on_shift if e.term_days <= 7 else term_shift) / 10_000.0
            * e.term_days / 360.0
            for e in book.entries
        )

        results.append(RepoCurveStress(
            scenario_name=name,
            on_shift_bps=on_shift,
            term_shift_bps=term_shift,
            carry_impact=carry_impact,
            financing_impact=financing_impact,
            total_impact=carry_impact,
        ))

    return results


@dataclass
class BalanceSheetMetrics:
    """Balance sheet efficiency for the repo desk."""
    total_assets: float
    total_capital_used: float
    annual_carry: float
    return_on_capital_pct: float
    leverage_ratio: float

    def to_dict(self) -> dict:
        return {
            "total_assets": self.total_assets, "capital_used": self.total_capital_used,
            "annual_carry": self.annual_carry, "roc_pct": self.return_on_capital_pct,
            "leverage": self.leverage_ratio,
        }


def balance_sheet_efficiency(
    book: RepoBook,
    haircut_pct: float = 2.0,
) -> BalanceSheetMetrics:
    """ROC = annualised_carry / capital. Leverage = assets / capital."""
    total_assets = sum(e.cash_amount for e in book.entries)
    capital = total_assets * haircut_pct / 100.0

    annual_carry = sum(
        e.carry * (365.0 / max(e.term_days, 1)) for e in book.entries
    )

    roc = (annual_carry / capital * 100.0) if capital > 0 else 0.0
    leverage = total_assets / capital if capital > 0 else 0.0

    return BalanceSheetMetrics(total_assets, capital, annual_carry, roc, leverage)


@dataclass
class MatchedBookEntry:
    """A matched pair: repo + reverse on same collateral."""
    issuer: str
    repo_cash: float
    reverse_cash: float
    repo_rate: float
    reverse_rate: float
    spread_earned_bps: float
    net_carry: float

    def to_dict(self) -> dict:
        return {
            "issuer": self.issuer,
            "repo_cash": self.repo_cash, "reverse_cash": self.reverse_cash,
            "repo_rate": self.repo_rate, "reverse_rate": self.reverse_rate,
            "spread_bps": self.spread_earned_bps, "net_carry": self.net_carry,
        }


def matched_book_analysis(book: RepoBook) -> list[MatchedBookEntry]:
    """Find matched repo/reverse pairs on same collateral and compute spread.

    The desk earns the spread between the rate it borrows at (repo)
    and the rate it lends at (reverse).
    """
    # Group by issuer
    by_issuer: dict[str, dict] = {}
    for e in book.entries:
        iss = e.collateral_issuer
        if iss not in by_issuer:
            by_issuer[iss] = {"repo": [], "reverse": []}
        by_issuer[iss][e.direction].append(e)

    matches = []
    for iss, sides in by_issuer.items():
        if not sides["repo"] or not sides["reverse"]:
            continue

        repo_cash = sum(e.cash_amount for e in sides["repo"])
        repo_rate = (sum(e.cash_amount * e.repo_rate for e in sides["repo"])
                     / repo_cash if repo_cash > 0 else 0.0)
        rev_cash = sum(e.cash_amount for e in sides["reverse"])
        rev_rate = (sum(e.cash_amount * e.repo_rate for e in sides["reverse"])
                    / rev_cash if rev_cash > 0 else 0.0)

        spread = (rev_rate - repo_rate) * 10_000  # bps earned
        matched_amt = min(repo_cash, rev_cash)
        avg_term = sum(e.term_days for e in sides["repo"] + sides["reverse"]) / \
                   len(sides["repo"] + sides["reverse"])
        net_carry = matched_amt * (rev_rate - repo_rate) * avg_term / 365.0

        matches.append(MatchedBookEntry(
            issuer=iss, repo_cash=repo_cash, reverse_cash=rev_cash,
            repo_rate=repo_rate, reverse_rate=rev_rate,
            spread_earned_bps=spread, net_carry=net_carry,
        ))

    return sorted(matches, key=lambda m: -abs(m.net_carry))


@dataclass
class FundingAttribution:
    """P&L attribution by strategy."""
    strategy: str
    total_cash: float
    total_carry: float
    pct_of_book: float

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "cash": self.total_cash,
            "carry": self.total_carry, "pct": self.pct_of_book,
        }


def funding_attribution(book: RepoBook) -> list[FundingAttribution]:
    """Attribute carry by strategy axis: GC vs special, ON vs term, repo vs reverse."""
    total_carry = book.net_carry()
    total_cash = sum(e.cash_amount for e in book.entries) or 1.0

    axes = {
        "GC_ON": lambda e: e.collateral_type == "GC" and e.term_days <= 1,
        "GC_term": lambda e: e.collateral_type == "GC" and e.term_days > 1,
        "special_ON": lambda e: e.collateral_type == "special" and e.term_days <= 1,
        "special_term": lambda e: e.collateral_type == "special" and e.term_days > 1,
        "reverse": lambda e: e.direction == "reverse",
    }

    result = []
    for strat, predicate in axes.items():
        entries = [e for e in book.entries if predicate(e)]
        cash = sum(e.cash_amount for e in entries)
        carry = sum(e.carry for e in entries)
        pct = cash / total_cash * 100.0
        result.append(FundingAttribution(strat, cash, carry, pct))

    return sorted(result, key=lambda f: -abs(f.total_carry))


def repo_book_pv(
    book: RepoBook,
    discount_curve,
    reference_date: date,
) -> dict[str, float]:
    """Total PV of all repo positions against a discount curve.

    Each position: PV = df(maturity) × repurchase_amount − cash_lent.

    Returns total PV, per-direction breakdown.
    """
    from pricebook.day_count import year_fraction, DayCountConvention
    from datetime import timedelta

    total_pv = 0.0
    repo_pv = 0.0
    reverse_pv = 0.0

    for e in book.entries:
        start = e.start_date or reference_date
        mat = start + timedelta(days=e.term_days)
        df = discount_curve.df(mat)
        dt = e.term_days / 360.0  # ACT/360 repo convention

        repurchase = e.cash_amount * (1 + e.repo_rate * dt)

        if e.direction == "repo":
            # Lent bond, borrowed cash. PV = df × repurchase − cash_lent
            pv = df * repurchase - e.cash_amount
            repo_pv += pv
        else:
            # Reverse: lent cash, borrowed bond. PV = cash_lent − df × repurchase
            pv = e.cash_amount - df * repurchase
            reverse_pv += pv

        total_pv += pv

    return {
        "total_pv": total_pv,
        "repo_pv": repo_pv,
        "reverse_pv": reverse_pv,
        "n_positions": len(book),
    }


def sofr_compounded_with_lookback(
    daily_rates: list[float],
    lookback_days: int = 2,
    lockout_days: int = 0,
) -> float:
    """Compounded SOFR with lookback and lockout conventions.

    Lookback: use rate from N days ago (SOFR published with lag).
    Lockout: last N days use the rate from the lockout start.

    Returns annualised compounded rate.
    """
    n = len(daily_rates)
    if n == 0:
        return 0.0

    # Apply lookback: shift rates back by lookback_days
    shifted = [0.0] * n
    for i in range(n):
        src = max(0, i - lookback_days)
        shifted[i] = daily_rates[src]

    # Apply lockout: freeze last N days
    if lockout_days > 0 and n > lockout_days:
        lock_rate = shifted[n - lockout_days - 1]
        for i in range(n - lockout_days, n):
            shifted[i] = lock_rate

    # Compound
    compound = 1.0
    for r in shifted:
        compound *= (1 + r / 360.0)

    total_yf = n / 360.0
    if total_yf <= 0:
        return 0.0
    return (compound - 1.0) / total_yf


def repo_key_rate_dv01(
    book: RepoBook,
    repo_curve,
    shift_bps: float = 1.0,
) -> dict[int, float]:
    """Key-rate DV01 on the repo curve — carry sensitivity per tenor bucket.

    Bumps each tenor on the repo curve independently and measures
    the carry change.

    Returns: {tenor_days: carry_change_per_bp}.
    """
    from pricebook.fixed_income.repo_term import RepoCurve, RepoRate

    base_carry = book.net_carry()
    shift = shift_bps / 10_000.0
    result = {}

    for i, tenor_days in enumerate(repo_curve._days):
        # Bump this tenor only
        new_rates = list(repo_curve._rates)
        new_rates[i] += shift
        bumped = RepoCurve(
            repo_curve.reference_date,
            [RepoRate(d, r) for d, r in zip(repo_curve._days, new_rates)],
        )

        # Reprice all trades at bumped repo rates
        bumped_carry = 0.0
        for e in book.entries:
            bumped_rate = bumped.rate(e.term_days)
            # Carry formula: coupon (ACT/365) - financing (ACT/360)
            dt_coupon = e.term_days / 365.0
            dt_fin = e.term_days / 360.0  # ACT/360 for financing
            sign = 1.0 if e.direction == "repo" else -1.0
            coupon = e.face_amount * e.coupon_rate * dt_coupon
            financing = e.cash_amount * bumped_rate * dt_fin
            bumped_carry += sign * (coupon - financing)

        result[tenor_days] = (bumped_carry - base_carry) / shift_bps

    return result


BASEL_HAIRCUT_FLOORS = {
    # Asset class → minimum haircut % (Basel III Table 1)
    "sovereign_0_1Y": 0.5,
    "sovereign_1_5Y": 2.0,
    "sovereign_5Y+": 4.0,
    "agency_0_1Y": 1.0,
    "agency_1_5Y": 3.0,
    "agency_5Y+": 6.0,
    "ig_corp_0_1Y": 2.0,
    "ig_corp_1_5Y": 4.0,
    "ig_corp_5Y+": 8.0,
    "hy_corp": 15.0,
    "equity_main_index": 15.0,
    "equity_other": 25.0,
    "cash_same_ccy": 0.0,
    "fx_mismatch_add_on": 8.0,  # additional for xccy
}


def regulatory_haircut(
    asset_class: str,
    maturity_years: float,
    is_cross_currency: bool = False,
) -> float:
    """Minimum regulatory haircut (Basel III).

    Args:
        asset_class: "sovereign", "agency", "ig_corp", "hy_corp", "equity"
        maturity_years: remaining maturity of the collateral.
        is_cross_currency: adds 8% FX add-on.
    """
    if asset_class in ("hy_corp", "equity_main_index", "equity_other"):
        key = asset_class
    else:
        if maturity_years <= 1:
            bucket = "0_1Y"
        elif maturity_years <= 5:
            bucket = "1_5Y"
        else:
            bucket = "5Y+"
        key = f"{asset_class}_{bucket}"

    haircut = BASEL_HAIRCUT_FLOORS.get(key, 10.0)

    if is_cross_currency:
        haircut += BASEL_HAIRCUT_FLOORS.get("fx_mismatch_add_on", 8.0)

    return haircut
