"""Repo desk: position management, GC/special tracking, financing optimisation.

Builds on :mod:`pricebook.bond_desk` (``RepoPosition``, ``repo_carry``,
``securities_lending_fee``) and :mod:`pricebook.funded` (``Repo``) with
desk-level tooling for repo operations.

* :class:`RepoBook` — positions by counterparty, collateral type, term.
* :func:`repo_rate_monitor` — z-score the current repo rate vs history.
* :func:`cheapest_to_deliver_repo` — select the bond that minimises
  financing cost.
* :func:`term_vs_overnight` — compare locking in term repo vs rolling
  overnight.
* :class:`FailsTracker` — track and cost settlement fails.

    book = RepoBook("GovtRepo")
    book.add(entry)
    pnl = book.net_carry()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.zscore import zscore as _zscore, ZScoreSignal


# ---- Repo trade entry ----

@dataclass
class RepoTradeEntry:
    """A single repo position with desk metadata.

    Attributes:
        counterparty: repo counterparty.
        collateral_issuer: issuer of the collateral bond.
        collateral_type: ``"GC"`` (general collateral) or ``"special"``.
        face_amount: face value of the collateral bond.
        bond_price: dirty price of the collateral (per 100 face).
        repo_rate: annualised repo rate.
        term_days: repo term in calendar days (0 = overnight).
        coupon_rate: annual coupon of the collateral bond.
        direction: ``"repo"`` (lend bond / borrow cash) or
            ``"reverse"`` (borrow bond / lend cash).
        start_date: repo start date.
    """
    counterparty: str
    collateral_issuer: str
    collateral_type: str = "GC"
    face_amount: float = 0.0
    bond_price: float = 100.0
    repo_rate: float = 0.0
    term_days: int = 1
    coupon_rate: float = 0.0
    direction: str = "repo"
    start_date: date | None = None

    @property
    def cash_amount(self) -> float:
        """Cash lent / borrowed = face × dirty_price / 100."""
        return self.face_amount * self.bond_price / 100.0

    @property
    def carry(self) -> float:
        """Net carry = coupon income − financing cost over the term."""
        dt = self.term_days / 365.0
        coupon = self.face_amount * self.coupon_rate * dt
        financing = self.cash_amount * self.repo_rate * dt
        sign = 1.0 if self.direction == "repo" else -1.0
        return sign * (coupon - financing)

    @property
    def financing_cost(self) -> float:
        dt = self.term_days / 365.0
        return self.cash_amount * self.repo_rate * dt


# ---- Repo book ----

@dataclass
class RepoCounterpartyExposure:
    """Aggregate repo exposure per counterparty."""
    counterparty: str
    total_cash: float
    n_trades: int
    avg_rate: float


@dataclass
class RepoCollateralSummary:
    """Aggregate by collateral type (GC vs special)."""
    collateral_type: str
    total_cash: float
    avg_rate: float
    n_trades: int


class RepoBook:
    """A collection of repo positions with aggregation.

    Args:
        name: book name (e.g. "GovtRepo", "IG_Repo").
    """

    def __init__(self, name: str):
        self.name = name
        self._entries: list[RepoTradeEntry] = []

    def add(self, entry: RepoTradeEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[RepoTradeEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def net_carry(self) -> float:
        """Total net carry across all repo positions."""
        return sum(e.carry for e in self._entries)

    def total_cash_out(self) -> float:
        """Total cash borrowed (repo direction)."""
        return sum(
            e.cash_amount for e in self._entries if e.direction == "repo"
        )

    def total_cash_in(self) -> float:
        """Total cash lent (reverse repo direction)."""
        return sum(
            e.cash_amount for e in self._entries if e.direction == "reverse"
        )

    def by_counterparty(self) -> list[RepoCounterpartyExposure]:
        """Aggregate exposure per counterparty."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            cp = e.counterparty
            if cp not in agg:
                agg[cp] = {"cash": 0.0, "rate_sum": 0.0, "count": 0}
            agg[cp]["cash"] += e.cash_amount
            agg[cp]["rate_sum"] += e.repo_rate
            agg[cp]["count"] += 1

        return [
            RepoCounterpartyExposure(
                counterparty=cp,
                total_cash=d["cash"],
                n_trades=d["count"],
                avg_rate=d["rate_sum"] / d["count"] if d["count"] > 0 else 0.0,
            )
            for cp, d in sorted(agg.items())
        ]

    def by_collateral_type(self) -> list[RepoCollateralSummary]:
        """Aggregate by GC vs special."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            ct = e.collateral_type
            if ct not in agg:
                agg[ct] = {"cash": 0.0, "rate_sum": 0.0, "count": 0}
            agg[ct]["cash"] += e.cash_amount
            agg[ct]["rate_sum"] += e.repo_rate
            agg[ct]["count"] += 1

        return [
            RepoCollateralSummary(
                collateral_type=ct,
                total_cash=d["cash"],
                avg_rate=d["rate_sum"] / d["count"] if d["count"] > 0 else 0.0,
                n_trades=d["count"],
            )
            for ct, d in sorted(agg.items())
        ]

    def gc_rate(self) -> float | None:
        """Weighted-average GC repo rate, or None if no GC trades."""
        gc = [e for e in self._entries if e.collateral_type == "GC"]
        if not gc:
            return None
        total_cash = sum(e.cash_amount for e in gc)
        if total_cash <= 0:
            return 0.0
        return sum(e.cash_amount * e.repo_rate for e in gc) / total_cash

    def special_rate(self, issuer: str) -> float | None:
        """Weighted-average special repo rate for a specific issuer."""
        sp = [
            e for e in self._entries
            if e.collateral_type == "special" and e.collateral_issuer == issuer
        ]
        if not sp:
            return None
        total_cash = sum(e.cash_amount for e in sp)
        if total_cash <= 0:
            return 0.0
        return sum(e.cash_amount * e.repo_rate for e in sp) / total_cash


# ---- Repo rate monitor ----

def repo_rate_monitor(
    current_rate: float,
    history: list[float],
    threshold: float = 2.0,
) -> ZScoreSignal:
    """Z-score the current repo rate vs history."""
    return _zscore(current_rate, history, threshold)


# ---- Cheapest-to-deliver repo ----

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
        return self.bond_price * self.repo_rate * self.term_days / 365.0

    @property
    def carry(self) -> float:
        dt = self.term_days / 365.0
        return self.coupon_rate * 100.0 * dt - self.financing_cost


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


# ---- Term vs overnight ----

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
    dt = term_days / 365.0
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


# ---- Fails tracking ----

@dataclass
class SettlementFail:
    """A single settlement fail."""
    counterparty: str
    issuer: str
    face_amount: float
    fail_date: date
    days_outstanding: int = 0
    penalty_rate_bps: float = 0.0

    @property
    def penalty_cost(self) -> float:
        """Penalty = face × penalty_rate × days / 365."""
        return (
            self.face_amount
            * (self.penalty_rate_bps / 10_000.0)
            * self.days_outstanding / 365.0
        )


class FailsTracker:
    """Track and cost settlement fails."""

    def __init__(self):
        self._fails: list[SettlementFail] = []

    def add(self, fail: SettlementFail) -> None:
        self._fails.append(fail)

    @property
    def fails(self) -> list[SettlementFail]:
        return list(self._fails)

    def __len__(self) -> int:
        return len(self._fails)

    def total_penalty(self) -> float:
        return sum(f.penalty_cost for f in self._fails)

    def total_face_outstanding(self) -> float:
        return sum(f.face_amount for f in self._fails)

    def by_counterparty(self) -> dict[str, float]:
        """Total fail face per counterparty."""
        result: dict[str, float] = {}
        for f in self._fails:
            result[f.counterparty] = result.get(f.counterparty, 0.0) + f.face_amount
        return result


# ---------------------------------------------------------------------------
# Maturity / Cash Ladder
# ---------------------------------------------------------------------------

@dataclass
class CashLadderBucket:
    """One bucket in the maturity ladder."""
    bucket: str          # "O/N", "1W", "1M", "3M", "6M", "1Y+"
    maturing_cash: float  # cash flowing in/out at this tenor
    n_trades: int
    avg_rate: float
    refinancing_cost: float  # cost to roll at current ON rate

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket, "maturing_cash": self.maturing_cash,
            "n_trades": self.n_trades, "avg_rate": self.avg_rate,
            "refinancing_cost": self.refinancing_cost,
        }


def cash_ladder(
    book: RepoBook,
    reference_date: date,
    overnight_rate: float = 0.0,
) -> list[CashLadderBucket]:
    """Build a maturity/cash ladder from the repo book.

    Groups positions by remaining tenor and computes the cash
    maturing in each bucket + cost to refinance at the overnight rate.

    Buckets: O/N (0-1d), 1W (2-7d), 1M (8-30d), 3M (31-90d),
             6M (91-180d), 1Y+ (181+d).
    """
    buckets_def = [
        ("O/N", 0, 1),
        ("1W", 2, 7),
        ("1M", 8, 30),
        ("3M", 31, 90),
        ("6M", 91, 180),
        ("1Y+", 181, 99999),
    ]

    result = []
    for label, lo, hi in buckets_def:
        matching = []
        for e in book.entries:
            remaining = e.term_days
            if e.start_date:
                elapsed = (reference_date - e.start_date).days
                remaining = max(0, e.term_days - elapsed)
            if lo <= remaining <= hi:
                matching.append(e)

        total_cash = sum(
            e.cash_amount * (1 if e.direction == "repo" else -1)
            for e in matching
        )
        avg_rate = (
            sum(e.repo_rate * e.cash_amount for e in matching)
            / sum(e.cash_amount for e in matching)
            if matching and sum(e.cash_amount for e in matching) > 0
            else 0.0
        )
        # Refinancing cost: if this bucket matures, roll at ON for same term
        mid_days = (lo + min(hi, 365)) / 2
        refi_cost = abs(total_cash) * overnight_rate * mid_days / 365.0

        result.append(CashLadderBucket(
            bucket=label, maturing_cash=total_cash,
            n_trades=len(matching), avg_rate=avg_rate,
            refinancing_cost=refi_cost,
        ))

    return result


# ---------------------------------------------------------------------------
# Repo Rate DV01
# ---------------------------------------------------------------------------

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
        dt = e.term_days / 365.0
        # Carry = sign × (coupon_income - cash × (repo_rate + shift) × dt)
        sign = 1.0 if e.direction == "repo" else -1.0
        coupon = e.face_amount * e.coupon_rate * dt
        financing_bumped = e.cash_amount * (e.repo_rate + shift) * dt
        bumped_c = sign * (coupon - financing_bumped)
        bumped_carry += bumped_c
        per_trade.append(bumped_c - base_c)

    return {
        "total_dv01": bumped_carry - base_carry,
        "base_carry": base_carry,
        "bumped_carry": bumped_carry,
        "per_trade_dv01": per_trade,
    }


# ---------------------------------------------------------------------------
# Carry P&L Decomposition
# ---------------------------------------------------------------------------

@dataclass
class CarryDecomposition:
    """Carry P&L split into components."""
    total_carry: float
    coupon_income: float
    repo_financing_cost: float
    specialness_benefit: float  # GC_cost - actual_cost (positive when on special)
    net_cash_position: float

    def to_dict(self) -> dict:
        return {
            "total_carry": self.total_carry,
            "coupon_income": self.coupon_income,
            "repo_financing_cost": self.repo_financing_cost,
            "specialness_benefit": self.specialness_benefit,
            "net_cash_position": self.net_cash_position,
        }


def carry_pnl_decomposition(
    book: RepoBook,
    gc_rate: float,
) -> CarryDecomposition:
    """Decompose book carry into coupon, repo cost, and specialness.

    coupon_income: total coupon earned on bonds held.
    repo_financing_cost: total interest paid on borrowed cash.
    specialness_benefit: savings from financing below GC (positive = good).
    """
    coupon_income = 0.0
    financing_cost = 0.0
    specialness = 0.0

    for e in book.entries:
        dt = e.term_days / 365.0
        sign = 1.0 if e.direction == "repo" else -1.0

        coupon = e.face_amount * e.coupon_rate * dt * sign
        financing = e.cash_amount * e.repo_rate * dt * sign
        # Specialness: what would financing cost at GC?
        gc_financing = e.cash_amount * gc_rate * dt * sign
        spec_benefit = gc_financing - financing  # positive when repo < GC

        coupon_income += coupon
        financing_cost += financing
        specialness += spec_benefit

    total = coupon_income - financing_cost
    net_cash = book.total_cash_out() - book.total_cash_in()

    return CarryDecomposition(
        total_carry=total,
        coupon_income=coupon_income,
        repo_financing_cost=financing_cost,
        specialness_benefit=specialness,
        net_cash_position=net_cash,
    )


# ---------------------------------------------------------------------------
# Rollover Risk
# ---------------------------------------------------------------------------

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
        extra_cost = on_cash * spike * days / 365.0
        annualised = spike_bps * (days / 365.0) if on_cash > 0 else 0.0
        results.append(RolloverScenario(
            scenario_name=name,
            on_rate_spike_bps=spike_bps,
            spike_duration_days=days,
            additional_cost=extra_cost,
            annualised_impact_bps=annualised,
        ))

    return results


# ---------------------------------------------------------------------------
# Counterparty Exposure Monitor
# ---------------------------------------------------------------------------

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
