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

import math
from dataclasses import dataclass, field
from datetime import date


# ---- Z-score core (same pattern as equity_rv / commodity_rv) ----

@dataclass
class ZScoreSignal:
    current: float
    mean: float
    std: float
    z_score: float | None
    signal: str


def _zscore(current: float, history: list[float], threshold: float = 2.0) -> ZScoreSignal:
    if not history or len(history) < 2:
        return ZScoreSignal(current, current, 0.0, None, "fair")
    mean = sum(history) / len(history)
    var = sum((h - mean) ** 2 for h in history) / len(history)
    std = math.sqrt(var) if var > 0 else 0.0
    z = (current - mean) / std if std > 1e-12 else None
    if z is not None and abs(z) >= threshold:
        signal = "rich" if z > 0 else "cheap"
    else:
        signal = "fair"
    return ZScoreSignal(current, mean, std, z, signal)


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
