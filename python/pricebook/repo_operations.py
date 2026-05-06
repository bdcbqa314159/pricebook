"""Repo operations: settlement fails, collateral, margin, netting, substitution.

Operational functions split from repo_desk.py for maintainability.
All symbols are re-exported from repo_desk.py for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

# Lazy import to avoid circular: repo_desk imports from here
def _get_repo_trade_entry():
    from pricebook.repo_desk import RepoTradeEntry
    return RepoTradeEntry


# Fail state machine
FAIL_STATES = ["open", "investigating", "resolving", "resolved", "bought_in"]


@dataclass
class CollateralPosition:
    """A bond in the collateral pool."""
    issuer: str
    face_amount: float
    pledged_to: dict[str, float] = field(default_factory=dict)  # {counterparty: face_pledged}

    @property
    def total_pledged(self) -> float:
        return sum(self.pledged_to.values())

    @property
    def available(self) -> float:
        return max(0, self.face_amount - self.total_pledged)

    def pledge(self, counterparty: str, amount: float) -> None:
        if amount > self.available:
            raise ValueError(
                f"Cannot pledge {amount}: only {self.available} available "
                f"({self.face_amount} total, {self.total_pledged} pledged)"
            )
        self.pledged_to[counterparty] = self.pledged_to.get(counterparty, 0) + amount

    def release(self, counterparty: str, amount: float) -> None:
        current = self.pledged_to.get(counterparty, 0)
        self.pledged_to[counterparty] = max(0, current - amount)

    def to_dict(self) -> dict:
        return {
            "issuer": self.issuer, "face": self.face_amount,
            "pledged": self.total_pledged, "available": self.available,
            "by_cp": dict(self.pledged_to),
        }


class CollateralPool:
    """Tracks bond inventory: what's pledged, what's free (Gap 4)."""

    def __init__(self):
        self._positions: dict[str, CollateralPosition] = {}

    def add_inventory(self, issuer: str, face_amount: float) -> None:
        if issuer in self._positions:
            self._positions[issuer].face_amount += face_amount
        else:
            self._positions[issuer] = CollateralPosition(issuer, face_amount)

    def pledge(self, issuer: str, counterparty: str, amount: float) -> None:
        if issuer not in self._positions:
            raise ValueError(f"No inventory for {issuer}")
        self._positions[issuer].pledge(counterparty, amount)

    def release(self, issuer: str, counterparty: str, amount: float) -> None:
        if issuer in self._positions:
            self._positions[issuer].release(counterparty, amount)

    def available(self, issuer: str) -> float:
        pos = self._positions.get(issuer)
        return pos.available if pos else 0.0

    def total_available(self) -> float:
        return sum(p.available for p in self._positions.values())

    def summary(self) -> list[dict]:
        return [p.to_dict() for p in sorted(self._positions.values(), key=lambda p: p.issuer)]

    def can_pledge(self, issuer: str, amount: float) -> bool:
        return self.available(issuer) >= amount


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


@dataclass
class HaircutAdjustment:
    """Haircut adjusted for market stress."""
    base_haircut_pct: float
    vol_multiplier: float
    stress_add_on_pct: float
    adjusted_haircut_pct: float
    regime: str  # "normal", "elevated", "stressed"

    def to_dict(self) -> dict:
        return {
            "base": self.base_haircut_pct, "vol_mult": self.vol_multiplier,
            "stress_add": self.stress_add_on_pct,
            "adjusted": self.adjusted_haircut_pct, "regime": self.regime,
        }


def dynamic_haircut(
    base_haircut_pct: float,
    current_vol: float,
    normal_vol: float = 0.05,
    stress_threshold: float = 2.0,
) -> HaircutAdjustment:
    """Adjust haircut based on market volatility.

    haircut_adj = base × max(1, vol / normal_vol)
    Plus stress add-on when vol > stress_threshold × normal_vol.

    Args:
        base_haircut_pct: normal market haircut (e.g. 2.0 for treasuries).
        current_vol: current realised or implied vol of the collateral.
        normal_vol: long-run average vol.
        stress_threshold: multiplier at which stress add-on kicks in.
    """
    vol_ratio = max(1.0, current_vol / normal_vol) if normal_vol > 0 else 1.0
    adjusted = base_haircut_pct * vol_ratio

    if current_vol > stress_threshold * normal_vol:
        stress_add = base_haircut_pct * 0.5  # +50% of base in stress
        adjusted += stress_add
        regime = "stressed"
    elif current_vol > 1.5 * normal_vol:
        stress_add = base_haircut_pct * 0.2  # +20% in elevated
        adjusted += stress_add
        regime = "elevated"
    else:
        stress_add = 0.0
        regime = "normal"

    return HaircutAdjustment(
        base_haircut_pct=base_haircut_pct,
        vol_multiplier=vol_ratio,
        stress_add_on_pct=stress_add,
        adjusted_haircut_pct=adjusted,
        regime=regime,
    )


@dataclass
class MarginCallScenario:
    """Result of a margin call simulation under rate shock."""
    scenario_name: str
    rate_shock_bps: float
    total_margin_call: float
    n_positions_affected: int
    largest_single_call: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name, "shock_bps": self.rate_shock_bps,
            "total_call": self.total_margin_call,
            "n_affected": self.n_positions_affected,
            "largest_call": self.largest_single_call,
        }


def margin_call_simulation(
    book: RepoBook,
    haircut_pct: float = 2.0,
    scenarios: list[tuple[str, float]] | None = None,
) -> list[MarginCallScenario]:
    """Simulate margin calls under repo rate shocks.

    When rates move, bond prices move, and the margin (haircut × notional)
    changes. The desk needs to post/receive the difference.

    Approximate: ΔMargin ≈ cash_amount × duration × Δrate × haircut adjustment.

    Args:
        haircut_pct: base haircut percentage.
        scenarios: list of (name, rate_shock_bps).
    """
    if scenarios is None:
        scenarios = [
            ("mild", 25), ("moderate", 50), ("severe", 100), ("crisis", 200),
        ]

    results = []
    for name, shock_bps in scenarios:
        shock = shock_bps / 10_000.0
        total_call = 0.0
        largest = 0.0
        n_affected = 0

        for e in book.entries:
            # Rough price impact: ΔP ≈ -duration × Δy × price
            # Use term_days as rough duration proxy (scaled)
            duration_proxy = min(e.term_days / 365.0, 10.0) * 5.0  # rough
            price_move = e.bond_price * duration_proxy * shock / 100.0
            margin_change = abs(e.face_amount * price_move / 100.0 * haircut_pct / 100.0)

            if margin_change > 0:
                total_call += margin_change
                largest = max(largest, margin_change)
                n_affected += 1

        results.append(MarginCallScenario(
            scenario_name=name, rate_shock_bps=shock_bps,
            total_margin_call=total_call, n_positions_affected=n_affected,
            largest_single_call=largest,
        ))

    return results


@dataclass
class FailResolution:
    """Resolution path for a settlement fail."""
    counterparty: str
    issuer: str
    face_amount: float
    days_outstanding: int
    penalty_cost: float
    category: str          # "collateral", "system", "counterparty"
    buy_in_cost: float
    escalated: bool

    def to_dict(self) -> dict:
        return {
            "counterparty": self.counterparty, "issuer": self.issuer,
            "face": self.face_amount, "days_out": self.days_outstanding,
            "penalty": self.penalty_cost, "category": self.category,
            "buy_in_cost": self.buy_in_cost, "escalated": self.escalated,
        }


def fail_workflow(
    tracker: FailsTracker,
    current_prices: dict[str, float] | None = None,
    contract_prices: dict[str, float] | None = None,
    escalation_days: int = 5,
) -> list[FailResolution]:
    """Process settlement fails: categorise, price buy-in, escalate.

    Buy-in cost = max(0, (current - contract) / 100 × face).
    Categories: system (≤1d), collateral (2-3d), counterparty (4d+).
    Escalate if days > escalation_days.
    """
    if current_prices is None:
        current_prices = {}
    if contract_prices is None:
        contract_prices = {}

    results = []
    for f in tracker.fails:
        curr = current_prices.get(f.issuer, 100.0)
        contract = contract_prices.get(f.issuer, 100.0)
        buy_in = max(0, (curr - contract) / 100.0 * f.face_amount)

        if f.days_outstanding <= 1:
            category = "system"
        elif f.days_outstanding <= 3:
            category = "collateral"
        else:
            category = "counterparty"

        results.append(FailResolution(
            counterparty=f.counterparty, issuer=f.issuer,
            face_amount=f.face_amount, days_outstanding=f.days_outstanding,
            penalty_cost=f.penalty_cost, category=category,
            buy_in_cost=buy_in, escalated=f.days_outstanding > escalation_days,
        ))

    return results


@dataclass
class SubstitutionCandidate:
    """A substitute bond ranked by cost."""
    bond_id: str
    repo_rate: float
    haircut_pct: float
    cost_vs_original_bps: float
    available: bool

    def to_dict(self) -> dict:
        return {
            "bond": self.bond_id, "repo_rate": self.repo_rate,
            "haircut": self.haircut_pct, "cost_bp": self.cost_vs_original_bps,
            "available": self.available,
        }


def find_substitutes(
    failed_repo_rate: float,
    alternatives: dict[str, tuple[float, float, bool]],
) -> list[SubstitutionCandidate]:
    """Find substitute collateral, sorted by cost.

    Args:
        failed_repo_rate: repo rate on the failed trade.
        alternatives: {bond_id: (repo_rate, haircut_pct, available)}.
    """
    candidates = []
    for bond_id, (rate, haircut, avail) in alternatives.items():
        cost_bp = (rate - failed_repo_rate) * 10_000
        candidates.append(SubstitutionCandidate(
            bond_id=bond_id, repo_rate=rate,
            haircut_pct=haircut, cost_vs_original_bps=cost_bp,
            available=avail,
        ))
    return sorted(candidates, key=lambda c: c.cost_vs_original_bps)


class _RepoBookMixin:
    """Added to RepoBook via monkey-patch below."""
    pass


def _repo_book_to_dict(self) -> dict:
    """Serialise the RepoBook."""
    entries = []
    for e in self._entries:
        entries.append({
            "counterparty": e.counterparty,
            "collateral_issuer": e.collateral_issuer,
            "collateral_type": e.collateral_type,
            "face_amount": e.face_amount,
            "bond_price": e.bond_price,
            "repo_rate": e.repo_rate,
            "term_days": e.term_days,
            "coupon_rate": e.coupon_rate,
            "direction": e.direction,
            "start_date": e.start_date.isoformat() if e.start_date else None,
        })
    return {"type": "repo_book", "params": {
        "name": self.name, "entries": entries,
    }}


@classmethod
def _repo_book_from_dict(cls, d: dict) -> "RepoBook":
    """Deserialise a RepoBook."""
    p = d["params"]
    book = cls(name=p.get("name", "repo_book"))
    for e in p.get("entries", []):
        sd = date.fromisoformat(e["start_date"]) if e.get("start_date") else None
        RTE = _get_repo_trade_entry()
        book.add(RTE(
            counterparty=e["counterparty"],
            collateral_issuer=e["collateral_issuer"],
            collateral_type=e.get("collateral_type", "GC"),
            face_amount=e["face_amount"],
            bond_price=e["bond_price"],
            repo_rate=e["repo_rate"],
            term_days=e["term_days"],
            coupon_rate=e.get("coupon_rate", 0.0),
            direction=e.get("direction", "repo"),
            start_date=sd,
        ))
    return book


@dataclass
class NettingResult:
    """Net exposure after netting repos with same counterparty."""
    counterparty: str
    gross_repo: float
    gross_reverse: float
    net_exposure: float
    n_trades: int

    def to_dict(self) -> dict:
        return {
            "counterparty": self.counterparty,
            "gross_repo": self.gross_repo,
            "gross_reverse": self.gross_reverse,
            "net": self.net_exposure,
            "n_trades": self.n_trades,
        }


def netting_by_counterparty(book: RepoBook) -> list[NettingResult]:
    """Compute net exposure per counterparty after netting.

    Under ISDA/GMRA netting agreements, repo and reverse repo
    with the same counterparty offset.
    """
    by_cp: dict[str, dict] = {}
    for e in book.entries:
        cp = e.counterparty
        if cp not in by_cp:
            by_cp[cp] = {"repo": 0.0, "reverse": 0.0, "n": 0}
        if e.direction == "repo":
            by_cp[cp]["repo"] += e.cash_amount
        else:
            by_cp[cp]["reverse"] += e.cash_amount
        by_cp[cp]["n"] += 1

    return [
        NettingResult(
            counterparty=cp,
            gross_repo=d["repo"],
            gross_reverse=d["reverse"],
            net_exposure=abs(d["repo"] - d["reverse"]),
            n_trades=d["n"],
        )
        for cp, d in sorted(by_cp.items())
    ]


@dataclass
class FailState:
    """Settlement fail with lifecycle state."""
    counterparty: str
    issuer: str
    face_amount: float
    fail_date: date
    days_outstanding: int
    state: str = "open"       # FAIL_STATES
    buy_in_triggered: bool = False
    buy_in_cost: float = 0.0

    def advance(self, new_state: str) -> None:
        if new_state not in FAIL_STATES:
            raise ValueError(f"Invalid state '{new_state}'. Must be one of {FAIL_STATES}")
        self.state = new_state

    def trigger_buy_in(self, current_price: float, contract_price: float) -> None:
        self.buy_in_triggered = True
        self.buy_in_cost = max(0, (current_price - contract_price) / 100.0 * self.face_amount)
        self.state = "bought_in"

    def to_dict(self) -> dict:
        return {
            "counterparty": self.counterparty, "issuer": self.issuer,
            "face": self.face_amount, "days": self.days_outstanding,
            "state": self.state, "buy_in": self.buy_in_triggered,
            "buy_in_cost": self.buy_in_cost,
        }


def auto_escalate_fails(
    fails: list[FailState],
    investigate_after: int = 2,
    resolve_after: int = 5,
    buy_in_after: int = 10,
    current_prices: dict[str, float] | None = None,
    contract_prices: dict[str, float] | None = None,
) -> list[FailState]:
    """Auto-advance fail states based on days outstanding."""
    if current_prices is None:
        current_prices = {}
    if contract_prices is None:
        contract_prices = {}

    for f in fails:
        if f.state == "resolved" or f.state == "bought_in":
            continue
        if f.days_outstanding >= buy_in_after:
            curr = current_prices.get(f.issuer, 100.0)
            contract = contract_prices.get(f.issuer, 100.0)
            f.trigger_buy_in(curr, contract)
        elif f.days_outstanding >= resolve_after:
            f.advance("resolving")
        elif f.days_outstanding >= investigate_after:
            f.advance("investigating")

    return fails
