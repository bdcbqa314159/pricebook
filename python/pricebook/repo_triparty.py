"""Tri-party repo: agent-based collateral management.

In a tri-party repo, a clearing agent (BNY Mellon, Euroclear, JP Morgan)
sits between the cash lender and borrower. The agent:
  - Holds the collateral
  - Applies eligible collateral schedules
  - Rebalances daily (substitution within eligible set)
  - Computes independent margin
  - Charges agent fees

    from pricebook.repo_triparty import (
        TriPartyAgent, EligibilitySchedule, TriPartyRepo,
        allocate_collateral,
    )

References:
    ICMA (2022). Guide to Best Practice in the European Repo Market.
    Federal Reserve Bank of NY. Tri-Party Repo Infrastructure Reform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


# ---------------------------------------------------------------------------
# Eligibility schedule
# ---------------------------------------------------------------------------

@dataclass
class EligibilityRule:
    """One rule in a collateral eligibility schedule."""
    asset_class: str          # "govt", "agency", "ig_corp", "hy_corp", "equity"
    min_rating: str           # "AAA", "AA", "A", "BBB", etc.
    max_maturity_years: float  # max remaining maturity
    haircut_pct: float        # haircut for this asset class
    concentration_limit_pct: float  # max % of pool from this class

    def to_dict(self) -> dict:
        return {
            "asset_class": self.asset_class, "min_rating": self.min_rating,
            "max_maturity": self.max_maturity_years, "haircut": self.haircut_pct,
            "concentration": self.concentration_limit_pct,
        }


class EligibilitySchedule:
    """Defines what collateral a tri-party agent accepts.

    Each counterparty pair has a schedule specifying:
    - Which asset classes are eligible
    - Minimum credit rating per class
    - Maximum maturity
    - Haircut per class
    - Concentration limits
    """

    def __init__(self, name: str, rules: list[EligibilityRule] | None = None):
        self.name = name
        self.rules = rules or []

    def is_eligible(self, asset_class: str, rating: str = "AAA",
                     maturity_years: float = 5.0) -> bool:
        """Check if a specific asset is eligible under this schedule."""
        rating_order = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
                         "BBB+", "BBB", "BBB-", "BB+", "BB", "B", "CCC"]
        for rule in self.rules:
            if rule.asset_class == asset_class:
                # Check rating
                try:
                    if rating_order.index(rating) <= rating_order.index(rule.min_rating):
                        # Check maturity
                        if maturity_years <= rule.max_maturity_years:
                            return True
                except ValueError:
                    pass
        return False

    def haircut_for(self, asset_class: str) -> float:
        """Get the haircut for an asset class."""
        for rule in self.rules:
            if rule.asset_class == asset_class:
                return rule.haircut_pct
        return 10.0  # default high haircut for unknown

    @classmethod
    def standard_govt(cls) -> "EligibilitySchedule":
        """Standard government bond eligible schedule."""
        return cls("govt_standard", [
            EligibilityRule("govt", "AAA", 30.0, 2.0, 100.0),
            EligibilityRule("agency", "AA", 10.0, 3.0, 50.0),
        ])

    @classmethod
    def broad_ig(cls) -> "EligibilitySchedule":
        """Broad investment-grade eligible schedule."""
        return cls("broad_ig", [
            EligibilityRule("govt", "AAA", 30.0, 2.0, 100.0),
            EligibilityRule("agency", "AA", 10.0, 3.0, 50.0),
            EligibilityRule("ig_corp", "BBB", 7.0, 5.0, 30.0),
        ])

    def to_dict(self) -> dict:
        return {"name": self.name, "rules": [r.to_dict() for r in self.rules]}


# ---------------------------------------------------------------------------
# Tri-party agent
# ---------------------------------------------------------------------------

@dataclass
class AgentFees:
    """Tri-party agent fee structure."""
    management_fee_bps: float = 1.0    # annual basis fee on notional
    transaction_fee: float = 50.0       # per substitution
    custody_fee_bps: float = 0.5       # annual custody on collateral

    @property
    def total_annual_bps(self) -> float:
        return self.management_fee_bps + self.custody_fee_bps

    def period_cost(self, notional: float, days: int, n_substitutions: int = 0) -> float:
        """Total agent cost for a period."""
        annual_cost = notional * self.total_annual_bps / 10_000.0
        period_cost = annual_cost * days / 365.0
        sub_cost = n_substitutions * self.transaction_fee
        return period_cost + sub_cost

    def to_dict(self) -> dict:
        return {"mgmt_bps": self.management_fee_bps,
                "txn_fee": self.transaction_fee,
                "custody_bps": self.custody_fee_bps}


class TriPartyAgent:
    """A tri-party clearing agent (BNY, Euroclear, JPM).

    The agent manages the collateral pool between the two parties,
    applies eligibility schedules, computes margin, and charges fees.
    """

    AGENTS = {
        "BNY": AgentFees(1.0, 50.0, 0.5),
        "Euroclear": AgentFees(1.5, 75.0, 0.75),
        "JPM": AgentFees(1.0, 40.0, 0.5),
    }

    def __init__(self, name: str, fees: AgentFees | None = None):
        self.name = name
        self.fees = fees or self.AGENTS.get(name, AgentFees())

    def to_dict(self) -> dict:
        return {"name": self.name, "fees": self.fees.to_dict()}


# ---------------------------------------------------------------------------
# Tri-party repo trade
# ---------------------------------------------------------------------------

@dataclass
class CollateralAllocation:
    """One piece of collateral allocated by the agent."""
    bond_id: str
    asset_class: str
    face_amount: float
    market_value: float
    haircut_pct: float
    collateral_value: float  # market_value × (1 - haircut)

    def to_dict(self) -> dict:
        return {
            "bond": self.bond_id, "class": self.asset_class,
            "face": self.face_amount, "mv": self.market_value,
            "haircut": self.haircut_pct, "value": self.collateral_value,
        }


class TriPartyRepo:
    """Tri-party repo: agent-managed collateral.

    Unlike bilateral repos where collateral is fixed, the agent can
    substitute collateral daily within the eligibility schedule.

    Args:
        cash_lender: who provides cash.
        cash_borrower: who receives cash and pledges collateral.
        agent: TriPartyAgent managing the collateral.
        cash_amount: amount of cash lent.
        repo_rate: agreed repo rate.
        term_days: repo term (0 = open).
        start_date: trade date.
        schedule: EligibilitySchedule defining acceptable collateral.
    """

    def __init__(
        self,
        cash_lender: str,
        cash_borrower: str,
        agent: TriPartyAgent,
        cash_amount: float,
        repo_rate: float,
        term_days: int = 1,
        start_date: date | None = None,
        schedule: EligibilitySchedule | None = None,
        cash_currency: str = "USD",
    ):
        self.cash_lender = cash_lender
        self.cash_borrower = cash_borrower
        self.agent = agent
        self.cash_amount = cash_amount
        self.repo_rate = repo_rate
        self.term_days = term_days
        self.start_date = start_date
        self.schedule = schedule or EligibilitySchedule.standard_govt()
        self.cash_currency = cash_currency
        self._allocations: list[CollateralAllocation] = []

    @property
    def interest(self) -> float:
        return self.cash_amount * self.repo_rate * self.term_days / 360.0

    @property
    def repurchase_amount(self) -> float:
        return self.cash_amount + self.interest

    @property
    def agent_cost(self) -> float:
        return self.agent.fees.period_cost(self.cash_amount, self.term_days)

    @property
    def all_in_cost(self) -> float:
        """Interest + agent fees."""
        return self.interest + self.agent_cost

    @property
    def all_in_rate(self) -> float:
        """Effective rate including agent fees."""
        if self.cash_amount <= 0 or self.term_days <= 0:
            return self.repo_rate
        total_cost = self.all_in_cost
        return total_cost / self.cash_amount * 360.0 / self.term_days

    @property
    def collateral_value(self) -> float:
        """Total collateral value (after haircuts)."""
        return sum(a.collateral_value for a in self._allocations)

    @property
    def margin_excess(self) -> float:
        """Excess collateral over required (positive = over-collateralised)."""
        return self.collateral_value - self.cash_amount

    def allocate(self, allocation: CollateralAllocation) -> None:
        """Agent allocates a piece of collateral."""
        self._allocations.append(allocation)

    def substitute(
        self,
        remove_bond: str,
        new_allocation: CollateralAllocation,
    ) -> float:
        """Agent substitutes one bond for another. Returns substitution cost."""
        self._allocations = [a for a in self._allocations if a.bond_id != remove_bond]
        self._allocations.append(new_allocation)
        return self.agent.fees.transaction_fee

    @property
    def allocations(self) -> list[CollateralAllocation]:
        return list(self._allocations)

    def pv(self, discount_curve, reference_date: date | None = None) -> float:
        """PV from cash borrower's perspective."""
        from datetime import timedelta
        ref = reference_date or self.start_date or date.today()
        sd = ref + timedelta(days=1)  # T+1
        mat = sd + timedelta(days=self.term_days) if self.term_days > 0 else None
        if mat is None:
            return 0.0
        df = discount_curve.df(mat)
        return self.cash_amount - df * self.repurchase_amount

    def pv_ctx(self, ctx) -> float:
        return self.pv(ctx.discount_curve)

    def to_dict(self) -> dict:
        return {"type": "tri_party_repo", "params": {
            "cash_lender": self.cash_lender,
            "cash_borrower": self.cash_borrower,
            "agent": self.agent.to_dict(),
            "cash_amount": self.cash_amount,
            "repo_rate": self.repo_rate,
            "term_days": self.term_days,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "schedule": self.schedule.to_dict(),
            "cash_currency": self.cash_currency,
            "allocations": [a.to_dict() for a in self._allocations],
        }}


# ---------------------------------------------------------------------------
# Collateral allocation optimiser
# ---------------------------------------------------------------------------

def allocate_collateral(
    cash_needed: float,
    available_bonds: list[dict],
    schedule: EligibilitySchedule,
) -> list[CollateralAllocation]:
    """Optimise collateral allocation to cover cash_needed.

    Greedy: allocate cheapest-haircut bonds first to minimise
    collateral usage.

    Args:
        cash_needed: cash amount to collateralise.
        available_bonds: [{bond_id, asset_class, face, market_value, rating, maturity_years}].
        schedule: eligibility rules.

    Returns:
        List of allocations that cover the cash_needed.
    """
    # Filter eligible and sort by haircut (cheapest first)
    eligible = []
    for b in available_bonds:
        if schedule.is_eligible(b["asset_class"], b.get("rating", "AAA"),
                                  b.get("maturity_years", 5.0)):
            haircut = schedule.haircut_for(b["asset_class"])
            eligible.append((b, haircut))

    eligible.sort(key=lambda x: x[1])  # cheapest haircut first

    allocations = []
    remaining = cash_needed

    for bond_info, haircut in eligible:
        if remaining <= 0:
            break
        mv = bond_info["market_value"]
        coll_value = mv * (1 - haircut / 100.0)

        if coll_value <= 0:
            continue

        # How much of this bond do we need?
        if coll_value >= remaining:
            # Partial: only need fraction
            fraction = remaining / coll_value
            used_mv = mv * fraction
            used_face = bond_info["face"] * fraction
        else:
            used_mv = mv
            used_face = bond_info["face"]

        alloc = CollateralAllocation(
            bond_id=bond_info["bond_id"],
            asset_class=bond_info["asset_class"],
            face_amount=used_face,
            market_value=used_mv,
            haircut_pct=haircut,
            collateral_value=used_mv * (1 - haircut / 100.0),
        )
        allocations.append(alloc)
        remaining -= alloc.collateral_value

    return allocations
