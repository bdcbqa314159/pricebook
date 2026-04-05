"""Waterfall engine: cashflow priority allocation and triggers.

Allocates cashflows top-down through tranches. Triggers divert
cashflows when coverage tests fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tranche:
    """A single tranche in a waterfall."""

    name: str
    target_notional: float
    coupon_rate: float
    seniority: int  # lower = more senior (0 = most senior)
    balance: float = 0.0
    interest_due: float = 0.0
    interest_paid: float = 0.0
    principal_paid: float = 0.0

    @property
    def outstanding(self) -> float:
        return self.target_notional - self.principal_paid

    @property
    def is_paid_off(self) -> bool:
        return self.outstanding <= 0.01


@dataclass
class Trigger:
    """A coverage test that can divert cashflows.

    If the test metric falls below the threshold, the trigger fires
    and redirects cashflows according to the action.
    """

    name: str
    metric: str  # "oc_ratio" or "ic_ratio"
    threshold: float
    action: str = "divert_to_senior"  # what happens when breached

    def is_breached(self, value: float) -> bool:
        return value < self.threshold


class WaterfallEngine:
    """Allocates cashflows through a priority waterfall.

    Tranches are ordered by seniority (most senior first).
    Triggers can redirect cashflows when coverage tests fail.
    """

    def __init__(
        self,
        tranches: list[Tranche],
        triggers: list[Trigger] | None = None,
    ):
        self.tranches = sorted(tranches, key=lambda t: t.seniority)
        self.triggers = triggers or []

    def allocate(
        self,
        total_cashflow: float,
        collateral_balance: float = 0.0,
    ) -> dict[str, Any]:
        """Allocate cashflows through the waterfall.

        Args:
            total_cashflow: available cash to distribute.
            collateral_balance: total collateral for coverage tests.

        Returns:
            dict with per-tranche allocation and trigger status.
        """
        remaining = total_cashflow
        allocations = {}
        trigger_status = {}

        # Check triggers
        for trigger in self.triggers:
            if trigger.metric == "oc_ratio":
                # OC = collateral / senior_outstanding
                senior_outstanding = sum(t.outstanding for t in self.tranches)
                ratio = collateral_balance / max(senior_outstanding, 1e-10)
                trigger_status[trigger.name] = {
                    "metric": ratio,
                    "threshold": trigger.threshold,
                    "breached": trigger.is_breached(ratio),
                }
            elif trigger.metric == "ic_ratio":
                # IC = total_cashflow / total_interest_due
                total_interest = sum(
                    t.outstanding * t.coupon_rate for t in self.tranches
                )
                ratio = total_cashflow / max(total_interest, 1e-10)
                trigger_status[trigger.name] = {
                    "metric": ratio,
                    "threshold": trigger.threshold,
                    "breached": trigger.is_breached(ratio),
                }

        # Any trigger breached → divert to senior
        divert = any(ts["breached"] for ts in trigger_status.values())

        # Interest waterfall (senior first)
        for tranche in self.tranches:
            interest = tranche.outstanding * tranche.coupon_rate
            paid = min(interest, remaining)
            tranche.interest_due = interest
            tranche.interest_paid = paid
            remaining -= paid
            allocations[tranche.name] = {"interest": paid, "principal": 0.0}

            if divert and tranche.seniority > 0:
                # Diversion: skip junior interest, redirect to senior principal
                break

        # Principal waterfall
        if not divert:
            # Normal: allocate principal senior-first
            for tranche in self.tranches:
                if tranche.is_paid_off:
                    continue
                principal = min(tranche.outstanding, remaining)
                tranche.principal_paid += principal
                remaining -= principal
                allocations[tranche.name]["principal"] = principal
        else:
            # Diverted: all remaining goes to most senior
            if self.tranches:
                senior = self.tranches[0]
                principal = min(senior.outstanding, remaining)
                senior.principal_paid += principal
                remaining -= principal
                allocations[senior.name]["principal"] = principal

        return {
            "allocations": allocations,
            "remaining": remaining,
            "trigger_status": trigger_status,
            "diverted": divert,
        }

    def reset(self) -> None:
        """Reset all tranche balances for re-simulation."""
        for t in self.tranches:
            t.interest_due = 0.0
            t.interest_paid = 0.0
            t.principal_paid = 0.0


# ---------------------------------------------------------------------------
# Structured note helpers
# ---------------------------------------------------------------------------


@dataclass
class AutocallObservation:
    """One observation date for an autocall note."""

    observation_date: float  # in years from start
    barrier: float           # autocall if reference > barrier
    redemption: float = 1.0  # fraction of notional returned


def autocall_payoff(
    reference_path: list[float],
    observations: list[AutocallObservation],
    notional: float = 1.0,
    coupon_rate: float = 0.05,
) -> dict[str, Any]:
    """Determine autocall outcome from a reference rate/price path.

    Returns dict with: called (bool), call_date (float), payout (float).
    """
    for obs in observations:
        # Find the path value nearest to observation time
        idx = min(int(obs.observation_date * len(reference_path) /
                      max(observations[-1].observation_date, 1e-10)),
                  len(reference_path) - 1)
        idx = max(0, idx)

        if reference_path[idx] >= obs.barrier:
            # Autocall triggered
            payout = notional * obs.redemption + notional * coupon_rate * obs.observation_date
            return {
                "called": True,
                "call_date": obs.observation_date,
                "payout": payout,
            }

    # Not called: return notional at maturity
    return {
        "called": False,
        "call_date": observations[-1].observation_date,
        "payout": notional,
    }
