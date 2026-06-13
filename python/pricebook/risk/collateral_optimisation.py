"""Collateral optimisation: LP solver to minimise posting cost across CSAs.

    from pricebook.risk.collateral_optimisation import CollateralOptimiser

References:
    Green (2015). XVA: Credit, Funding and Capital Valuation Adjustments.
    Lou (2017). Collateral optimisation with multiple credit support annexes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class CollateralAsset:
    """An asset available for collateral posting."""
    asset_id: str
    available_amount: float
    yield_rate: float
    funding_rate: float
    haircut_pct: float
    currency: str = "USD"

    @property
    def cost_per_unit(self) -> float:
        """Cost of posting one unit: funding - yield + haircut cost."""
        return self.funding_rate - self.yield_rate + self.haircut_pct * self.funding_rate

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CSARequirement:
    """One CSA's collateral requirement."""
    csa_id: str
    counterparty: str
    required_amount: float
    eligible_asset_ids: list[str]

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CollateralAllocation:
    """Allocation of one asset to one CSA."""
    csa_id: str
    asset_id: str
    amount: float
    cost: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CollateralAllocationResult:
    """Full optimisation result."""
    allocations: list[CollateralAllocation]
    total_cost: float
    cost_by_csa: dict[str, float]
    cost_by_asset: dict[str, float]
    naive_cost: float
    savings: float
    solver_status: str
    unmet_requirements: list[str]

    def to_dict(self) -> dict:
        return {
            "total_cost": self.total_cost,
            "naive_cost": self.naive_cost,
            "savings": self.savings,
            "solver_status": self.solver_status,
            "cost_by_csa": self.cost_by_csa,
            "cost_by_asset": self.cost_by_asset,
            "unmet_requirements": self.unmet_requirements,
            "allocations": [a.to_dict() for a in self.allocations],
        }


@dataclass
class SubstitutionImpact:
    """What-if: swap one asset for another in a CSA."""
    current_asset_id: str
    proposed_asset_id: str
    csa_id: str
    current_cost: float
    proposed_cost: float
    savings: float
    feasible: bool

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class StressResult:
    """Collateral adequacy under stress."""
    scenario_name: str
    haircut_shock_pct: float
    total_cost_stressed: float
    cost_increase: float
    margin_shortfall: float
    shortfall_csas: list[str]

    def to_dict(self) -> dict:
        return vars(self)


# ═══════════════════════════════════════════════════════════════
# Optimiser
# ═══════════════════════════════════════════════════════════════

class CollateralOptimiser:
    """LP-based collateral optimiser across multiple CSAs.

    Decision variables: x[i,j] = amount of asset j allocated to CSA i.
    Objective: min Σ cost_j × x[i,j]
    Constraints:
        Σ_j x[i,j] >= required[i]  (coverage)
        Σ_i x[i,j] <= available[j]  (inventory)
        x[i,j] = 0 if j not eligible for CSA i
    """

    def __init__(
        self,
        requirements: list[CSARequirement],
        assets: list[CollateralAsset],
    ):
        self.requirements = requirements
        self.assets = assets
        self._asset_map = {a.asset_id: a for a in assets}

    def optimise(self) -> CollateralAllocationResult:
        """Solve the LP."""
        n_csas = len(self.requirements)
        n_assets = len(self.assets)
        n_vars = n_csas * n_assets

        if n_vars == 0:
            return CollateralAllocationResult([], 0.0, {}, {}, 0.0, 0.0, "empty", [])

        # Variable index: x[i*n_assets + j] = amount of asset j to CSA i
        # Objective: minimize cost
        c = np.zeros(n_vars)
        for i in range(n_csas):
            for j, asset in enumerate(self.assets):
                idx = i * n_assets + j
                if asset.asset_id in self.requirements[i].eligible_asset_ids:
                    c[idx] = asset.cost_per_unit
                else:
                    c[idx] = 1e10  # effectively infinite cost for ineligible

        # Coverage constraints: Σ_j (1 - haircut_j) × x[i,j] >= required[i]
        # In linprog form: -Σ_j (1 - haircut_j) × x[i,j] <= -required[i]
        #
        # Fix T4-RISK27: pre-fix used `Σ_j x[i,j] >= required[i]` without
        # applying the haircut.  For an asset with 5% haircut, posting
        # $100 covers only $95 of the requirement, but pre-fix LP
        # treated $100 = $100 of coverage.  Result: every non-cash CSA
        # allocation was silently under-collateralised by the haircut
        # amount.  Now we multiply each x[i,j] by (1 - h_j) before
        # summing into the coverage constraint.
        A_ub_rows = []
        b_ub_rows = []

        for i, req in enumerate(self.requirements):
            row = np.zeros(n_vars)
            for j, asset in enumerate(self.assets):
                row[i * n_assets + j] = -(1.0 - asset.haircut_pct)
            A_ub_rows.append(row)
            b_ub_rows.append(-req.required_amount)

        # Availability constraints: Σ_i x[i,j] <= available[j]
        for j, asset in enumerate(self.assets):
            row = np.zeros(n_vars)
            for i in range(n_csas):
                row[i * n_assets + j] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(asset.available_amount)

        A_ub = np.array(A_ub_rows) if A_ub_rows else None
        b_ub = np.array(b_ub_rows) if b_ub_rows else None

        # Bounds: x >= 0
        bounds = [(0, None)] * n_vars

        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not result.success:
            return CollateralAllocationResult(
                [], 0.0, {}, {}, self._naive_cost(), 0.0,
                "infeasible", [r.csa_id for r in self.requirements],
            )

        # Extract allocations
        allocations = []
        cost_by_csa: dict[str, float] = {}
        cost_by_asset: dict[str, float] = {}

        for i, req in enumerate(self.requirements):
            csa_cost = 0.0
            for j, asset in enumerate(self.assets):
                idx = i * n_assets + j
                amount = result.x[idx]
                if amount > 1e-6:
                    cost = amount * asset.cost_per_unit
                    allocations.append(CollateralAllocation(req.csa_id, asset.asset_id, amount, cost))
                    csa_cost += cost
                    cost_by_asset[asset.asset_id] = cost_by_asset.get(asset.asset_id, 0.0) + cost
            cost_by_csa[req.csa_id] = csa_cost

        total_cost = sum(cost_by_csa.values())
        naive = self._naive_cost()

        # Check unmet requirements — use post-haircut value (must match coverage constraint).
        unmet = []
        for i, req in enumerate(self.requirements):
            allocated = sum(
                (1.0 - self.assets[j].haircut_pct) * result.x[i * n_assets + j]
                for j in range(n_assets)
            )
            if allocated < req.required_amount - 1e-6:
                unmet.append(req.csa_id)

        return CollateralAllocationResult(
            allocations=allocations,
            total_cost=total_cost,
            cost_by_csa=cost_by_csa,
            cost_by_asset=cost_by_asset,
            naive_cost=naive,
            savings=naive - total_cost,
            solver_status="optimal",
            unmet_requirements=unmet,
        )

    def _naive_cost(self) -> float:
        """Baseline: per-CSA greedy allocation respecting availability.

        Walk CSAs sequentially; for each, fill from cheapest-effective
        eligible asset until either the CSA is covered or supply runs
        out, then spill over to the next-cheapest.  This is what a
        non-LP human would do greedy-by-CSA, and is a feasible baseline
        for the LP to beat.

        Must use haircut-grossed cost (matches the LP's coverage
        constraint convention) so the comparison is apples-to-apples.
        """
        def effective_cost(a: CollateralAsset) -> float:
            return a.cost_per_unit / max(1.0 - a.haircut_pct, 1e-12)

        remaining = {a.asset_id: a.available_amount for a in self.assets}
        total = 0.0
        for req in self.requirements:
            eligible = sorted(
                (a for a in self.assets if a.asset_id in req.eligible_asset_ids),
                key=effective_cost,
            )
            net_to_cover = req.required_amount
            for asset in eligible:
                if net_to_cover <= 1e-12:
                    break
                # Max net coverage from this asset given remaining supply.
                max_net_from_asset = remaining[asset.asset_id] * (1.0 - asset.haircut_pct)
                net_take = min(net_to_cover, max_net_from_asset)
                gross_take = net_take / max(1.0 - asset.haircut_pct, 1e-12)
                total += gross_take * asset.cost_per_unit
                remaining[asset.asset_id] -= gross_take
                net_to_cover -= net_take
        return total

    def what_if_substitution(
        self,
        current_asset_id: str,
        proposed_asset_id: str,
        csa_id: str,
        amount: float,
    ) -> SubstitutionImpact:
        """Cost impact of swapping one asset for another."""
        current = self._asset_map.get(current_asset_id)
        proposed = self._asset_map.get(proposed_asset_id)
        req = next((r for r in self.requirements if r.csa_id == csa_id), None)

        if not current or not proposed or not req:
            return SubstitutionImpact(current_asset_id, proposed_asset_id, csa_id, 0, 0, 0, False)

        feasible = proposed.asset_id in req.eligible_asset_ids and proposed.available_amount >= amount
        current_cost = amount * current.cost_per_unit
        proposed_cost = amount * proposed.cost_per_unit

        return SubstitutionImpact(
            current_asset_id, proposed_asset_id, csa_id,
            current_cost, proposed_cost,
            current_cost - proposed_cost, feasible,
        )


def stress_collateral(
    result: CollateralAllocationResult,
    requirements: list[CSARequirement],
    assets: list[CollateralAsset],
    scenarios: list[tuple[str, float]] | None = None,
) -> list[StressResult]:
    """Stress test collateral allocations.

    Args:
        scenarios: list of (name, haircut_shock_pct). Default: mild/moderate/severe/crisis.
    """
    if scenarios is None:
        scenarios = [
            ("mild", 0.01),
            ("moderate", 0.02),
            ("severe", 0.05),
            ("crisis", 0.10),
        ]

    results = []
    for name, shock in scenarios:
        stressed_assets = []
        for a in assets:
            stressed = CollateralAsset(
                a.asset_id, a.available_amount,
                a.yield_rate, a.funding_rate,
                a.haircut_pct + shock, a.currency,
            )
            stressed_assets.append(stressed)

        opt = CollateralOptimiser(requirements, stressed_assets)
        stressed_result = opt.optimise()

        shortfall_csas = stressed_result.unmet_requirements
        shortfall = sum(
            r.required_amount - sum(
                a.amount for a in stressed_result.allocations if a.csa_id == r.csa_id
            )
            for r in requirements
            if r.csa_id in shortfall_csas
        )

        results.append(StressResult(
            scenario_name=name,
            haircut_shock_pct=shock,
            total_cost_stressed=stressed_result.total_cost,
            cost_increase=stressed_result.total_cost - result.total_cost,
            margin_shortfall=max(shortfall, 0.0),
            shortfall_csas=shortfall_csas,
        ))

    return results
