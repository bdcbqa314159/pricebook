"""Regression for L2 phase-2 audit of `risk.collateral_optimisation`:

Pre-fix `CollateralOptimiser` coverage constraint
``Σ_j x[i,j] >= required[i]`` did NOT apply the haircut.  For a CSA
needing $100 covered by an asset with 5% haircut, the LP would
allocate exactly $100 of gross asset — but only $95 of *post-haircut*
value counts as collateral.  Net result: a $5 silent shortfall on
every non-cash allocation.

Fix: multiply each x[i,j] by (1 - haircut_j) in the coverage
constraint, so the requirement is satisfied in post-haircut value.
"""

from __future__ import annotations

import pytest

from pricebook.risk.collateral_optimisation import (
    CollateralAsset, CollateralOptimiser, CSARequirement,
)


class TestCoverageNetsHaircut:
    def test_5pct_haircut_requires_extra_gross(self):
        """CSA needing $100 covered by asset with 5% haircut → gross > $100."""
        # Only one asset available; LP must allocate $100/0.95 ≈ $105.26 gross.
        assets = [
            CollateralAsset("BOND", available_amount=200.0,
                            yield_rate=0.0, funding_rate=0.05,
                            haircut_pct=0.05),
        ]
        req = CSARequirement("csa1", "cp1", required_amount=100.0,
                              eligible_asset_ids=["BOND"])
        opt = CollateralOptimiser([req], assets)
        result = opt.optimise()
        assert result.solver_status == "optimal"
        # Gross allocation should be ≥ 100/(1 - 0.05) = 105.263...
        gross = sum(a.amount for a in result.allocations)
        assert gross == pytest.approx(100.0 / 0.95, rel=1e-6)
        # Post-haircut value covers requirement.
        post_haircut = 0.95 * gross
        assert post_haircut >= 100.0 - 1e-6

    def test_zero_haircut_unchanged(self):
        """Cash (zero haircut) → gross = required exactly."""
        assets = [
            CollateralAsset("CASH", available_amount=200.0,
                            yield_rate=0.0, funding_rate=0.05,
                            haircut_pct=0.0),
        ]
        req = CSARequirement("csa1", "cp1", required_amount=100.0,
                              eligible_asset_ids=["CASH"])
        opt = CollateralOptimiser([req], assets)
        result = opt.optimise()
        gross = sum(a.amount for a in result.allocations)
        assert gross == pytest.approx(100.0, rel=1e-6)


class TestUnmetWithHaircut:
    def test_insufficient_after_haircut(self):
        """If only $95 of an asset with 5% haircut is available, can't cover $100."""
        # Need $100, have $95 of asset with 5% haircut (= $90.25 post-haircut).
        assets = [
            CollateralAsset("BOND", available_amount=95.0,
                            yield_rate=0.0, funding_rate=0.05,
                            haircut_pct=0.05),
        ]
        req = CSARequirement("csa1", "cp1", required_amount=100.0,
                              eligible_asset_ids=["BOND"])
        opt = CollateralOptimiser([req], assets)
        result = opt.optimise()
        # LP is infeasible OR unmet flagged.
        assert "csa1" in result.unmet_requirements or result.solver_status == "infeasible"


class TestMultiCSAHaircutAware:
    def test_optimiser_prefers_low_haircut_when_costs_close(self):
        """Same cost-per-unit, but lower haircut → fewer units needed → optimiser prefers it."""
        # Two assets:
        # - LOW: haircut 0%, cost = 0.05.
        # - HIGH: haircut 0.20, cost = 0.05.
        # Both have same nominal cost_per_unit BUT to cover $100 you need
        # $100 of LOW vs $125 of HIGH → LOW is cheaper in absolute terms.
        # Note: cost_per_unit in the dataclass already includes haircut effect;
        # we override by constructing identical funding/yield/different haircut.
        assets = [
            CollateralAsset("LOW", available_amount=1000.0,
                            yield_rate=0.0, funding_rate=0.05,
                            haircut_pct=0.0),
            CollateralAsset("HIGH", available_amount=1000.0,
                            yield_rate=0.0, funding_rate=0.05,
                            haircut_pct=0.20),
        ]
        req = CSARequirement("csa1", "cp1", required_amount=100.0,
                              eligible_asset_ids=["LOW", "HIGH"])
        opt = CollateralOptimiser([req], assets)
        result = opt.optimise()
        # Prefer LOW.
        low_alloc = sum(a.amount for a in result.allocations if a.asset_id == "LOW")
        high_alloc = sum(a.amount for a in result.allocations if a.asset_id == "HIGH")
        assert low_alloc >= high_alloc
