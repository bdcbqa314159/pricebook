"""Tests for collateral optimisation."""
import pytest
from pricebook.risk.collateral_optimisation import (
    CollateralAsset, CSARequirement, CollateralOptimiser,
    CollateralAllocationResult, stress_collateral,
)

@pytest.fixture
def assets():
    return [
        CollateralAsset("CASH", 50e6, yield_rate=0.0, funding_rate=0.05, haircut_pct=0.0),
        CollateralAsset("UST_10Y", 30e6, yield_rate=0.04, funding_rate=0.05, haircut_pct=0.02),
        CollateralAsset("CORP_IG", 20e6, yield_rate=0.055, funding_rate=0.05, haircut_pct=0.05),
    ]

@pytest.fixture
def requirements():
    return [
        CSARequirement("CSA_A", "BankA", 20e6, ["CASH", "UST_10Y", "CORP_IG"]),
        CSARequirement("CSA_B", "BankB", 15e6, ["CASH", "UST_10Y"]),
    ]

class TestOptimiser:
    def test_optimal(self, assets, requirements):
        opt = CollateralOptimiser(requirements, assets)
        r = opt.optimise()
        assert r.solver_status == "optimal"
        assert r.total_cost <= r.naive_cost + 1e-6
        assert r.savings >= -1e-6

    def test_allocations_cover(self, assets, requirements):
        r = CollateralOptimiser(requirements, assets).optimise()
        for req in requirements:
            allocated = sum(a.amount for a in r.allocations if a.csa_id == req.csa_id)
            assert allocated >= req.required_amount - 1e-6

    def test_infeasible(self):
        assets = [CollateralAsset("CASH", 5e6, 0, 0.05, 0)]
        reqs = [CSARequirement("CSA", "Bank", 100e6, ["CASH"])]
        r = CollateralOptimiser(reqs, assets).optimise()
        assert len(r.unmet_requirements) > 0

    def test_single_csa(self, assets):
        reqs = [CSARequirement("CSA_A", "BankA", 10e6, ["CASH", "UST_10Y"])]
        r = CollateralOptimiser(reqs, assets).optimise()
        assert r.solver_status == "optimal"

    def test_to_dict(self, assets, requirements):
        d = CollateralOptimiser(requirements, assets).optimise().to_dict()
        assert "total_cost" in d
        assert "savings" in d

class TestSubstitution:
    def test_cheaper(self, assets, requirements):
        opt = CollateralOptimiser(requirements, assets)
        impact = opt.what_if_substitution("CASH", "UST_10Y", "CSA_A", 5e6)
        assert isinstance(impact.savings, float)

class TestStress:
    def test_stress_increases_cost(self, assets, requirements):
        base = CollateralOptimiser(requirements, assets).optimise()
        stressed = stress_collateral(base, requirements, assets)
        assert len(stressed) >= 3
        # Higher haircuts should increase cost
        for s in stressed:
            assert s.total_cost_stressed >= base.total_cost - 1e-6
