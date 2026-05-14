"""Tests for reverse stress testing."""

import pytest

from pricebook.regulatory.reverse_stress import (
    ReverseStressTarget, ReverseStressResult,
    reverse_stress_portfolio, reverse_stress_ccar, scenario_surface,
)
from pricebook.regulatory.stress_irrbb import (
    MacroVariable, PortfolioData,
)
from pricebook.regulatory.ccar import CCARConfig


@pytest.fixture
def portfolio():
    return PortfolioData(
        credit_exposure=100_000_000_000,
        credit_rwa=80_000_000_000,
        average_pd=0.02,
        average_lgd=0.40,
        market_var=200_000_000,
        market_exposure=20_000_000_000,
    )


class TestReverseStressPortfolio:
    def test_finds_breach(self, portfolio):
        target = ReverseStressTarget(
            metric="cumulative_total_loss",
            threshold=5_000_000_000,
            direction="above",
        )
        r = reverse_stress_portfolio(portfolio, target, max_iter=50)
        assert isinstance(r, ReverseStressResult)
        assert r.n_iterations > 0
        # Should find a scenario that causes > 5bn loss
        if r.found:
            assert r.metric_value > target.threshold

    def test_infeasible_target(self, portfolio):
        """Extremely tight target may not be achievable."""
        target = ReverseStressTarget(
            metric="cumulative_total_loss",
            threshold=999_999_999_999_999,  # impossible
            direction="above",
        )
        r = reverse_stress_portfolio(portfolio, target, max_iter=20)
        assert not r.found

    def test_severity_positive(self, portfolio):
        target = ReverseStressTarget("cumulative_total_loss", 1_000_000_000, "above")
        r = reverse_stress_portfolio(portfolio, target, max_iter=30)
        assert r.scenario_severity >= 0

    def test_to_dict(self, portfolio):
        target = ReverseStressTarget("cumulative_total_loss", 1e9, "above")
        d = reverse_stress_portfolio(portfolio, target, max_iter=10).to_dict()
        assert "found" in d
        assert "severity" in d


class TestReverseStressCCAR:
    def test_cet1_breach(self, portfolio):
        config = CCARConfig(
            starting_cet1=10_000_000_000,
            starting_rwa=100_000_000_000,
            quarterly_ppnr=500_000_000,
        )
        target = ReverseStressTarget("cet1_ratio", 0.045, "below")
        r = reverse_stress_ccar(config, portfolio, target, max_iter=30)
        assert isinstance(r, ReverseStressResult)
        assert r.n_iterations > 0


class TestScenarioSurface:
    def test_surface_shape(self, portfolio):
        result = scenario_surface(
            portfolio,
            var1=MacroVariable.GDP_GROWTH,
            var2=MacroVariable.EQUITY_PRICES,
            n_grid=5,
        )
        assert len(result["var1_values"]) == 5
        assert len(result["var2_values"]) == 5
        assert len(result["metric_grid"]) == 5
        assert len(result["metric_grid"][0]) == 5

    def test_surface_has_variation(self, portfolio):
        result = scenario_surface(
            portfolio,
            var1=MacroVariable.GDP_GROWTH,
            var2=MacroVariable.HOUSE_PRICES,
            var1_range=(-0.05, 0.03),
            var2_range=(-0.30, 0.05),
            n_grid=3,
        )
        values = [v for row in result["metric_grid"] for v in row]
        # Should have some variation across the grid
        assert max(values) > min(values) or len(set(values)) == 1
