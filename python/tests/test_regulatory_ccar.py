"""Tests for CCAR/DFAST stress capital projection."""

import pytest

from pricebook.regulatory.ccar import (
    CCARConfig, QuarterResult, CCARResult,
    project_capital_trajectory, run_ccar_suite, ccar_summary,
)
from pricebook.regulatory.stress_irrbb import (
    ScenarioType, PortfolioData, create_scenario_paths,
)


@pytest.fixture
def config():
    return CCARConfig(
        n_quarters=9,
        starting_cet1=50_000_000_000,   # $50bn CET1
        starting_rwa=400_000_000_000,   # $400bn RWA → 12.5% CET1
        quarterly_ppnr=5_000_000_000,   # $5bn/quarter
        quarterly_dividend=1_000_000_000,
        quarterly_buyback=500_000_000,
    )


@pytest.fixture
def portfolio():
    return PortfolioData(
        credit_exposure=300_000_000_000,
        credit_rwa=250_000_000_000,
        average_pd=0.02,
        average_lgd=0.40,
        market_var=500_000_000,
        market_exposure=50_000_000_000,
    )


class TestProjection:
    def test_baseline_passes(self, config, portfolio):
        scenario = create_scenario_paths(ScenarioType.BASELINE, 3)
        r = project_capital_trajectory(config, scenario, portfolio)
        assert r.passes_minimum
        assert r.minimum_cet1_ratio > config.cet1_minimum
        assert len(r.quarterly_results) == 9

    def test_adverse_lower_ratio(self, config, portfolio):
        baseline = create_scenario_paths(ScenarioType.BASELINE, 3)
        adverse = create_scenario_paths(ScenarioType.ADVERSE, 3)
        r_base = project_capital_trajectory(config, baseline, portfolio)
        r_adv = project_capital_trajectory(config, adverse, portfolio)
        assert r_adv.minimum_cet1_ratio < r_base.minimum_cet1_ratio

    def test_severely_adverse_lowest(self, config, portfolio):
        adverse = create_scenario_paths(ScenarioType.ADVERSE, 3)
        severe = create_scenario_paths(ScenarioType.SEVERELY_ADVERSE, 3)
        r_adv = project_capital_trajectory(config, adverse, portfolio)
        r_sev = project_capital_trajectory(config, severe, portfolio)
        assert r_sev.minimum_cet1_ratio <= r_adv.minimum_cet1_ratio

    def test_buyback_suspended_under_stress(self, config, portfolio):
        scenario = create_scenario_paths(ScenarioType.ADVERSE, 3)
        r = project_capital_trajectory(config, scenario, portfolio)
        for q in r.quarterly_results:
            assert q.buybacks == 0  # suspended under stress

    def test_dividends_continue(self, config, portfolio):
        scenario = create_scenario_paths(ScenarioType.ADVERSE, 3)
        r = project_capital_trajectory(config, scenario, portfolio)
        for q in r.quarterly_results:
            assert q.dividends == config.quarterly_dividend

    def test_quarter_arithmetic(self, config, portfolio):
        scenario = create_scenario_paths(ScenarioType.BASELINE, 3)
        r = project_capital_trajectory(config, scenario, portfolio)
        q = r.quarterly_results[0]
        expected_change = q.net_income - q.dividends - q.buybacks
        assert abs(q.capital_change - expected_change) < 1.0

    def test_to_dict(self, config, portfolio):
        scenario = create_scenario_paths(ScenarioType.BASELINE, 3)
        d = project_capital_trajectory(config, scenario, portfolio).to_dict()
        assert "minimum_cet1_ratio" in d
        assert "quarterly_results" in d
        assert "passes_minimum" in d


class TestSuite:
    def test_three_scenarios(self, config, portfolio):
        results = run_ccar_suite(config, portfolio)
        assert len(results) == 3
        assert "baseline" in results
        assert "adverse" in results
        assert "severely_adverse" in results

    def test_ordering(self, config, portfolio):
        results = run_ccar_suite(config, portfolio)
        # Baseline should be best, severely_adverse worst
        assert results["baseline"].minimum_cet1_ratio >= results["adverse"].minimum_cet1_ratio
        assert results["adverse"].minimum_cet1_ratio >= results["severely_adverse"].minimum_cet1_ratio


class TestSummary:
    def test_summary(self, config, portfolio):
        results = run_ccar_suite(config, portfolio)
        s = ccar_summary(results)
        assert "worst_scenario" in s
        assert "all_pass" in s
        assert isinstance(s["all_pass"], bool)


class TestEdgeCases:
    def test_undercapitalised_bank(self, portfolio):
        """Bank with low starting CET1 should fail."""
        config = CCARConfig(
            starting_cet1=10_000_000_000,   # only $10bn
            starting_rwa=400_000_000_000,   # $400bn → 2.5% CET1 (below 4.5%)
            quarterly_ppnr=1_000_000_000,
        )
        scenario = create_scenario_paths(ScenarioType.SEVERELY_ADVERSE, 3)
        r = project_capital_trajectory(config, scenario, portfolio)
        assert not r.passes_minimum

    def test_zero_rwa(self):
        """Zero RWA should not crash."""
        config = CCARConfig(starting_cet1=100, starting_rwa=0)
        portfolio = PortfolioData(credit_exposure=0, credit_rwa=0, average_pd=0.01, average_lgd=0.4)
        scenario = create_scenario_paths(ScenarioType.BASELINE, 3)
        r = project_capital_trajectory(config, scenario, portfolio)
        assert len(r.quarterly_results) == 9
