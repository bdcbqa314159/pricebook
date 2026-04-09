"""Tests for stress testing + IRRBB."""

import pytest

from pricebook.regulatory.stress_irrbb import (
    ScenarioType, MacroVariable, StressScenario, PortfolioData,
    STANDARD_SCENARIOS,
    calculate_pd_stress_multiplier, calculate_lgd_stress_multiplier,
    stress_credit_portfolio, stress_market_portfolio,
    create_scenario_paths, run_integrated_stress_test,
    IRRBB_SHOCK_SCENARIOS,
    calculate_pv01, calculate_duration_gap, calculate_eve_impact,
    calculate_eve_all_scenarios, calculate_nii_sensitivity,
    calculate_irrbb_capital,
)


# ---- Stress multipliers ----

class TestStressMultipliers:
    def test_baseline_pd_unchanged(self):
        """At baseline values, multiplier = 1."""
        m = calculate_pd_stress_multiplier(0.02, 0.05, 0.03)
        assert m == pytest.approx(1.0)

    def test_recession_pd_higher(self):
        """Recession → PD multiplier > 1."""
        m = calculate_pd_stress_multiplier(-0.05, 0.12, -0.30)
        assert m > 1.0

    def test_pd_capped(self):
        """Multiplier capped at 10x."""
        m = calculate_pd_stress_multiplier(-0.50, 0.50, -0.90)
        assert m <= 10.0

    def test_lgd_multiplier_capped(self):
        m = calculate_lgd_stress_multiplier(-0.50, 0.20)
        assert m <= 2.0

    def test_lgd_baseline(self):
        m = calculate_lgd_stress_multiplier(0.03, 0.01)
        assert m == pytest.approx(1.0)


# ---- Stress credit portfolio ----

class TestStressCredit:
    def test_adverse_higher_losses(self):
        portfolio = PortfolioData(
            credit_exposure=1_000_000_000,
            credit_rwa=500_000_000,
            average_pd=0.02, average_lgd=0.45,
        )
        baseline = create_scenario_paths(ScenarioType.BASELINE)
        adverse = create_scenario_paths(ScenarioType.SEVERELY_ADVERSE)

        r_base = stress_credit_portfolio(portfolio, baseline, year=0)
        r_adv = stress_credit_portfolio(portfolio, adverse, year=0)
        assert r_adv["stressed_el"] > r_base["stressed_el"]

    def test_stressed_pd_capped_at_1(self):
        portfolio = PortfolioData(
            credit_exposure=1_000_000_000,
            credit_rwa=500_000_000,
            average_pd=0.20, average_lgd=0.45,  # high baseline PD
        )
        adverse = create_scenario_paths(ScenarioType.SEVERELY_ADVERSE)
        r = stress_credit_portfolio(portfolio, adverse, year=0)
        assert r["stressed_pd"] <= 1.0


# ---- Market stress ----

class TestStressMarket:
    def test_severe_higher_losses(self):
        portfolio = PortfolioData(
            credit_exposure=0, credit_rwa=0, average_pd=0, average_lgd=0,
            market_var=10_000_000, market_exposure=100_000_000,
        )
        baseline = create_scenario_paths(ScenarioType.BASELINE)
        severe = create_scenario_paths(ScenarioType.SEVERELY_ADVERSE)

        r_base = stress_market_portfolio(portfolio, baseline, year=0)
        r_sev = stress_market_portfolio(portfolio, severe, year=0)
        assert r_sev["stressed_var"] > r_base["stressed_var"]


# ---- Integrated stress test ----

class TestIntegratedStress:
    def test_three_year_run(self):
        portfolio = PortfolioData(
            credit_exposure=1_000_000_000, credit_rwa=500_000_000,
            average_pd=0.02, average_lgd=0.45,
            market_var=10_000_000, market_exposure=100_000_000,
        )
        scenario = create_scenario_paths(ScenarioType.ADVERSE, horizon_years=3)
        r = run_integrated_stress_test(portfolio, scenario)
        assert len(r["yearly"]) == 3
        assert r["cumulative_total_loss"] >= 0

    def test_severe_worse_than_baseline(self):
        portfolio = PortfolioData(
            credit_exposure=1_000_000_000, credit_rwa=500_000_000,
            average_pd=0.02, average_lgd=0.45,
            market_var=10_000_000, market_exposure=100_000_000,
        )
        baseline_run = run_integrated_stress_test(
            portfolio, create_scenario_paths(ScenarioType.BASELINE),
        )
        severe_run = run_integrated_stress_test(
            portfolio, create_scenario_paths(ScenarioType.SEVERELY_ADVERSE),
        )
        assert severe_run["cumulative_total_loss"] > baseline_run["cumulative_total_loss"]


# ---- IRRBB ----

class TestPV01:
    def test_basic(self):
        # 100M × 5Y duration × 0.0001 = 50,000
        assert calculate_pv01(100_000_000, 5) == 50_000


class TestDurationGap:
    def test_positive_gap(self):
        """Asset duration > liability duration → positive gap."""
        assets = [{"notional": 100, "duration": 5}]
        liabs = [{"notional": 80, "duration": 2}]
        gap = calculate_duration_gap(assets, liabs)
        assert gap["duration_gap"] > 0
        assert gap["equity"] == 20

    def test_pv01_difference(self):
        assets = [{"notional": 100_000_000, "duration": 5}]
        liabs = [{"notional": 80_000_000, "duration": 5}]
        gap = calculate_duration_gap(assets, liabs)
        assert gap["net_pv01"] == pytest.approx(10_000)  # (100M - 80M) × 5 × 0.0001


class TestEVE:
    def test_parallel_up_loses_money(self):
        """Asset-sensitive book: rates up → EVE down."""
        assets = [{"notional": 100, "duration": 5}]
        liabs = [{"notional": 80, "duration": 2}]
        gap = calculate_duration_gap(assets, liabs)
        impact = calculate_eve_impact(gap, rate_shock_bps=200)
        assert impact["eve_change"] < 0  # rates up → loss

    def test_all_scenarios(self):
        assets = [{"notional": 100, "duration": 5}]
        liabs = [{"notional": 80, "duration": 2}]
        r = calculate_eve_all_scenarios(assets, liabs, "USD")
        assert "parallel_up" in r["scenarios"]
        assert "parallel_down" in r["scenarios"]
        assert r["worst_scenario"] in ("parallel_up", "parallel_down")


class TestNII:
    def test_positive_gap_benefits_from_up_shock(self):
        assets = {"1y_or_less": 100_000_000}
        liabs = {"1y_or_less": 50_000_000}
        r = calculate_nii_sensitivity(assets, liabs, rate_shock_bps=200, horizon_years=1)
        assert r["total_nii_change"] > 0  # gap positive, rates up → income up


class TestIRRBBCapital:
    def test_outlier(self):
        eve_result = {"worst_eve_change": -200_000_000}
        r = calculate_irrbb_capital(eve_result, tier1_capital=1_000_000_000)
        # 200M loss vs 15% × 1B = 150M threshold → outlier
        assert r["is_outlier"]
        assert r["capital_charge"] == 50_000_000

    def test_within_limit(self):
        eve_result = {"worst_eve_change": -100_000_000}
        r = calculate_irrbb_capital(eve_result, tier1_capital=1_000_000_000)
        assert not r["is_outlier"]
        assert r["capital_charge"] == 0


# ---- Standard scenarios ----

class TestStandardScenarios:
    def test_severely_adverse_worst(self):
        baseline = STANDARD_SCENARIOS[ScenarioType.BASELINE]
        severe = STANDARD_SCENARIOS[ScenarioType.SEVERELY_ADVERSE]
        # GDP growth in severe should be lower
        assert severe[MacroVariable.GDP_GROWTH] < baseline[MacroVariable.GDP_GROWTH]
        assert severe[MacroVariable.UNEMPLOYMENT] > baseline[MacroVariable.UNEMPLOYMENT]


class TestIRRBBShocks:
    def test_parallel_shocks_defined(self):
        assert IRRBB_SHOCK_SCENARIOS["parallel_up"]["USD"] == 200
        assert IRRBB_SHOCK_SCENARIOS["parallel_down"]["USD"] == -200
