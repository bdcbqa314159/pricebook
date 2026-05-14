"""Tests for loan portfolio stress testing."""

import pytest
import numpy as np

from pricebook.loan_stress import (
    MacroScenario, PREDEFINED_SCENARIOS,
    PortfolioStressResult, ConcentrationMetrics, MigrationResult,
    macro_stress_scenario, correlated_default_simulation,
    portfolio_loss_distribution, concentration_metrics, migration_matrix,
)


class TestMacroScenario:
    def test_predefined(self):
        sc = macro_stress_scenario("recession")
        assert sc.name == "recession"
        assert sc.pd_multiplier > 1.0

    def test_unknown(self):
        with pytest.raises(ValueError):
            macro_stress_scenario("unknown")

    def test_passthrough(self):
        sc = MacroScenario("custom", -0.01, 0.01, 0.01, 1.5, 0.05, 0.0)
        assert macro_stress_scenario(sc) is sc

    def test_to_dict(self):
        d = PREDEFINED_SCENARIOS["recession"].to_dict()
        assert "pd_multiplier" in d


class TestCorrelatedDefaults:
    def test_shape(self):
        defaults = correlated_default_simulation(100, [0.02]*100, 0.20, n_paths=1000)
        assert defaults.shape == (1000, 100)
        assert defaults.dtype == bool

    def test_default_rate_close_to_pd(self):
        defaults = correlated_default_simulation(500, [0.05]*500, 0.0, n_paths=5000, seed=42)
        avg_rate = float(np.mean(defaults))
        assert abs(avg_rate - 0.05) < 0.01

    def test_higher_correlation_wider_default_count(self):
        d_low = correlated_default_simulation(100, [0.05]*100, 0.05, n_paths=5000, seed=42)
        d_high = correlated_default_simulation(100, [0.05]*100, 0.50, n_paths=5000, seed=42)
        # Higher correlation → more variance in per-path default count
        counts_low = np.sum(d_low, axis=1)
        counts_high = np.sum(d_high, axis=1)
        assert np.std(counts_high) > np.std(counts_low)

    def test_zero_pd(self):
        defaults = correlated_default_simulation(10, [0.0]*10, 0.20, n_paths=100)
        assert np.sum(defaults) == 0

    def test_scalar_pd(self):
        defaults = correlated_default_simulation(50, 0.03, 0.20, n_paths=100)
        assert defaults.shape == (100, 50)


class TestPortfolioLoss:
    def test_basic(self):
        result = portfolio_loss_distribution(
            pds=[0.02]*20, notionals=[1_000_000]*20,
            recoveries=[0.70]*20, n_paths=5000, seed=42,
        )
        assert result.expected_loss > 0
        assert result.var_99 > result.expected_loss
        assert result.es_99 >= result.var_99

    def test_with_scenario(self):
        base = portfolio_loss_distribution(
            pds=[0.02]*20, notionals=[1e6]*20, recoveries=[0.70]*20,
            n_paths=5000, seed=42,
        )
        stressed = portfolio_loss_distribution(
            pds=[0.02]*20, notionals=[1e6]*20, recoveries=[0.70]*20,
            scenario="credit_crisis", n_paths=5000, seed=42,
        )
        assert stressed.expected_loss > base.expected_loss
        assert stressed.scenario_name == "credit_crisis"

    def test_by_industry(self):
        result = portfolio_loss_distribution(
            pds=[0.02]*4, notionals=[1e6]*4, recoveries=[0.70]*4,
            industries=["tech", "tech", "retail", "retail"],
            n_paths=5000, seed=42,
        )
        assert len(result.loss_by_industry) == 2
        assert "tech" in result.loss_by_industry
        assert "retail" in result.loss_by_industry

    def test_to_dict(self):
        r = portfolio_loss_distribution([0.02]*5, [1e6]*5, [0.7]*5, n_paths=100)
        d = r.to_dict()
        assert "var_99" in d
        assert "es_99" in d


class TestConcentration:
    def test_equal_weight(self):
        names = [f"N{i}" for i in range(10)]
        notionals = [100]*10
        m = concentration_metrics(names, notionals)
        assert abs(m.hhi - 0.10) < 1e-10
        assert abs(m.effective_n - 10.0) < 1e-10
        assert abs(m.max_single_name_pct - 0.10) < 1e-10

    def test_concentrated(self):
        names = ["A", "B", "C"]
        notionals = [900, 50, 50]
        m = concentration_metrics(names, notionals)
        assert m.hhi > 0.80
        assert m.max_single_name_pct == 0.90
        assert m.top_10_pct == 1.0

    def test_industry_hhi(self):
        names = ["A", "B", "C", "D"]
        notionals = [250, 250, 250, 250]
        industries = ["tech", "tech", "retail", "retail"]
        m = concentration_metrics(names, notionals, industries)
        assert abs(m.industry_hhi - 0.50) < 1e-10

    def test_to_dict(self):
        d = concentration_metrics(["A"], [100]).to_dict()
        assert "hhi" in d
        assert "effective_n" in d


class TestMigration:
    @pytest.fixture
    def simple_transition(self):
        return {
            "A": {"A": 0.90, "B": 0.08, "D": 0.02},
            "B": {"A": 0.05, "B": 0.85, "D": 0.10},
            "D": {"D": 1.0},
        }

    def test_basic(self, simple_transition):
        result = migration_matrix(
            initial_ratings=["A", "A", "B"],
            notionals=[100, 100, 100],
            transition_probs=simple_transition,
        )
        assert result.expected_default_pct > 0
        assert result.upgrade_pct >= 0
        assert result.downgrade_pct > 0

    def test_multi_year(self, simple_transition):
        r1 = migration_matrix(["A"]*10, [100]*10, simple_transition, horizon=1)
        r5 = migration_matrix(["A"]*10, [100]*10, simple_transition, horizon=5)
        assert r5.expected_default_pct > r1.expected_default_pct

    def test_to_dict(self, simple_transition):
        d = migration_matrix(["A"], [100], simple_transition).to_dict()
        assert "expected_default_pct" in d
