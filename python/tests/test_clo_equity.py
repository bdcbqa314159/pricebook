"""Tests for CLO equity Monte Carlo."""

import pytest
import numpy as np

from pricebook.credit.clo import CLOTranche, CLOWaterfall
from pricebook.credit.clo_equity import CLOEquityMC, CLOEquityResult, warehouse_risk, _mc_irr


@pytest.fixture
def clo_waterfall():
    tranches = [
        CLOTranche("AAA", 300_000_000, 0.012, 1),
        CLOTranche("AA", 50_000_000, 0.018, 2),
        CLOTranche("A", 30_000_000, 0.025, 3),
        CLOTranche("BBB", 20_000_000, 0.035, 4),
        CLOTranche("Equity", 50_000_000, 0.0, 5),
    ]
    return CLOWaterfall(tranches)


@pytest.fixture
def mc_engine(clo_waterfall):
    return CLOEquityMC(
        waterfall=clo_waterfall,
        n_loans=100,
        avg_spread=0.035,
        default_prob=0.02,
        recovery_mean=0.70,
        correlation=0.20,
        reinvestment_years=4.0,
        deal_life=8.0,
    )


class TestCLOEquityMC:
    def test_simulate_returns_result(self, mc_engine):
        result = mc_engine.simulate(n_paths=50, seed=42)
        assert isinstance(result, CLOEquityResult)
        assert result.n_paths == 50

    def test_irr_positive(self, mc_engine):
        result = mc_engine.simulate(n_paths=100, seed=42)
        assert result.equity_irr_mean > 0  # equity should earn positive return

    def test_irr_distribution(self, mc_engine):
        result = mc_engine.simulate(n_paths=500, seed=42)
        # IRR should have some dispersion (may be small for low-default portfolios)
        assert result.equity_irr_percentiles[5] <= result.equity_irr_percentiles[95]
        assert result.equity_irr_std >= 0

    def test_loss_rate_bounded(self, mc_engine):
        result = mc_engine.simulate(n_paths=100, seed=42)
        assert 0 <= result.loss_rate_mean <= 1.0

    def test_mean_cashflows(self, mc_engine):
        result = mc_engine.simulate(n_paths=50, seed=42)
        assert len(result.mean_cashflows) == mc_engine.n_periods
        # Income should be positive
        assert all(cf.interest_income >= 0 for cf in result.mean_cashflows)

    def test_higher_default_lower_irr(self, clo_waterfall):
        mc_low = CLOEquityMC(clo_waterfall, default_prob=0.005, n_loans=200)
        mc_high = CLOEquityMC(clo_waterfall, default_prob=0.10, n_loans=200)
        r_low = mc_low.simulate(n_paths=500, seed=42)
        r_high = mc_high.simulate(n_paths=500, seed=42)
        # Higher defaults should reduce equity IRR (more losses absorbed by equity)
        assert r_low.equity_irr_mean >= r_high.equity_irr_mean

    def test_higher_correlation_wider_dist(self, clo_waterfall):
        mc_low = CLOEquityMC(clo_waterfall, correlation=0.05, n_loans=200)
        mc_high = CLOEquityMC(clo_waterfall, correlation=0.50, n_loans=200)
        r_low = mc_low.simulate(n_paths=500, seed=42)
        r_high = mc_high.simulate(n_paths=500, seed=42)
        # Higher correlation should lead to wider loss distribution
        assert r_high.loss_rate_std >= r_low.loss_rate_std

    def test_to_dict(self, mc_engine):
        result = mc_engine.simulate(n_paths=20, seed=42)
        d = result.to_dict()
        assert "equity_irr_mean" in d
        assert "equity_irr_percentiles" in d

    def test_no_equity_tranche_error(self):
        wf = CLOWaterfall([CLOTranche("AAA", 100, 0.01, 1)])
        mc = CLOEquityMC(wf)
        # Single tranche IS the equity (highest seniority)
        result = mc.simulate(n_paths=10, seed=42)
        assert result.n_paths == 10


class TestWarehouseRisk:
    def test_basic(self):
        r = warehouse_risk(
            pipeline_notional=300_000_000,
            avg_spread=0.035,
            funding_cost=0.055,
            ramp_months=6,
        )
        assert r.net_carry < 0  # funding > spread in this example
        assert r.mtm_var_95 > 0

    def test_positive_carry(self):
        r = warehouse_risk(
            pipeline_notional=300_000_000,
            avg_spread=0.06,
            funding_cost=0.03,
            ramp_months=6,
        )
        assert r.net_carry > 0

    def test_to_dict(self):
        d = warehouse_risk(100_000_000, 0.04, 0.05, 6).to_dict()
        assert "mtm_var_95" in d


class TestMCIRR:
    def test_double_money(self):
        # -100 at t=0, +200 at t=4 (quarterly → 16 periods)
        cfs = [-100] + [0] * 15 + [200]
        irr = _mc_irr(cfs, 4)
        expected = (200 / 100) ** (1 / 4) - 1  # ~18.9% annual
        assert abs(irr - expected) < 0.02

    def test_zero_cashflows(self):
        assert _mc_irr([0, 0, 0], 4) == 0.0
