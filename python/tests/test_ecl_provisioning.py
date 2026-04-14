"""Tests for ECL provisioning (IFRS 9 / CECL)."""

import math

import pytest

from pricebook.ecl_provisioning import (
    ECLResult,
    PortfolioECLResult,
    ScenarioECL,
    StageResult,
    cumulative_pds_from_hazard,
    ecl_12_month,
    ecl_lifetime,
    ecl_portfolio,
    marginal_pds_from_cumulative,
    stage_classification,
)


# ---- Stage classification ----

class TestStageClassification:
    def test_stage_1_performing(self):
        result = stage_classification(pd_origination=0.01, pd_current=0.012)
        assert result.stage == 1
        assert result.reason == "performing"

    def test_stage_2_sicr_relative(self):
        """PD doubled → stage 2."""
        result = stage_classification(pd_origination=0.01, pd_current=0.025)
        assert result.stage == 2

    def test_stage_2_sicr_absolute(self):
        """PD increased by > 50bp → stage 2."""
        result = stage_classification(pd_origination=0.001, pd_current=0.008)
        assert result.stage == 2

    def test_stage_2_days_past_due(self):
        """DPD > 30 → stage 2."""
        result = stage_classification(0.01, 0.01, days_past_due=35)
        assert result.stage == 2

    def test_stage_3_impaired(self):
        result = stage_classification(0.01, 0.05, is_impaired=True)
        assert result.stage == 3

    def test_stage_3_dpd_90(self):
        result = stage_classification(0.01, 0.01, days_past_due=95)
        assert result.stage == 3


# ---- ECL computation ----

class TestECL12Month:
    def test_basic(self):
        """ECL = PD × LGD × EAD."""
        result = ecl_12_month(pd_12m=0.02, lgd=0.45, ead=1_000_000)
        assert result.ecl == pytest.approx(9_000)
        assert result.stage == 1
        assert result.horizon == "12-month"

    def test_with_discount(self):
        result = ecl_12_month(0.02, 0.45, 1_000_000, discount_rate=0.05)
        assert result.ecl < 9_000  # discounting reduces ECL

    def test_zero_pd(self):
        result = ecl_12_month(0.0, 0.45, 1_000_000)
        assert result.ecl == 0.0


class TestECLLifetime:
    def test_lifetime_exceeds_12month(self):
        """Lifetime ECL ≥ 12-month ECL."""
        pd_12m = 0.02
        lgd = 0.45
        ead = 1_000_000
        ecl_1y = ecl_12_month(pd_12m, lgd, ead)

        cum_pds = cumulative_pds_from_hazard(0.02, 5)
        marg = marginal_pds_from_cumulative(cum_pds)
        ecl_lt = ecl_lifetime(marg, lgd, ead)

        assert ecl_lt.ecl >= ecl_1y.ecl

    def test_single_period(self):
        """1-period lifetime = 12-month."""
        result = ecl_lifetime([0.02], 0.45, 1_000_000)
        assert result.ecl == pytest.approx(9_000)

    def test_marginal_pds_sum(self):
        """Marginal PDs should sum to cumulative PD at maturity."""
        cum = cumulative_pds_from_hazard(0.03, 10)
        marg = marginal_pds_from_cumulative(cum)
        assert sum(marg) == pytest.approx(cum[-1])


# ---- Marginal / cumulative PDs ----

class TestPDConversions:
    def test_cumulative_increasing(self):
        cum = cumulative_pds_from_hazard(0.02, 5)
        for i in range(1, len(cum)):
            assert cum[i] > cum[i - 1]

    def test_cumulative_bounded(self):
        cum = cumulative_pds_from_hazard(0.02, 30)
        assert all(0 <= p <= 1 for p in cum)

    def test_marginal_non_negative(self):
        cum = cumulative_pds_from_hazard(0.05, 10)
        marg = marginal_pds_from_cumulative(cum)
        assert all(m >= 0 for m in marg)

    def test_round_trip(self):
        """Cumulative → marginal → sum = cumulative."""
        cum = cumulative_pds_from_hazard(0.03, 7)
        marg = marginal_pds_from_cumulative(cum)
        assert sum(marg) == pytest.approx(cum[-1], rel=1e-10)


# ---- Portfolio ECL ----

class TestPortfolioECL:
    def test_probability_weighted(self):
        """Portfolio ECL = Σ scenario_prob × scenario_ECL."""
        exposures = [
            {"ead": 1_000_000, "pd_base": 0.02, "stage": 1, "maturity_years": 5},
            {"ead": 2_000_000, "pd_base": 0.05, "stage": 2, "maturity_years": 3},
        ]
        scenarios = [
            ("base", 0.50, 1.0),
            ("stress", 0.30, 2.0),
            ("severe", 0.20, 3.0),
        ]
        result = ecl_portfolio(exposures, scenarios, lgd=0.45)

        # Probabilities sum to 1
        assert sum(s.probability for s in result.by_scenario) == pytest.approx(1.0)
        # Weighted ECL = sum of weighted
        assert result.total_ecl == pytest.approx(
            sum(s.weighted_ecl for s in result.by_scenario)
        )
        assert result.n_exposures == 2

    def test_stress_higher_ecl(self):
        """Stress scenario should produce higher ECL than base."""
        exposures = [{"ead": 1_000_000, "pd_base": 0.02, "stage": 1}]
        scenarios = [("base", 0.5, 1.0), ("stress", 0.5, 2.0)]
        result = ecl_portfolio(exposures, scenarios, lgd=0.45)
        base_ecl = result.by_scenario[0].ecl
        stress_ecl = result.by_scenario[1].ecl
        assert stress_ecl > base_ecl

    def test_single_scenario(self):
        exposures = [{"ead": 1_000_000, "pd_base": 0.02, "stage": 1}]
        scenarios = [("base", 1.0, 1.0)]
        result = ecl_portfolio(exposures, scenarios, lgd=0.45)
        assert result.total_ecl == pytest.approx(
            result.by_scenario[0].ecl
        )
