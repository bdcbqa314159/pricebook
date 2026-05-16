"""Tests for PE performance benchmarking."""

import math
import pytest

from pricebook.pe.pe_performance import (
    PMEResult, VintageCohort, PacingResult, GPEconomics,
    kaplan_schoar_pme, direct_alpha, long_nickels_pme,
    vintage_cohort, commitment_pacing, gp_economics,
    clawback_exposure, _pe_irr, _compound_return, _median,
)
from pricebook.credit.fund_participation import FundParticipation


# ═══════════════════════════════════════════════════════════════
# PME
# ═══════════════════════════════════════════════════════════════

class TestPME:
    def test_outperformance(self):
        """Fund that doubles money while index returns 5% pa → PME > 1."""
        result = kaplan_schoar_pme(
            contributions=[100, 0, 0, 0, 0],
            distributions=[0, 0, 0, 0, 250],
            index_returns=[0.05, 0.05, 0.05, 0.05, 0.05],
        )
        assert result.pme_ratio > 1.0
        assert result.direct_alpha > 0.0

    def test_underperformance(self):
        """Fund returns less than index → PME < 1."""
        result = kaplan_schoar_pme(
            contributions=[100, 0, 0, 0, 0],
            distributions=[0, 0, 0, 0, 110],
            index_returns=[0.10, 0.10, 0.10, 0.10, 0.10],
        )
        assert result.pme_ratio < 1.0
        assert result.direct_alpha < 0.0

    def test_matching_returns(self):
        """Fund matches index → PME ≈ 1.

        Note: KS-PME compares FV of distributions vs FV of contributions,
        both grown by the index from their respective dates. A single
        contribution at t=0 and single distribution at t=N with matching
        return gives PME = (1+r) because the distribution at t=N is not
        grown further but the contribution at t=0 is grown for N periods.
        For a true PME=1, distribute at the very end with no further index growth.
        """
        # Contribution at t=0 grows by index for 5 periods.
        # Distribution at t=4 grows by index for 1 period.
        # For PME=1, set fund_nav at the end instead of distribution.
        terminal = 100 * 1.08 ** 5
        result = kaplan_schoar_pme(
            contributions=[100, 0, 0, 0, 0],
            distributions=[0, 0, 0, 0, 0],
            index_returns=[0.08, 0.08, 0.08, 0.08, 0.08],
            fund_nav=terminal,
        )
        assert abs(result.pme_ratio - 1.0) < 0.05

    def test_empty_cashflows(self):
        result = kaplan_schoar_pme([], [], [])
        assert result.pme_ratio == 1.0

    def test_with_nav(self):
        """Unrealised NAV contributes to PME."""
        result = kaplan_schoar_pme(
            contributions=[100],
            distributions=[0],
            index_returns=[0.05],
            fund_nav=150,
        )
        assert result.pme_ratio > 1.0

    def test_to_dict(self):
        r = kaplan_schoar_pme([100], [120], [0.05])
        d = r.to_dict()
        assert "pme_ratio" in d
        assert "direct_alpha" in d


class TestDirectAlpha:
    def test_positive(self):
        assert direct_alpha(0.15, 0.08) == pytest.approx(0.07)

    def test_negative(self):
        assert direct_alpha(0.05, 0.10) == pytest.approx(-0.05)


class TestLongNickels:
    def test_outperformance(self):
        ln = long_nickels_pme(
            contributions=[100, 0, 0, 0, 0],
            distributions=[0, 0, 0, 0, 0],
            index_returns=[0.05, 0.05, 0.05, 0.05, 0.05],
            fund_nav=200,
        )
        # Fund NAV 200 vs index portfolio of 100 growing at 5% = ~127.6
        assert ln > 1.0


# ═══════════════════════════════════════════════════════════════
# Vintage Cohort
# ═══════════════════════════════════════════════════════════════

class TestVintageCohort:
    def test_single_vintage(self):
        funds = [
            FundParticipation(100_000, vintage_year=2020, gross_return=0.10),
            FundParticipation(100_000, vintage_year=2020, gross_return=0.15),
            FundParticipation(100_000, vintage_year=2020, gross_return=0.08),
            FundParticipation(100_000, vintage_year=2020, gross_return=0.12),
        ]
        cohorts = vintage_cohort(funds)
        assert len(cohorts) == 1
        assert cohorts[0].vintage_year == 2020
        assert cohorts[0].n_funds == 4
        assert cohorts[0].upper_quartile_irr >= cohorts[0].median_irr
        assert cohorts[0].lower_quartile_irr <= cohorts[0].median_irr

    def test_multiple_vintages(self):
        funds = [
            FundParticipation(100_000, vintage_year=2018, gross_return=0.10),
            FundParticipation(100_000, vintage_year=2019, gross_return=0.12),
            FundParticipation(100_000, vintage_year=2020, gross_return=0.08),
        ]
        cohorts = vintage_cohort(funds)
        assert len(cohorts) == 3
        years = [c.vintage_year for c in cohorts]
        assert years == [2018, 2019, 2020]

    def test_to_dict(self):
        funds = [FundParticipation(100_000, vintage_year=2020)]
        cohorts = vintage_cohort(funds)
        d = cohorts[0].to_dict()
        assert "vintage_year" in d
        assert "median_irr" in d


# ═══════════════════════════════════════════════════════════════
# Commitment Pacing
# ═══════════════════════════════════════════════════════════════

class TestCommitmentPacing:
    def test_pacing_length(self):
        results = commitment_pacing(
            target_allocation=0.15,
            portfolio_value=1_000_000_000,
            horizon=10,
        )
        assert len(results) == 10

    def test_nav_grows(self):
        results = commitment_pacing(
            target_allocation=0.15,
            portfolio_value=1_000_000_000,
            existing_nav=50_000_000,
            horizon=10,
        )
        assert results[-1].expected_nav > results[0].expected_nav

    def test_net_cashflow_sign(self):
        """Early years should have negative net CF (more calls than distributions)."""
        results = commitment_pacing(
            target_allocation=0.15,
            portfolio_value=1_000_000_000,
            existing_nav=0,
            horizon=10,
        )
        # First year: calls > distributions (building allocation)
        assert results[0].net_cashflow < 0

    def test_to_dict(self):
        results = commitment_pacing(0.10, 1_000_000, horizon=3)
        d = results[0].to_dict()
        assert "expected_nav" in d


# ═══════════════════════════════════════════════════════════════
# GP Economics
# ═══════════════════════════════════════════════════════════════

class TestGPEconomics:
    def test_basic_economics(self):
        gpe = gp_economics(
            fund_size=1_000_000_000,
            fund_life=10,
            mgmt_fee_rate=0.015,
            carry_rate=0.20,
            hurdle_rate=0.08,
            gross_return=0.15,
        )
        assert gpe.mgmt_fee_total > 0
        assert gpe.carry_total > 0
        assert gpe.gp_commitment > 0
        assert gpe.total_gp_revenue > 0

    def test_mgmt_fee_npv(self):
        gpe = gp_economics(fund_size=1_000_000_000)
        # NPV should be less than total (discounted)
        assert gpe.mgmt_fee_npv < gpe.mgmt_fee_total

    def test_carry_above_hurdle(self):
        """With 15% gross and 8% hurdle, carry should be positive."""
        gpe = gp_economics(fund_size=1_000_000_000, gross_return=0.15, hurdle_rate=0.08)
        assert gpe.carry_total > 0

    def test_carry_below_hurdle(self):
        """With 5% gross and 8% hurdle, no carry."""
        gpe = gp_economics(fund_size=1_000_000_000, gross_return=0.05, hurdle_rate=0.08)
        assert gpe.carry_total == 0.0

    def test_gp_commitment(self):
        gpe = gp_economics(fund_size=1_000_000_000, gp_commitment_pct=0.02)
        assert gpe.gp_commitment == 20_000_000

    def test_to_dict(self):
        d = gp_economics(fund_size=100_000_000).to_dict()
        assert "carry_npv" in d
        assert "clawback_exposure" in d


# ═══════════════════════════════════════════════════════════════
# Clawback
# ═══════════════════════════════════════════════════════════════

class TestClawback:
    def test_no_clawback(self):
        assert clawback_exposure(100, 100) == 0.0
        assert clawback_exposure(80, 100) == 0.0

    def test_clawback_triggered(self):
        assert clawback_exposure(120, 100) == 20.0

    def test_zero_exposure(self):
        assert clawback_exposure(0, 0) == 0.0


# ═══════════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════════

class TestHelpers:
    def test_irr_known(self):
        # -100 now, +150 in 3 years → IRR = (150/100)^(1/3) - 1 ≈ 14.47%
        irr = _pe_irr([-100, 0, 0, 150])
        expected = (150 / 100) ** (1 / 3) - 1
        assert abs(irr - expected) < 0.01

    def test_compound_return(self):
        ret = _compound_return([0.10, 0.10, 0.10])
        assert abs(ret - 0.10) < 1e-10

    def test_compound_return_varying(self):
        ret = _compound_return([0.20, -0.10, 0.15])
        product = 1.20 * 0.90 * 1.15
        expected = product ** (1 / 3) - 1
        assert abs(ret - expected) < 1e-10

    def test_median_odd(self):
        assert _median([1, 2, 3]) == 2

    def test_median_even(self):
        assert _median([1, 2, 3, 4]) == 2.5

    def test_median_empty(self):
        assert _median([]) == 0.0
