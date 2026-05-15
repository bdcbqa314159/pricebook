"""Tests for LBO deal model."""

import math
import pytest

from pricebook.pe.lbo import (
    LBOModel, SourcesAndUses, FCFProjection, DebtYear, ExitAnalysis,
    LBOResult, _irr_simple,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def base_model():
    """Standard mid-market LBO: $500M EV, $100M EBITDA, 5.0x entry."""
    return LBOModel(
        enterprise_value=500_000_000,
        entry_ebitda=100_000_000,
        equity_pct=0.40,
        senior_debt_turns=4.0,
        mezz_debt_turns=1.0,
        ebitda_growth=0.05,
        ebitda_margin=0.20,
        tax_rate=0.25,
        capex_pct_revenue=0.03,
        da_pct_revenue=0.02,
        senior_rate=0.06,
        senior_amort_pct=0.01,
        mezz_cash_rate=0.06,
        mezz_pik_rate=0.04,
        sweep_pct=0.50,
        hold_period=5,
    )


# ═══════════════════════════════════════════════════════════════
# Sources & Uses
# ═══════════════════════════════════════════════════════════════

class TestSourcesAndUses:
    def test_balance(self, base_model):
        su = base_model.sources_and_uses()
        assert abs(su.check_balance()) < 1.0, f"Sources != Uses, diff={su.check_balance()}"

    def test_senior_debt_turns(self, base_model):
        su = base_model.sources_and_uses()
        assert su.senior_debt == 400_000_000  # 4.0x × 100M

    def test_mezzanine_turns(self, base_model):
        su = base_model.sources_and_uses()
        assert su.mezzanine == 100_000_000  # 1.0x × 100M

    def test_equity_positive(self, base_model):
        su = base_model.sources_and_uses()
        assert su.equity > 0

    def test_transaction_fees(self, base_model):
        su = base_model.sources_and_uses()
        assert su.transaction_fees == 500_000_000 * 0.02  # 2% of EV

    def test_to_dict(self, base_model):
        d = base_model.sources_and_uses().to_dict()
        assert "equity" in d
        assert "total_sources" in d
        assert "balance" in d


# ═══════════════════════════════════════════════════════════════
# EBITDA & FCF Projection
# ═══════════════════════════════════════════════════════════════

class TestProjection:
    def test_ebitda_growth(self, base_model):
        ebitda = base_model.project_ebitda()
        assert len(ebitda) == 6  # year 0 through 5
        assert ebitda[0] == 100_000_000
        assert abs(ebitda[1] - 105_000_000) < 1.0  # 5% growth
        assert ebitda[5] > ebitda[4]  # monotonically increasing

    def test_ebitda_custom_growth(self):
        model = LBOModel(
            enterprise_value=500_000_000,
            entry_ebitda=100_000_000,
            ebitda_growth=[0.10, 0.08, 0.05, 0.03, 0.02],
            hold_period=5,
        )
        ebitda = model.project_ebitda()
        assert abs(ebitda[1] - 110_000_000) < 1.0

    def test_fcf_length(self, base_model):
        fcfs = base_model.project_fcf()
        assert len(fcfs) == 5

    def test_fcf_positive(self, base_model):
        fcfs = base_model.project_fcf()
        for f in fcfs:
            assert f.fcf > 0, f"Year {f.year} FCF is negative: {f.fcf}"

    def test_fcf_ebitda_to_fcf(self, base_model):
        fcfs = base_model.project_fcf()
        f = fcfs[0]
        # FCF = EBITDA - taxes - capex - NWC change
        expected = f.ebitda - f.taxes - f.capex - f.nwc_change
        assert abs(f.fcf - expected) < 1.0

    def test_revenue_from_margin(self, base_model):
        fcfs = base_model.project_fcf()
        f = fcfs[0]
        expected_revenue = f.ebitda / 0.20
        assert abs(f.revenue - expected_revenue) < 1.0


# ═══════════════════════════════════════════════════════════════
# Debt Schedule
# ═══════════════════════════════════════════════════════════════

class TestDebtSchedule:
    def test_schedule_length(self, base_model):
        ds = base_model.debt_schedule()
        assert len(ds) == 5

    def test_senior_amortises(self, base_model):
        ds = base_model.debt_schedule()
        assert ds[-1].senior_closing < ds[0].senior_opening

    def test_senior_interest(self, base_model):
        ds = base_model.debt_schedule()
        d = ds[0]
        expected = d.senior_opening * 0.06
        assert abs(d.senior_interest - expected) < 1.0

    def test_mezz_pik_accretes(self, base_model):
        ds = base_model.debt_schedule()
        assert ds[-1].mezz_closing > ds[0].mezz_opening  # PIK adds to principal

    def test_leverage_declining(self, base_model):
        ds = base_model.debt_schedule()
        # Leverage should generally decline as EBITDA grows and debt amortises
        assert ds[-1].net_leverage < ds[0].net_leverage

    def test_sweep_reduces_senior(self, base_model):
        ds = base_model.debt_schedule()
        total_sweep = sum(d.senior_sweep for d in ds)
        assert total_sweep > 0  # sweep should be active

    def test_no_negative_debt(self, base_model):
        ds = base_model.debt_schedule()
        for d in ds:
            assert d.senior_closing >= 0
            assert d.mezz_closing >= 0


# ═══════════════════════════════════════════════════════════════
# Exit Analysis
# ═══════════════════════════════════════════════════════════════

class TestExitAnalysis:
    def test_exit_at_entry_multiple(self, base_model):
        ea = base_model.exit_analysis(5.0)  # same as entry
        assert ea.moic > 1.0  # should be >1 due to debt paydown

    def test_higher_multiple_higher_irr(self, base_model):
        ea_low = base_model.exit_analysis(4.0)
        ea_high = base_model.exit_analysis(6.0)
        assert ea_high.equity_irr > ea_low.equity_irr

    def test_moic_monotonic_in_multiple(self, base_model):
        moics = [base_model.exit_analysis(m).moic for m in [4.0, 5.0, 6.0, 7.0]]
        for i in range(len(moics) - 1):
            assert moics[i + 1] >= moics[i]

    def test_exit_year_variation(self, base_model):
        ea3 = base_model.exit_analysis(5.0, exit_year=3)
        ea5 = base_model.exit_analysis(5.0, exit_year=5)
        # Later exit with same multiple: more debt paydown → higher MOIC
        assert ea5.moic >= ea3.moic

    def test_equity_value_bridge(self, base_model):
        ea = base_model.exit_analysis(5.0)
        assert abs(ea.equity_value - (ea.enterprise_value - ea.net_debt)) < 1.0

    def test_invalid_exit_year(self, base_model):
        with pytest.raises(ValueError):
            base_model.exit_analysis(5.0, exit_year=0)
        with pytest.raises(ValueError):
            base_model.exit_analysis(5.0, exit_year=10)


# ═══════════════════════════════════════════════════════════════
# Full Run
# ═══════════════════════════════════════════════════════════════

class TestRun:
    def test_run_default(self, base_model):
        result = base_model.run()
        assert isinstance(result, LBOResult)
        assert len(result.exit_analyses) > 0
        assert len(result.fcf_projections) == 5
        assert len(result.debt_schedule) == 5

    def test_run_custom_multiples(self, base_model):
        result = base_model.run(exit_multiples=[4.0, 5.0, 6.0])
        assert len(result.exit_analyses) == 3

    def test_result_to_dict(self, base_model):
        d = base_model.run().to_dict()
        assert "sources_and_uses" in d
        assert "exit_analyses" in d
        assert "entry_multiple" in d


# ═══════════════════════════════════════════════════════════════
# Sensitivity Table
# ═══════════════════════════════════════════════════════════════

class TestSensitivity:
    def test_sensitivity_shape(self, base_model):
        grid = base_model.sensitivity_table(
            row_param="exit_multiple",
            col_param="hold_period",
            row_values=[4.0, 5.0, 6.0],
            col_values=[3.0, 4.0, 5.0],
        )
        assert len(grid) == 3
        assert len(grid[0]) == 3

    def test_sensitivity_irr_increases_with_multiple(self, base_model):
        grid = base_model.sensitivity_table(
            row_param="exit_multiple",
            col_param="hold_period",
            row_values=[4.0, 5.0, 6.0],
            col_values=[5.0],
        )
        irrs = [row[0] for row in grid]
        assert irrs[2] > irrs[0]  # 6x exit > 4x exit

    def test_sensitivity_growth_param(self, base_model):
        grid = base_model.sensitivity_table(
            row_param="ebitda_growth",
            col_param="exit_multiple",
            row_values=[0.0, 0.05, 0.10],
            col_values=[5.0],
        )
        irrs = [row[0] for row in grid]
        assert irrs[2] > irrs[0]  # 10% growth > 0% growth


# ═══════════════════════════════════════════════════════════════
# IRR Helper
# ═══════════════════════════════════════════════════════════════

class TestIRR:
    def test_double_money_5_years(self):
        irr = _irr_simple(100, 200, 5)
        expected = 2 ** (1 / 5) - 1  # ~14.87%
        assert abs(irr - expected) < 1e-10

    def test_triple_money_3_years(self):
        irr = _irr_simple(100, 300, 3)
        expected = 3 ** (1 / 3) - 1  # ~44.22%
        assert abs(irr - expected) < 1e-10

    def test_zero_investment(self):
        assert _irr_simple(0, 100, 5) == 0.0

    def test_zero_terminal(self):
        assert _irr_simple(100, 0, 5) == 0.0


# ═══════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_no_mezzanine(self):
        model = LBOModel(
            enterprise_value=500_000_000,
            entry_ebitda=100_000_000,
            mezz_debt_turns=0.0,
        )
        su = model.sources_and_uses()
        assert su.mezzanine == 0.0
        ds = model.debt_schedule()
        for d in ds:
            assert d.mezz_cash_interest == 0.0
            assert d.mezz_pik_interest == 0.0

    def test_no_sweep(self):
        model = LBOModel(
            enterprise_value=500_000_000,
            entry_ebitda=100_000_000,
            sweep_pct=0.0,
        )
        ds = model.debt_schedule()
        for d in ds:
            assert d.senior_sweep == 0.0

    def test_bullet_loan(self):
        model = LBOModel(
            enterprise_value=500_000_000,
            entry_ebitda=100_000_000,
            senior_amort_pct=0.0,
            sweep_pct=0.0,
        )
        ds = model.debt_schedule()
        for d in ds:
            assert d.senior_amort == 0.0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            LBOModel(enterprise_value=-1, entry_ebitda=100)
        with pytest.raises(ValueError):
            LBOModel(enterprise_value=100, entry_ebitda=-1)

    def test_rollover_equity(self):
        model = LBOModel(
            enterprise_value=500_000_000,
            entry_ebitda=100_000_000,
            rollover_equity=50_000_000,
        )
        su = model.sources_and_uses()
        assert su.rollover_equity == 50_000_000
        assert abs(su.check_balance()) < 1.0
