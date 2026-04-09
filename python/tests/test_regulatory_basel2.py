"""Tests for Basel II / II.5 legacy framework."""

import pytest

from pricebook.regulatory.basel2 import (
    B2_SA_SOVEREIGN_RW, B2_SA_BANK_RW, B2_SA_CORPORATE_RW,
    b2_get_sovereign_rw, b2_get_bank_rw, b2_get_corporate_rw,
    b2_calculate_sa_rwa,
    b2_calculate_correlation, b2_calculate_irb_rwa,
    BusinessLine, BusinessLineIncome, AMAParameters,
    b2_calculate_bia_capital, b2_calculate_tsa_capital, b2_calculate_ama_capital,
    DerivativeType,
    b2_get_cem_addon_factor, b2_calculate_cem_ead, b2_calculate_cem_netting,
    b25_stressed_var, b25_market_risk_capital,
)


# ---- Basel 2 SA ----

class TestB2SA:
    def test_sovereign(self):
        assert b2_get_sovereign_rw("AAA") == 0
        assert b2_get_sovereign_rw("BBB") == 50
        assert b2_get_sovereign_rw("unrated") == 100

    def test_corporate_b2_higher_than_b3(self):
        """Basel II BBB corporate = 100% (vs Basel III's 75%)."""
        assert b2_get_corporate_rw("BBB") == 100

    def test_bank_option_2(self):
        assert b2_get_bank_rw("AAA") == 20
        assert b2_get_bank_rw("BBB") == 50
        assert b2_get_bank_rw("unrated") == 50

    def test_bank_option_1_one_notch_worse(self):
        assert b2_get_bank_rw("AAA", option=1) == 20  # sovereign 0 → 20
        assert b2_get_bank_rw("A", option=1) == 50    # sovereign 20 → 50

    def test_sa_rwa(self):
        r = b2_calculate_sa_rwa(10_000_000, "corporate", "BBB")
        assert r["risk_weight_pct"] == 100
        assert r["rwa"] == 10_000_000

    def test_residential_mortgage(self):
        r = b2_calculate_sa_rwa(10_000_000, "residential_mortgage")
        assert r["risk_weight_pct"] == 35


# ---- Basel 2 IRB ----

class TestB2IRB:
    def test_correlation_in_range(self):
        for pd in [0.001, 0.01, 0.05]:
            r = b2_calculate_correlation(pd, "corporate")
            assert 0.12 <= r <= 0.24

    def test_retail_mortgage_fixed(self):
        assert b2_calculate_correlation(0.01, "retail_mortgage") == 0.15

    def test_irb_rwa_positive(self):
        r = b2_calculate_irb_rwa(10_000_000, pd=0.01, lgd=0.45, maturity=5)
        assert r["rwa"] > 0
        assert r["approach"] == "B2-IRB"

    def test_irb_scales_with_pd(self):
        low = b2_calculate_irb_rwa(10_000_000, pd=0.001, lgd=0.45)
        high = b2_calculate_irb_rwa(10_000_000, pd=0.10, lgd=0.45)
        assert high["rwa"] > low["rwa"]


# ---- BIA ----

class TestBIA:
    def test_basic(self):
        r = b2_calculate_bia_capital(100_000_000, 110_000_000, 90_000_000)
        assert r["average_gross_income"] == 100_000_000
        assert r["capital_requirement"] == 15_000_000  # 15%

    def test_negative_year_excluded(self):
        r = b2_calculate_bia_capital(100_000_000, -50_000_000, 100_000_000)
        assert r["positive_years_count"] == 2
        assert r["average_gross_income"] == 100_000_000

    def test_all_negative(self):
        r = b2_calculate_bia_capital(-100, -200, -300)
        assert r["capital_requirement"] == 0


# ---- TSA ----

class TestTSA:
    def test_single_business_line(self):
        bli = BusinessLineIncome(
            business_line=BusinessLine.RETAIL_BANKING,
            gross_income_year1=100_000_000,
            gross_income_year2=100_000_000,
            gross_income_year3=100_000_000,
        )
        r = b2_calculate_tsa_capital([bli])
        # 12% beta × 100M = 12M per year, avg = 12M
        assert r["capital_requirement"] == pytest.approx(12_000_000)

    def test_multiple_lines(self):
        blis = [
            BusinessLineIncome(BusinessLine.CORPORATE_FINANCE, 50_000_000, 50_000_000, 50_000_000),
            BusinessLineIncome(BusinessLine.RETAIL_BANKING, 100_000_000, 100_000_000, 100_000_000),
        ]
        r = b2_calculate_tsa_capital(blis)
        # 18% × 50M + 12% × 100M = 9M + 12M = 21M per year
        assert r["capital_requirement"] == pytest.approx(21_000_000)


# ---- AMA ----

class TestAMA:
    def test_basic(self):
        params = AMAParameters(
            expected_loss=10_000_000,
            unexpected_loss_999=100_000_000,
        )
        r = b2_calculate_ama_capital(params)
        assert r["capital_requirement"] == 100_000_000

    def test_insurance_capped_at_20pct(self):
        params = AMAParameters(
            expected_loss=10_000_000,
            unexpected_loss_999=100_000_000,
            insurance_mitigation=50_000_000,  # would be 50% but capped at 20%
        )
        r = b2_calculate_ama_capital(params)
        # 100M × (1) - max(50M, 20% × 100M) = 100M - 20M = 80M
        assert r["capital_requirement"] == pytest.approx(80_000_000)

    def test_beicf_adjustment(self):
        params = AMAParameters(expected_loss=0, unexpected_loss_999=100_000_000)
        r = b2_calculate_ama_capital(params, business_environment_factor=1.2)
        assert r["capital_requirement"] == 120_000_000


# ---- CEM ----

class TestCEM:
    def test_ir_short_zero_addon(self):
        f = b2_get_cem_addon_factor(DerivativeType.INTEREST_RATE, 0.5)
        assert f == 0.0

    def test_eq_higher_addon(self):
        ir = b2_get_cem_addon_factor(DerivativeType.INTEREST_RATE, 5)
        eq = b2_get_cem_addon_factor(DerivativeType.EQUITY, 5)
        assert eq > ir

    def test_single_trade(self):
        r = b2_calculate_cem_ead(
            notional=10_000_000, mark_to_market=500_000,
            derivative_type=DerivativeType.INTEREST_RATE, maturity=5,
        )
        # CE = 500K, addon = 10M × 0.005 = 50K → EAD = 550K
        assert r["ead"] == pytest.approx(550_000)

    def test_negative_mtm_zero_ce(self):
        r = b2_calculate_cem_ead(
            notional=10_000_000, mark_to_market=-200_000,
            derivative_type=DerivativeType.INTEREST_RATE, maturity=5,
        )
        assert r["current_exposure"] == 0
        assert r["ead"] == 50_000  # only the addon

    def test_netting_reduces_ead(self):
        trades = [
            {"notional": 10_000_000, "mark_to_market": 1_000_000,
             "derivative_type": DerivativeType.INTEREST_RATE, "maturity": 5},
            {"notional": 10_000_000, "mark_to_market": -800_000,
             "derivative_type": DerivativeType.INTEREST_RATE, "maturity": 5},
        ]
        netted = b2_calculate_cem_netting(trades)
        # Single sum without netting: ~1M + ~50K = ~1.05M
        # Netted: net CE = 200K, smaller
        assert netted["ead"] < 1_050_000


# ---- Basel 2.5 ----

class TestB25:
    def test_stressed_var(self):
        r = b25_stressed_var(var_1day=1_000_000, multiplier=3.0)
        # 1M × sqrt(10) × 3 ≈ 9.49M
        import math
        assert r["capital_requirement"] == pytest.approx(3.0 * 1_000_000 * math.sqrt(10))

    def test_market_risk_total(self):
        r = b25_market_risk_capital(
            var_capital=10_000_000, stressed_var_capital=15_000_000,
            irc_capital=5_000_000, crm_capital=2_000_000,
            specific_risk_capital=3_000_000,
        )
        assert r["total_capital"] == 35_000_000
        assert r["total_rwa"] == 35_000_000 * 12.5
