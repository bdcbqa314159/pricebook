"""Tests for liquidity (LCR/NSFR) and operational risk (SMA)."""

import math
import pytest

from pricebook.regulatory.liquidity_op import (
    HQLA_HAIRCUTS, calculate_hqla,
    calculate_cash_outflows, calculate_cash_inflows,
    calculate_lcr,
    ASF_FACTORS, RSF_FACTORS,
    calculate_asf, calculate_rsf, calculate_nsfr,
    BIC_BUCKETS,
    calculate_business_indicator, calculate_bic,
    calculate_loss_component, calculate_ilm, calculate_sma_capital,
)


# ---- HQLA ----

class TestHQLA:
    def test_l1_no_haircut(self):
        assets = [{"amount": 100_000_000, "asset_type": "L1_cash"}]
        r = calculate_hqla(assets)
        assert r["level1"] == 100_000_000
        assert r["total_hqla"] == 100_000_000

    def test_l2a_haircut(self):
        assets = [{"amount": 100_000_000, "asset_type": "L2A_corporate_bonds_AA"}]
        r = calculate_hqla(assets)
        # 15% haircut → but L2A capped at 0 with no L1
        assert r["level2a_gross"] == 85_000_000

    def test_l2_cap(self):
        """L2 capped at 40/60 of L1."""
        assets = [
            {"amount": 60_000_000, "asset_type": "L1_cash"},
            {"amount": 100_000_000, "asset_type": "L2A_sovereign_20pct"},
        ]
        r = calculate_hqla(assets)
        # Max L2 = 60M × 40/60 = 40M (after haircut: gross 85M, capped at 40M)
        assert r["level2a_adjusted"] == 40_000_000


# ---- Cash flows ----

class TestCashFlows:
    def test_outflows(self):
        liabs = [{"amount": 100_000_000, "liability_type": "retail_stable"}]
        r = calculate_cash_outflows(liabs)
        assert r["total_outflows"] == 3_000_000  # 3% × 100M

    def test_inflows(self):
        recv = [{"amount": 100_000_000, "receivable_type": "wholesale_non_financial"}]
        r = calculate_cash_inflows(recv)
        assert r["total_inflows_gross"] == 50_000_000  # 50%


# ---- LCR ----

class TestLCR:
    def test_compliant(self):
        hqla = [{"amount": 200_000_000, "asset_type": "L1_cash"}]
        liabs = [{"amount": 1_000_000_000, "liability_type": "retail_stable"}]  # 30M outflows
        recv = []
        r = calculate_lcr(hqla, liabs, recv)
        # LCR = 200M / 30M = 666% → compliant
        assert r["is_compliant"]
        assert r["lcr_pct"] > 100

    def test_inflow_cap(self):
        """Inflows capped at 75% of outflows."""
        hqla = [{"amount": 100_000_000, "asset_type": "L1_cash"}]
        liabs = [{"amount": 1_000_000_000, "liability_type": "retail_stable"}]  # 30M outflows
        recv = [{"amount": 1_000_000_000, "receivable_type": "wholesale_financial"}]  # 1B inflows
        r = calculate_lcr(hqla, liabs, recv)
        assert r["capped_inflows"] == r["gross_outflows"] * 0.75


# ---- ASF / RSF / NSFR ----

class TestASF:
    def test_tier1_full_factor(self):
        sources = [{"amount": 100_000_000, "funding_type": "tier1_capital"}]
        r = calculate_asf(sources)
        assert r["total_asf"] == 100_000_000

    def test_short_term_zero(self):
        sources = [{"amount": 100_000_000, "funding_type": "wholesale_non_operational_lt_6m"}]
        r = calculate_asf(sources)
        assert r["total_asf"] == 0


class TestRSF:
    def test_l1_zero(self):
        assets = [{"amount": 100_000_000, "asset_type": "L1_assets"}]
        r = calculate_rsf(assets)
        assert r["total_rsf"] == 0

    def test_long_loan_high_factor(self):
        assets = [{"amount": 100_000_000, "asset_type": "loans_to_non_financials_gt_1y_ge_35rw"}]
        r = calculate_rsf(assets)
        assert r["total_rsf"] == 85_000_000  # 85%

    def test_off_balance_sheet(self):
        assets = []
        r = calculate_rsf(assets, off_balance_sheet=100_000_000)
        assert r["off_balance_sheet_rsf"] == 5_000_000  # 5%


class TestNSFR:
    def test_compliant(self):
        funding = [{"amount": 200_000_000, "funding_type": "tier1_capital"}]
        assets = [{"amount": 100_000_000, "asset_type": "loans_to_non_financials_lt_1y"}]
        r = calculate_nsfr(funding, assets)
        # ASF = 200M, RSF = 50M → NSFR = 400% → compliant
        assert r["is_compliant"]

    def test_breach(self):
        funding = [{"amount": 50_000_000, "funding_type": "wholesale_non_operational_lt_6m"}]  # ASF = 0
        assets = [{"amount": 100_000_000, "asset_type": "loans_to_non_financials_gt_1y_ge_35rw"}]
        r = calculate_nsfr(funding, assets)
        assert not r["is_compliant"]


# ---- Business Indicator ----

class TestBI:
    def test_basic(self):
        r = calculate_business_indicator(
            interest_income=100_000_000, interest_expense=40_000_000,
            interest_earning_assets=2_000_000_000,
            fee_income=50_000_000, fee_expense=10_000_000,
            trading_book_pnl=20_000_000, banking_book_pnl=15_000_000,
        )
        # ILDC: net interest = 60M, cap = 45M → min = 45M
        assert r["ildc"] == 45_000_000
        # SC: max(50M, 10M) + max(0, 0) = 50M
        assert r["sc"] == 50_000_000
        # FC: |20M| + |15M| = 35M
        assert r["fc"] == 35_000_000
        # BI = 45M + 50M + 35M = 130M
        assert r["bi"] == 130_000_000


# ---- BIC ----

class TestBIC:
    def test_bucket1_only(self):
        bic = calculate_bic(500_000_000)
        # 500M × 12% = 60M
        assert bic == pytest.approx(60_000_000)

    def test_bucket2(self):
        bic = calculate_bic(2_000_000_000)
        # 1B × 12% + 1B × 15% = 270M
        assert bic == pytest.approx(270_000_000)

    def test_bucket3(self):
        bic = calculate_bic(50_000_000_000)
        # 1B × 12% + 29B × 15% + 20B × 18% = 120M + 4.35B + 3.6B = 8.07B
        assert bic == pytest.approx(120_000_000 + 29_000_000_000 * 0.15 + 20_000_000_000 * 0.18)

    def test_zero_bi(self):
        assert calculate_bic(0) == 0


# ---- ILM ----

class TestILM:
    def test_default_use_ilm_false(self):
        assert calculate_ilm(bic=100, lc=50, use_ilm=False) == 1.0

    def test_lc_equals_bic(self):
        """When LC = BIC, ratio = 1, ILM = ln(e) = 1."""
        ilm = calculate_ilm(bic=100, lc=100, use_ilm=True)
        assert ilm == pytest.approx(1.0)

    def test_lc_greater(self):
        """LC > BIC → ILM > 1."""
        ilm = calculate_ilm(bic=100, lc=200, use_ilm=True)
        assert ilm > 1.0


# ---- SMA capital ----

class TestSMA:
    def test_no_loss_history(self):
        r = calculate_sma_capital(bi=500_000_000, average_annual_loss=0, use_ilm=False)
        assert r["capital_requirement"] == 60_000_000  # BIC × ILM(=1)
        assert r["rwa"] == 60_000_000 * 12.5

    def test_with_loss_history(self):
        r = calculate_sma_capital(bi=2_000_000_000, average_annual_loss=10_000_000)
        assert r["capital_requirement"] > 0
        assert r["bic"] > 0
        assert r["ilm"] > 0
