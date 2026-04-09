"""Tests for capital framework: output floor, leverage, G-SIB, TLAC."""

import pytest

from pricebook.regulatory.capital_framework import (
    OUTPUT_FLOOR_TRANSITION,
    calculate_output_floor, calculate_output_floor_by_risk_type,
    calculate_leverage_ratio,
    calculate_large_exposure,
    calculate_collateral_haircut, calculate_exposure_with_collateral,
    calculate_ead_off_balance_sheet,
    calculate_gsib_score, GSIB_BUCKETS,
    calculate_tlac_requirement, calculate_mrel_requirement,
)


# ---- Output Floor ----

class TestOutputFloor:
    def test_floor_binding(self):
        """Low IRB RWA → floor binding."""
        r = calculate_output_floor(rwa_irb=50_000_000, rwa_standardised=100_000_000, year=2028)
        assert r["floor_is_binding"]
        assert r["floored_rwa"] == 72_500_000  # 72.5% × 100M

    def test_floor_not_binding(self):
        """High IRB RWA → IRB binds."""
        r = calculate_output_floor(rwa_irb=80_000_000, rwa_standardised=100_000_000, year=2028)
        assert not r["floor_is_binding"]
        assert r["floored_rwa"] == 80_000_000

    def test_transition(self):
        """2023 floor = 50%."""
        r = calculate_output_floor(rwa_irb=20_000_000, rwa_standardised=100_000_000, year=2023)
        assert r["floor_percentage"] == 0.50
        assert r["floored_rwa"] == 50_000_000

    def test_full_implementation_2028(self):
        assert OUTPUT_FLOOR_TRANSITION[2028] == 0.725

    def test_by_risk_type(self):
        r = calculate_output_floor_by_risk_type(
            credit_risk_irb=50_000_000, credit_risk_sa=100_000_000,
            operational_risk=10_000_000,
        )
        assert "breakdown" in r
        assert r["floored_rwa"] >= 0


# ---- Leverage Ratio ----

class TestLeverageRatio:
    def test_compliant(self):
        r = calculate_leverage_ratio(
            tier1_capital=10_000_000_000, on_balance_sheet=200_000_000_000,
        )
        # 10B / 200B = 5%
        assert r["leverage_ratio_pct"] == pytest.approx(5.0)
        assert r["is_compliant"]

    def test_breach(self):
        r = calculate_leverage_ratio(
            tier1_capital=4_000_000_000, on_balance_sheet=200_000_000_000,
        )
        assert not r["is_compliant"]

    def test_gsib_higher_minimum(self):
        r_normal = calculate_leverage_ratio(
            tier1_capital=10_000_000_000, on_balance_sheet=200_000_000_000,
            is_gsib=False,
        )
        r_gsib = calculate_leverage_ratio(
            tier1_capital=10_000_000_000, on_balance_sheet=200_000_000_000,
            is_gsib=True, gsib_buffer_pct=0.02,
        )
        assert r_gsib["minimum_requirement"] > r_normal["minimum_requirement"]

    def test_total_exposure_sums(self):
        r = calculate_leverage_ratio(
            tier1_capital=10_000_000_000,
            on_balance_sheet=100_000_000_000,
            derivatives_exposure=50_000_000_000,
            sft_exposure=30_000_000_000,
            off_balance_sheet=20_000_000_000,
        )
        assert r["total_exposure"] == 200_000_000_000


# ---- Large Exposures ----

class TestLargeExposures:
    def test_within_limit(self):
        r = calculate_large_exposure(exposure_value=20_000_000, tier1_capital=200_000_000)
        assert not r["is_breach"]
        assert r["exposure_pct"] == 10.0

    def test_breach(self):
        r = calculate_large_exposure(exposure_value=60_000_000, tier1_capital=200_000_000)
        assert r["is_breach"]
        assert r["exposure_pct"] == 30.0
        assert r["excess_above_limit"] > 0

    def test_reportable(self):
        r = calculate_large_exposure(exposure_value=20_000_000, tier1_capital=200_000_000)
        assert r["is_reportable"]

    def test_gsib_to_gsib_lower_limit(self):
        r = calculate_large_exposure(
            exposure_value=35_000_000, tier1_capital=200_000_000,
            is_gsib_to_gsib=True,
        )
        # 17.5% > 15% G-SIB limit
        assert r["is_breach"]


# ---- CRM (collateral) ----

class TestCRM:
    def test_cash_zero_haircut(self):
        assert calculate_collateral_haircut("cash") == 0.0

    def test_equity_higher_haircut(self):
        assert calculate_collateral_haircut("equity_other") > calculate_collateral_haircut("cash")

    def test_exposure_with_cash(self):
        r = calculate_exposure_with_collateral(
            exposure=10_000_000, collateral_value=8_000_000, collateral_type="cash",
        )
        assert r["adjusted_exposure"] == pytest.approx(2_000_000)

    def test_exposure_with_equity(self):
        r = calculate_exposure_with_collateral(
            exposure=10_000_000, collateral_value=10_000_000, collateral_type="equity_other",
        )
        # 10M × (1 - 0.25) = 7.5M, so adj exposure = 2.5M
        assert r["adjusted_exposure"] == pytest.approx(2_500_000)

    def test_full_collateral(self):
        r = calculate_exposure_with_collateral(
            exposure=10_000_000, collateral_value=15_000_000, collateral_type="cash",
        )
        assert r["adjusted_exposure"] == 0


# ---- Off-balance sheet CCF ----

class TestOffBS:
    def test_short_term_commitment(self):
        r = calculate_ead_off_balance_sheet(10_000_000, "1y_or_less")
        assert r["ead"] == 2_000_000  # 20% CCF

    def test_long_term_commitment(self):
        r = calculate_ead_off_balance_sheet(10_000_000, "over_1y")
        assert r["ead"] == 5_000_000  # 50% CCF

    def test_guarantee(self):
        r = calculate_ead_off_balance_sheet(10_000_000, "guarantees_substitute")
        assert r["ead"] == 10_000_000  # 100% CCF


# ---- G-SIB ----

class TestGSIB:
    def test_no_gsib_low_score(self):
        r = calculate_gsib_score({})
        assert r["total_score"] == 0
        assert not r["is_gsib"]

    def test_bucket_1(self):
        # Construct data to hit bucket 1 (130-229)
        # Total exposures of 1.5T → score = 1.5T/100T × 10000 × 1.0 × 0.20 = 30
        # Need ~150 score → ~7.5T total exposures
        bank = {"total_exposures": 7_500_000_000_000}  # → score 150
        r = calculate_gsib_score(bank)
        assert r["total_score"] >= 130
        if r["total_score"] < 230:
            assert r["bucket"] == 1
            assert r["buffer_requirement"] == 0.010

    def test_buffer_increases_with_bucket(self):
        # Buckets 1-5 have increasing buffers
        buffers = [GSIB_BUCKETS[b]["buffer"] for b in [1, 2, 3, 4, 5]]
        assert buffers == sorted(buffers)


# ---- TLAC ----

class TestTLAC:
    def test_rwa_binding(self):
        """Low leverage exposure → RWA binds."""
        r = calculate_tlac_requirement(rwa=100_000_000_000, leverage_exposure=200_000_000_000)
        # 18% × 100B = 18B vs 6.75% × 200B = 13.5B → RWA binds
        assert r["binding_constraint"] == "RWA"
        assert r["tlac_requirement"] == 18_000_000_000

    def test_leverage_binding(self):
        """High leverage exposure → leverage binds."""
        r = calculate_tlac_requirement(rwa=100_000_000_000, leverage_exposure=400_000_000_000)
        assert r["binding_constraint"] == "Leverage"
        assert r["tlac_requirement"] == 27_000_000_000  # 6.75% × 400B

    def test_with_gsib_buffer(self):
        r = calculate_tlac_requirement(
            rwa=100_000_000_000, leverage_exposure=200_000_000_000, gsib_buffer=0.02,
        )
        assert r["rwa_requirement_pct"] == 20.0  # 18% + 2%


# ---- MREL ----

class TestMREL:
    def test_resolution_entity_floor(self):
        r = calculate_mrel_requirement(
            rwa=100_000_000_000, leverage_exposure=200_000_000_000,
            loss_absorption_amount=5_000_000_000,
            recapitalisation_amount=5_000_000_000,
        )
        # Loss + recap = 10B; 8% × 100B = 8B; 3% × 200B = 6B → max = 10B
        assert r["mrel_requirement"] == 10_000_000_000

    def test_floor_binds(self):
        r = calculate_mrel_requirement(
            rwa=100_000_000_000, leverage_exposure=200_000_000_000,
            loss_absorption_amount=2_000_000_000,
            recapitalisation_amount=2_000_000_000,
        )
        # Loss+recap = 4B; floor = max(8B, 6B) = 8B
        assert r["mrel_requirement"] == 8_000_000_000
