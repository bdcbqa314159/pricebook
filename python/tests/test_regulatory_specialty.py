"""Tests for specialty regulatory: crypto, step-in, FX, simplified SA, Pillar 3."""

import pytest

from pricebook.regulatory.specialty import (
    classify_crypto_asset,
    calculate_group1a_rwa, calculate_group1b_rwa, calculate_group2_rwa,
    check_group2_exposure_limit,
    EntityType, StepInIndicator,
    UnconsolidatedEntity, StepInAssessment,
    assess_step_in_indicators, calculate_step_in_capital_impact,
    INDICATOR_WEIGHTS,
    FXRates, get_default_fx_rates,
    simplified_sa_market_risk,
    generate_km1_template, generate_ov1_template,
)


# ---- Crypto classification ----

class TestCryptoClassification:
    def test_tokenised_traditional(self):
        assert classify_crypto_asset(is_tokenised_traditional=True) == "1a"

    def test_stablecoin_passes_all(self):
        assert classify_crypto_asset(
            has_effective_stabilisation=True,
            passes_redemption_test=True,
            passes_reserve_test=True,
        ) == "1b"

    def test_stablecoin_fails_one(self):
        assert classify_crypto_asset(
            has_effective_stabilisation=True,
            passes_redemption_test=False,
            passes_reserve_test=True,
        ) == "2b"

    def test_infrastructure_risk(self):
        assert classify_crypto_asset(
            is_tokenised_traditional=True,
            infrastructure_risk_acceptable=False,
        ) == "2b"


class TestCryptoRWA:
    def test_group1a(self):
        r = calculate_group1a_rwa(10_000_000, underlying_rw=20)
        # 10M × 0.2 + 10M × 0.025 = 2M + 250K
        assert r["rwa"] == pytest.approx(2_250_000)

    def test_group1b(self):
        r = calculate_group1b_rwa(10_000_000, weighted_reserve_rw=0)
        # 10M × 0 + 10M × 0.025 (redemption) + 10M × 0.025 (infra)
        assert r["total_rwa"] == pytest.approx(500_000)

    def test_group2a_netting(self):
        r = calculate_group2_rwa(long_exposure=10_000_000, short_exposure=8_000_000, is_group_2a=True)
        assert r["net_exposure"] == 2_000_000
        assert r["rwa"] == 2_000_000  # 100% on net

    def test_group2b_no_netting(self):
        r = calculate_group2_rwa(long_exposure=10_000_000, short_exposure=8_000_000, is_group_2a=False)
        assert r["gross_exposure"] == 18_000_000
        assert r["risk_weight_pct"] == 1250
        assert r["rwa"] == 18_000_000 * 12.5

    def test_exposure_limit_breach(self):
        r = check_group2_exposure_limit(
            group2a_exposure=20_000_000, group2b_exposure=15_000_000,
            tier1_capital=1_000_000_000,
        )
        # Total 35M / 1B = 3.5% > 2% limit
        assert r["group2_limit_breached"]
        # 2b: 15M / 1B = 1.5% > 1% limit
        assert r["group2b_limit_breached"]

    def test_within_limits(self):
        r = check_group2_exposure_limit(
            group2a_exposure=10_000_000, group2b_exposure=5_000_000,
            tier1_capital=1_000_000_000,
        )
        assert not r["group2_limit_breached"]
        assert not r["group2b_limit_breached"]


# ---- Step-in risk ----

class TestStepIn:
    def test_low_risk(self):
        entity = UnconsolidatedEntity(
            entity_id="E1", entity_name="MMF1",
            entity_type=EntityType.MONEY_MARKET_FUND,
            total_assets=100_000_000,
            is_sponsored=False, uses_bank_name=False,
            past_support_provided=False,
        )
        r = assess_step_in_indicators(entity)
        assert r.risk_level == "low"
        assert r.overall_score < 0.3

    def test_high_risk_sponsored(self):
        entity = UnconsolidatedEntity(
            entity_id="E1", entity_name="Conduit1",
            entity_type=EntityType.CONDUIT,
            total_assets=500_000_000,
            is_sponsored=True, uses_bank_name=True,
            past_support_provided=True,
        )
        r = assess_step_in_indicators(
            entity,
            has_implicit_support_expectation=True,
            involvement_level=1.0,
            reputational_impact=1.0,
            investor_expectation_level=1.0,
            provides_credit_enhancement=True,
            provides_liquidity_support=True,
            revenue_from_entity_significant=True,
        )
        assert r.risk_level == "high"
        assert r.overall_score >= 0.6

    def test_capital_impact_high(self):
        entity = UnconsolidatedEntity(
            entity_id="E1", entity_name="X",
            entity_type=EntityType.CONDUIT,
            total_assets=1_000_000_000, is_sponsored=True, uses_bank_name=True,
            past_support_provided=True,
        )
        assessment = assess_step_in_indicators(
            entity, has_implicit_support_expectation=True,
            involvement_level=1.0, reputational_impact=1.0,
            investor_expectation_level=1.0,
            provides_credit_enhancement=True, provides_liquidity_support=True,
            revenue_from_entity_significant=True,
        )
        r = calculate_step_in_capital_impact(assessment, entity_rwa_if_consolidated=500_000_000)
        # High risk → 100% factor
        assert r["implied_rwa"] == 500_000_000

    def test_capital_impact_low(self):
        entity = UnconsolidatedEntity(
            entity_id="E1", entity_name="X",
            entity_type=EntityType.INVESTMENT_FUND,
            total_assets=100_000_000,
        )
        assessment = assess_step_in_indicators(entity)
        r = calculate_step_in_capital_impact(assessment, entity_rwa_if_consolidated=50_000_000)
        # Low risk → 0% factor
        assert r["implied_rwa"] == 0


# ---- FX ----

class TestFX:
    def test_set_get(self):
        fx = FXRates()
        fx.set_spot("EURUSD", 1.08)
        assert fx.get_spot("EURUSD") == 1.08

    def test_convert_direct(self):
        fx = FXRates()
        fx.set_spot("EURUSD", 1.08)
        assert fx.convert(1_000_000, "EUR", "USD") == pytest.approx(1_080_000)

    def test_convert_inverse(self):
        fx = FXRates()
        fx.set_spot("EURUSD", 1.08)
        # 1.08M USD → ~1M EUR
        assert fx.convert(1_080_000, "USD", "EUR") == pytest.approx(1_000_000)

    def test_same_currency(self):
        fx = FXRates()
        assert fx.convert(1_000_000, "USD", "USD") == 1_000_000

    def test_triangulation_via_usd(self):
        fx = FXRates()
        fx.set_spot("EURUSD", 1.08)
        fx.set_spot("USDJPY", 150.0)
        # 1M EUR → 1.08M USD → 162M JPY
        assert fx.convert(1_000_000, "EUR", "JPY") == pytest.approx(162_000_000)

    def test_no_rate_raises(self):
        fx = FXRates()
        with pytest.raises(ValueError):
            fx.convert(100, "ZAR", "BRL")

    def test_default_rates(self):
        fx = get_default_fx_rates()
        assert fx.get_spot("EURUSD") is not None
        assert fx.convert(100, "EUR", "USD") > 100  # 1 EUR > 1 USD


# ---- Simplified SA ----

class TestSimplifiedSA:
    def test_basic(self):
        r = simplified_sa_market_risk(
            girr_sensitivity=10_000_000,
            fx_net_position=5_000_000,
        )
        # 10M × 2% + 5M × 10% = 200K + 500K = 700K
        assert r["total_capital"] == pytest.approx(700_000)

    def test_zero(self):
        r = simplified_sa_market_risk()
        assert r["total_capital"] == 0


# ---- Pillar 3 ----

class TestPillar3:
    def test_km1(self):
        r = generate_km1_template(
            cet1=10_000_000_000, tier1=12_000_000_000,
            total_capital=15_000_000_000, rwa=100_000_000_000,
            leverage_exposure=300_000_000_000,
            lcr_hqla=50_000_000_000, lcr_net_outflows=40_000_000_000,
            nsfr_asf=200_000_000_000, nsfr_rsf=180_000_000_000,
        )
        assert r["capital_ratios"]["cet1_ratio_pct"] == 10.0
        assert r["lcr_pct"] == 125.0
        assert r["nsfr_pct"] > 100

    def test_ov1(self):
        r = generate_ov1_template(
            credit_rwa=80_000_000_000, market_rwa=15_000_000_000,
            operational_rwa=10_000_000_000, cva_rwa=2_000_000_000,
        )
        assert r["total_rwa"] == 107_000_000_000
        assert r["capital_requirement"] == pytest.approx(107_000_000_000 * 0.08)
