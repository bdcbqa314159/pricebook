"""Tests for FRTB Internal Models Approach."""

import pytest

from pricebook.regulatory.market_risk_ima import (
    ESRiskFactor, DRCPosition, DeskPLA, FRTBIMAConfig,
    get_liquidity_horizon,
    calculate_liquidity_adjusted_es, calculate_stressed_es,
    calculate_nmrf_charge, simulate_drc_portfolio, calculate_ima_drc,
    evaluate_backtesting, evaluate_pla,
    calculate_imcc, calculate_frtb_ima_capital,
    quick_frtb_ima, compare_ima_vs_sa,
    LIQUIDITY_HORIZONS, LH_STEPS, PLUS_FACTOR_TABLE,
)


# ---- Liquidity horizons ----

class TestLiquidityHorizons:
    def test_ir_major_10day(self):
        assert get_liquidity_horizon("IR", "major") == 10

    def test_eq_large_cap_40day(self):
        assert get_liquidity_horizon("EQ", "large_cap") == 40

    def test_unknown_120(self):
        assert get_liquidity_horizon("UNKNOWN", "thing") == 120


# ---- Liquidity-adjusted ES ----

class TestLiquidityAdjustedES:
    def test_single_factor(self):
        rf = [ESRiskFactor("IR", "major", es_10day=1_000_000)]
        r = calculate_liquidity_adjusted_es(rf)
        assert r["es_total"] > 0

    def test_empty(self):
        assert calculate_liquidity_adjusted_es([])["es_total"] == 0.0

    def test_longer_horizon_more_es(self):
        ir = [ESRiskFactor("IR", "major", es_10day=1_000_000)]  # 10-day
        cr = [ESRiskFactor("CR", "other", es_10day=1_000_000)]  # 120-day
        r_ir = calculate_liquidity_adjusted_es(ir)
        r_cr = calculate_liquidity_adjusted_es(cr)
        assert r_cr["es_total"] > r_ir["es_total"]

    def test_modellable_only(self):
        rf = [
            ESRiskFactor("IR", "major", es_10day=1_000_000, is_modellable=True),
            ESRiskFactor("EQ", "other", es_10day=2_000_000, is_modellable=False),
        ]
        r = calculate_liquidity_adjusted_es(rf)
        assert r["num_factors"] == 1


# ---- Stressed ES ----

class TestStressedES:
    def test_basic(self):
        rf = [ESRiskFactor("IR", "major", es_10day=1_000_000, stressed_es_10day=2_000_000)]
        r = calculate_stressed_es(rf, 1_000_000, 1_000_000)
        assert r["ses_total"] > 0

    def test_no_stressed_factors(self):
        rf = [ESRiskFactor("IR", "major", es_10day=1_000_000)]
        r = calculate_stressed_es(rf, 1_000_000, 1_000_000)
        assert r["ses_total"] == 0.0


# ---- NMRF ----

class TestNMRF:
    def test_zero_with_no_nmrf(self):
        rf = [ESRiskFactor("IR", "major", es_10day=1_000_000, is_modellable=True)]
        assert calculate_nmrf_charge(rf)["nmrf_total"] == 0.0

    def test_nmrf_charge(self):
        rf = [
            ESRiskFactor("EQ", "other", es_10day=500_000, is_modellable=False),
            ESRiskFactor("CR", "HY", es_10day=300_000, is_modellable=False),
        ]
        r = calculate_nmrf_charge(rf)
        assert r["nmrf_total"] > 0
        assert r["num_factors"] == 2


# ---- DRC simulation ----

class TestDRC:
    def test_no_positions(self):
        r = calculate_ima_drc([])
        assert r["drc_charge"] == 0.0

    def test_single_position(self):
        pos = [DRCPosition("p1", "Corp_A", 10_000_000, 10_000_000, pd=0.05, lgd=0.45)]
        r = calculate_ima_drc(pos, FRTBIMAConfig(drc_num_simulations=5_000))
        assert r["drc_charge"] >= 0

    def test_higher_pd_higher_drc(self):
        config = FRTBIMAConfig(drc_num_simulations=10_000)
        low = calculate_ima_drc(
            [DRCPosition("p1", "Corp_A", 10_000_000, 10_000_000, pd=0.001)],
            config,
        )
        high = calculate_ima_drc(
            [DRCPosition("p1", "Corp_B", 10_000_000, 10_000_000, pd=0.20)],
            config,
        )
        assert high["drc_charge"] >= low["drc_charge"]


# ---- Backtesting ----

class TestBacktest:
    def test_green(self):
        r = evaluate_backtesting(3)
        assert r["zone"] == "green"
        assert r["plus_factor"] == 0.0

    def test_yellow(self):
        r = evaluate_backtesting(7)
        assert r["zone"] == "yellow"
        assert r["plus_factor"] == 0.65

    def test_red(self):
        r = evaluate_backtesting(15)
        assert r["zone"] == "red"
        assert r["plus_factor"] == 1.0


# ---- PLA ----

class TestPLA:
    def test_green_desk(self):
        desks = [DeskPLA("desk1", spearman_correlation=0.85, kl_divergence=0.05)]
        r = evaluate_pla(desks)
        assert r["desks"][0]["overall_zone"] == "green"

    def test_red_desk(self):
        desks = [DeskPLA("desk1", spearman_correlation=0.5, kl_divergence=0.20)]
        r = evaluate_pla(desks)
        assert r["desks"][0]["overall_zone"] == "red"
        assert not r["desks"][0]["ima_eligible"]

    def test_summary(self):
        desks = [
            DeskPLA("d1", 0.85, 0.05),  # green
            DeskPLA("d2", 0.75, 0.10),  # amber
            DeskPLA("d3", 0.5, 0.15),   # red
        ]
        r = evaluate_pla(desks)
        assert r["summary"]["green"] == 1
        assert r["summary"]["amber"] == 1
        assert r["summary"]["red"] == 1


# ---- IMCC ----

class TestIMCC:
    def test_basic(self):
        rf = [ESRiskFactor("IR", "major", es_10day=1_000_000)]
        r = calculate_imcc(rf, es_current=1_000_000, ses=2_000_000)
        assert r["imcc"] > 0
        assert r["es_component"] > 0
        assert r["ses_component"] > 0

    def test_plus_factor_increases(self):
        rf = [ESRiskFactor("IR", "major", es_10day=1_000_000)]
        config_low = FRTBIMAConfig(plus_factor=0.0)
        config_high = FRTBIMAConfig(plus_factor=0.5)
        r_low = calculate_imcc(rf, 1_000_000, 2_000_000, config=config_low)
        r_high = calculate_imcc(rf, 1_000_000, 2_000_000, config=config_high)
        assert r_high["imcc"] >= r_low["imcc"]


# ---- Total IMA capital ----

class TestFRTBIMA:
    def test_total(self):
        rf = [ESRiskFactor("IR", "major", es_10day=1_000_000, stressed_es_10day=2_000_000)]
        drc = [DRCPosition("p1", "Corp_A", 10_000_000, 10_000_000, pd=0.02)]
        config = FRTBIMAConfig(drc_num_simulations=5_000)
        r = calculate_frtb_ima_capital(rf, drc, config)
        assert r["total_capital"] > 0
        assert r["total_rwa"] == pytest.approx(r["total_capital"] * 12.5)

    def test_quick_ima(self):
        r = quick_frtb_ima(
            es_10day_total=5_000_000,
            stressed_es_10day_total=9_000_000,
            drc_positions=[
                {"obligor": "Corp_A", "notional": 10_000_000, "rating": "BBB"},
            ],
            plus_factor=0.0,
        )
        assert r["total_capital"] > 0

    def test_compare_ima_vs_sa(self):
        rf = [ESRiskFactor("IR", "major", es_10day=1_000_000, stressed_es_10day=2_000_000)]
        drc = [DRCPosition("p1", "Corp_A", 10_000_000, 10_000_000, pd=0.02)]
        delta_sa = {"GIRR": [{"bucket": "USD", "sensitivity": 1_000_000, "risk_weight": 11}]}
        config = FRTBIMAConfig(drc_num_simulations=2_000)
        r = compare_ima_vs_sa(rf, drc, delta_sa, config=config)
        assert "ima_capital" in r
        assert "sa_capital" in r
