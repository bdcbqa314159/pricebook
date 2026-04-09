"""Tests for full SA-CCR, BA-CVA, SA-CVA, CCP."""

import math
import pytest

from pricebook.regulatory.counterparty import (
    SA_CCR_SUPERVISORY_FACTORS, BA_CVA_RISK_WEIGHTS,
    calculate_maturity_factor, calculate_supervisory_duration,
    calculate_supervisory_delta, calculate_adjusted_notional,
    calculate_addon_single_trade, calculate_replacement_cost,
    calculate_pfe_multiplier, calculate_sa_ccr_ead,
    calculate_supervisory_discount, calculate_ba_cva, calculate_sa_cva,
    calculate_ccp_trade_exposure, calculate_ccp_default_fund,
)


# ---- Maturity factor ----

class TestMaturityFactor:
    def test_unmargined_1y(self):
        mf = calculate_maturity_factor(1.0, is_margined=False)
        assert mf == pytest.approx(1.0)

    def test_unmargined_short(self):
        mf = calculate_maturity_factor(0.25, is_margined=False)
        assert mf == pytest.approx(0.5)

    def test_margined_lower(self):
        mf_margined = calculate_maturity_factor(1.0, is_margined=True, mpor_days=10)
        # 1.5 × sqrt(10/250) = 1.5 × 0.2 = 0.3
        assert mf_margined == pytest.approx(0.3, rel=0.01)

    def test_supervisory_duration(self):
        sd = calculate_supervisory_duration(5)
        # (1 - exp(-0.25))/0.05 ≈ 4.42
        assert sd == pytest.approx((1 - math.exp(-0.25)) / 0.05)


# ---- Supervisory delta ----

class TestSupervisoryDelta:
    def test_long_non_option(self):
        assert calculate_supervisory_delta(is_long=True) == 1.0

    def test_short_non_option(self):
        assert calculate_supervisory_delta(is_long=False) == -1.0

    def test_long_call(self):
        assert calculate_supervisory_delta(is_long=True, is_option=True, option_type="call") == 0.5

    def test_long_put(self):
        assert calculate_supervisory_delta(is_long=True, is_option=True, option_type="put") == -0.5


# ---- Adjusted notional ----

class TestAdjustedNotional:
    def test_ir_uses_duration(self):
        adj = calculate_adjusted_notional(10_000_000, "IR", maturity=5.0)
        # SD(5) ≈ 4.42
        sd = calculate_supervisory_duration(5)
        assert adj == pytest.approx(10_000_000 * sd)

    def test_eq_uses_mf(self):
        adj = calculate_adjusted_notional(10_000_000, "EQ_SINGLE", maturity=2.0)
        # MF for unmargined 2Y → sqrt(min(2,1)) = 1
        assert adj == pytest.approx(10_000_000 * 1.0)


# ---- Single-trade add-on ----

class TestAddOn:
    def test_ir_addon(self):
        addon = calculate_addon_single_trade(
            10_000_000, "IR", maturity=5.0, delta=1.0,
        )
        # SF = 0.005, SD = 4.42 → addon ≈ 0.005 × 10M × 4.42
        expected = 0.005 * 10_000_000 * calculate_supervisory_duration(5)
        assert addon == pytest.approx(expected)

    def test_eq_higher_addon(self):
        ir = calculate_addon_single_trade(10_000_000, "IR", maturity=1.0)
        eq = calculate_addon_single_trade(10_000_000, "EQ_SINGLE", maturity=1.0)
        assert eq > ir


# ---- Replacement Cost ----

class TestReplacementCost:
    def test_unmargined_positive_mtm(self):
        rc = calculate_replacement_cost(mtm=1_000_000, is_margined=False)
        assert rc == 1_000_000

    def test_unmargined_negative_mtm(self):
        rc = calculate_replacement_cost(mtm=-500_000, is_margined=False)
        assert rc == 0

    def test_collateralised(self):
        rc = calculate_replacement_cost(mtm=1_000_000, collateral_held=800_000)
        assert rc == 200_000

    def test_margined_threshold(self):
        rc = calculate_replacement_cost(
            mtm=100_000, is_margined=True, threshold=500_000, mta=10_000,
        )
        # max(100K, 510K, 0) = 510K
        assert rc == 510_000


# ---- PFE multiplier ----

class TestPFEMultiplier:
    def test_positive_v_minus_c(self):
        m = calculate_pfe_multiplier(mtm=1_000_000, collateral=0, addon_aggregate=500_000)
        assert m == 1.0

    def test_negative_v_minus_c(self):
        m = calculate_pfe_multiplier(mtm=-500_000, collateral=0, addon_aggregate=500_000)
        assert m < 1.0
        assert m >= 0.05  # floor


# ---- SA-CCR EAD ----

class TestSACCREAD:
    def test_single_trade(self):
        trades = [{"notional": 10_000_000, "asset_class": "IR", "maturity": 5, "mtm": 0, "delta": 1}]
        r = calculate_sa_ccr_ead(trades)
        assert r["ead"] > 0
        assert r["replacement_cost"] == 0
        assert r["pfe"] > 0
        assert r["alpha"] == 1.4

    def test_collateral_reduces(self):
        trades = [{"notional": 10_000_000, "asset_class": "IR", "maturity": 5, "mtm": 1_000_000, "delta": 1}]
        r1 = calculate_sa_ccr_ead(trades, collateral_held=0)
        r2 = calculate_sa_ccr_ead(trades, collateral_held=500_000)
        assert r2["ead"] < r1["ead"]

    def test_multi_class_addons(self):
        trades = [
            {"notional": 10_000_000, "asset_class": "IR", "maturity": 5, "mtm": 0, "delta": 1},
            {"notional": 5_000_000, "asset_class": "EQ_SINGLE", "maturity": 1, "mtm": 0, "delta": 1},
        ]
        r = calculate_sa_ccr_ead(trades)
        assert "IR" in r["addons_by_class"]
        assert "EQ_SINGLE" in r["addons_by_class"]


# ---- BA-CVA ----

class TestBACVA:
    def test_single_counterparty(self):
        cps = [{"ead": 10_000_000, "rating": "BBB", "maturity": 5}]
        r = calculate_ba_cva(cps)
        assert r["k_cva"] > 0
        assert r["rwa"] == pytest.approx(r["k_cva"] * 12.5)

    def test_higher_rating_lower_charge(self):
        cps_aaa = [{"ead": 10_000_000, "rating": "AAA", "maturity": 5}]
        cps_bb = [{"ead": 10_000_000, "rating": "BB", "maturity": 5}]
        r_aaa = calculate_ba_cva(cps_aaa)
        r_bb = calculate_ba_cva(cps_bb)
        assert r_bb["k_cva"] > r_aaa["k_cva"]

    def test_supervisory_discount(self):
        df1 = calculate_supervisory_discount(1)
        df5 = calculate_supervisory_discount(5)
        # Longer maturity → smaller discount factor (DF decreases with M)
        assert df5 < df1
        assert 0 < df1 < 1
        assert df1 == pytest.approx((1 - math.exp(-0.05)) / 0.05)


# ---- SA-CVA ----

class TestSACVA:
    def test_basic(self):
        delta = {"IR": 1_000_000, "FX": 500_000}
        r = calculate_sa_cva(delta)
        assert r["k_sa_cva"] > 0
        assert r["multiplier"] == 1.25

    def test_with_vega(self):
        delta = {"IR": 1_000_000}
        vega = {"IR_vol": 100_000}
        r = calculate_sa_cva(delta, vega)
        assert r["k_vega"] > 0


# ---- CCP ----

class TestCCP:
    def test_qccp_2pct(self):
        r = calculate_ccp_trade_exposure(10_000_000, is_qualifying_ccp=True)
        assert r["risk_weight_pct"] == 2.0
        assert r["rwa"] == 200_000

    def test_non_qccp_100pct(self):
        r = calculate_ccp_trade_exposure(10_000_000, is_qualifying_ccp=False)
        assert r["risk_weight_pct"] == 100.0
        assert r["rwa"] == 10_000_000

    def test_default_fund_qccp(self):
        r = calculate_ccp_default_fund(
            df_contribution=1_000_000,
            k_ccp=10_000_000,
            total_df=100_000_000,
            ccp_capital=20_000_000,
            is_qualifying_ccp=True,
        )
        assert r["rwa"] >= 0
        assert r["df_share"] == 0.01

    def test_default_fund_non_qccp(self):
        r = calculate_ccp_default_fund(
            df_contribution=1_000_000, is_qualifying_ccp=False,
        )
        assert r["risk_weight_pct"] == 100
        assert r["rwa"] == 1_000_000
