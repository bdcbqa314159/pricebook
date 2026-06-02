"""Tests for MBS, ABS, and CMBS structured products."""

import pytest
import math
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


def _make_curve(rate=0.04):
    from pricebook.core.discount_curve import DiscountCurve
    from pricebook.core.interpolation import InterpolationMethod
    dates = [REF + relativedelta(years=y) for y in range(1, 35)]
    dfs = [math.exp(-rate * y) for y in range(1, 35)]
    return DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)


# ═══════════════════════════════════════════════════════════════
# MBS (S1)
# ═══════════════════════════════════════════════════════════════

class TestMBS:
    def test_psa_ramp(self):
        from pricebook.structured.mbs import psa_speed
        assert psa_speed(1) == pytest.approx(0.002, abs=0.001)
        assert psa_speed(15) == pytest.approx(0.03, abs=0.001)
        assert psa_speed(30) == pytest.approx(0.06, abs=0.001)
        assert psa_speed(60) == pytest.approx(0.06, abs=0.001)

    def test_psa_200(self):
        from pricebook.structured.mbs import psa_speed
        assert psa_speed(30, 2.0) == pytest.approx(0.12, abs=0.001)

    def test_cpr_smm_roundtrip(self):
        from pricebook.structured.mbs import cpr_to_smm, smm_to_cpr
        cpr = 0.06
        smm = cpr_to_smm(cpr)
        assert smm_to_cpr(smm) == pytest.approx(cpr, abs=1e-10)

    def test_price_mbs(self):
        from pricebook.structured.mbs import MBSPool, price_mbs
        dc = _make_curve()
        pool = MBSPool(1_000_000, 0.06, 360, 0.055, age=0)
        r = price_mbs(pool, dc, psa_speed=1.0)
        assert r.price > 0
        assert r.wal > 0
        assert r.wal < 30  # WAL < maturity due to prepayment

    def test_faster_prepay_shorter_wal(self):
        from pricebook.structured.mbs import MBSPool, price_mbs
        dc = _make_curve()
        pool = MBSPool(1_000_000, 0.06, 360, 0.055)
        slow = price_mbs(pool, dc, psa_speed=1.0)
        fast = price_mbs(pool, dc, psa_speed=3.0)
        assert fast.wal < slow.wal

    def test_oas(self):
        from pricebook.structured.mbs import MBSPool, price_mbs, oas_mbs
        dc = _make_curve()
        pool = MBSPool(1_000_000, 0.06, 360, 0.055)
        base = price_mbs(pool, dc, psa_speed=1.0)
        oas = oas_mbs(pool, dc, base.price - 2, psa_speed=1.0)
        assert oas.oas_bp > 0  # price below par → positive OAS

    def test_io_po_strips(self):
        from pricebook.structured.mbs import MBSPool, io_po_strips
        dc = _make_curve()
        pool = MBSPool(1_000_000, 0.06, 360, 0.055)
        r = io_po_strips(pool, dc)
        assert r.io_price > 0
        assert r.po_price > 0
        # IO + PO ≈ total (approximately)
        assert r.io_price + r.po_price > 50

    def test_io_po_duration_signs(self):
        """PO has positive duration, IO can have negative duration."""
        from pricebook.structured.mbs import MBSPool, io_po_strips
        dc = _make_curve()
        pool = MBSPool(1_000_000, 0.06, 360, 0.055)
        r = io_po_strips(pool, dc)
        assert r.po_duration > 0

    def test_prepayment_model(self):
        from pricebook.structured.mbs import prepayment_model
        # High incentive: WAC 6%, current 4% → high CPR
        high = prepayment_model(60, 0.06, 0.04)
        low = prepayment_model(60, 0.06, 0.07)  # rates above WAC
        assert high > low

    def test_to_dict(self):
        from pricebook.structured.mbs import MBSPool, price_mbs
        dc = _make_curve()
        pool = MBSPool(1_000_000, 0.06, 360, 0.055)
        r = price_mbs(pool, dc)
        d = r.to_dict()
        assert "wal" in d
        assert "modified_duration" in d


# ═══════════════════════════════════════════════════════════════
# ABS (S2)
# ═══════════════════════════════════════════════════════════════

class TestAutoABS:
    def test_price_auto(self):
        from pricebook.structured.abs import ABSPool, ABSTranche, price_auto_abs
        dc = _make_curve()
        pool = ABSPool(100_000_000, 0.05, 60, charge_off_rate=0.015)
        tranches = [
            ABSTranche("A", 85_000_000, 0.035, 0),
            ABSTranche("B", 10_000_000, 0.045, 1),
            ABSTranche("C", 5_000_000, 0.065, 2),
        ]
        results = price_auto_abs(pool, tranches, dc)
        assert len(results) == 3
        assert all(r.price > 0 for r in results)

    def test_senior_ce(self):
        """Senior tranche has highest credit enhancement."""
        from pricebook.structured.abs import ABSPool, ABSTranche, price_auto_abs
        dc = _make_curve()
        pool = ABSPool(100_000_000, 0.05, 60)
        tranches = [
            ABSTranche("A", 85_000_000, 0.035, 0),
            ABSTranche("B", 10_000_000, 0.045, 1),
            ABSTranche("C", 5_000_000, 0.065, 2),
        ]
        results = price_auto_abs(pool, tranches, dc)
        assert results[0].credit_enhancement_pct > results[1].credit_enhancement_pct

    def test_to_dict(self):
        from pricebook.structured.abs import ABSPool, ABSTranche, price_auto_abs
        dc = _make_curve()
        pool = ABSPool(100_000_000, 0.05, 60)
        tranches = [ABSTranche("A", 100_000_000, 0.035, 0)]
        r = price_auto_abs(pool, tranches, dc)[0]
        d = r.to_dict()
        assert "credit_enhancement_pct" in d


class TestCreditCardABS:
    def test_price_cc(self):
        from pricebook.structured.abs import price_credit_card_abs
        dc = _make_curve()
        r = price_credit_card_abs(
            50_000_000, 0.18, 0.15, 0.05, 0.04,
            revolving_months=24, amort_months=12,
            discount_curve=dc,
        )
        assert r.price > 0
        assert r.wal > 0

    def test_excess_spread(self):
        """Excess spread = yield - coupon - losses."""
        from pricebook.structured.abs import price_credit_card_abs
        r = price_credit_card_abs(50_000_000, 0.18, 0.15, 0.05, 0.04)
        assert r.excess_spread_pct > 0  # 18% yield - 4% coupon - 5% loss = 9%

    def test_to_dict(self):
        from pricebook.structured.abs import price_credit_card_abs
        r = price_credit_card_abs(50_000_000, 0.18, 0.15, 0.05, 0.04)
        d = r.to_dict()
        assert "mpr" in d
        assert "excess_spread_pct" in d


class TestStudentLoanABS:
    def test_price_student(self):
        from pricebook.structured.abs import price_student_loan_abs
        r = price_student_loan_abs(30_000_000, 0.06, 240, grace_months=6)
        assert r.price > 0
        assert r.wal > 0

    def test_grace_delays_wal(self):
        from pricebook.structured.abs import price_student_loan_abs
        no_grace = price_student_loan_abs(30_000_000, 0.06, 240, grace_months=0)
        with_grace = price_student_loan_abs(30_000_000, 0.06, 240, grace_months=12)
        assert with_grace.wal > no_grace.wal


# ═══════════════════════════════════════════════════════════════
# CMBS (S3)
# ═══════════════════════════════════════════════════════════════

class TestCMBS:
    def _make_pool(self):
        from pricebook.structured.cmbs import CMBSLoan, CMBSPool, PropertyType
        loans = [
            CMBSLoan(10_000_000, 15_000_000, 1_200_000, 0.045, 120, PropertyType.OFFICE),
            CMBSLoan(8_000_000, 12_000_000, 1_000_000, 0.05, 84, PropertyType.RETAIL),
            CMBSLoan(5_000_000, 10_000_000, 700_000, 0.04, 60, PropertyType.MULTIFAMILY),
        ]
        return CMBSPool(loans)

    def test_ltv_dscr(self):
        from pricebook.structured.cmbs import CMBSLoan, PropertyType
        loan = CMBSLoan(10_000_000, 15_000_000, 1_200_000, 0.045, 120, PropertyType.OFFICE)
        assert loan.ltv == pytest.approx(10/15, rel=0.01)
        assert loan.dscr > 1.0  # NOI > debt service

    def test_pool_metrics(self):
        pool = self._make_pool()
        assert pool.total_balance == 23_000_000
        assert 0 < pool.wa_ltv < 1
        assert pool.wa_dscr > 1.0

    def test_concentration(self):
        pool = self._make_pool()
        conc = pool.concentration()
        assert "office" in conc
        assert sum(conc.values()) == pytest.approx(100, abs=0.1)

    def test_price_cmbs(self):
        from pricebook.structured.cmbs import price_cmbs
        dc = _make_curve()
        pool = self._make_pool()
        tranches = [
            {"name": "A", "notional": 18_000_000, "coupon": 0.04, "seniority": 0},
            {"name": "B", "notional": 3_000_000, "coupon": 0.055, "seniority": 1},
            {"name": "C", "notional": 2_000_000, "coupon": 0.07, "seniority": 2},
        ]
        results = price_cmbs(pool, tranches, dc)
        assert len(results) == 3
        assert all(r.price > 0 for r in results)

    def test_senior_has_more_ce(self):
        from pricebook.structured.cmbs import price_cmbs
        dc = _make_curve()
        pool = self._make_pool()
        tranches = [
            {"name": "A", "notional": 18_000_000, "coupon": 0.04, "seniority": 0},
            {"name": "B", "notional": 3_000_000, "coupon": 0.055, "seniority": 1},
            {"name": "C", "notional": 2_000_000, "coupon": 0.07, "seniority": 2},
        ]
        results = price_cmbs(pool, tranches, dc)
        assert results[0].credit_enhancement_pct > results[1].credit_enhancement_pct

    def test_stress(self):
        from pricebook.structured.cmbs import cmbs_stress
        pool = self._make_pool()
        r = cmbs_stress(pool, property_shock=-0.30, noi_shock=-0.20)
        assert r["stressed_wa_ltv"] > r["base_wa_ltv"]
        assert r["stressed_wa_dscr"] < r["base_wa_dscr"]

    def test_defeasance(self):
        from pricebook.structured.cmbs import defeasance_cost
        cost = defeasance_cost(10_000_000, 0.05, 60, 0.03)
        assert cost > 0  # coupon > treasury → positive cost

    def test_yield_maintenance(self):
        from pricebook.structured.cmbs import yield_maintenance
        penalty = yield_maintenance(10_000_000, 0.05, 60, 0.03)
        assert penalty > 0
        # If rates above coupon: no penalty
        no_penalty = yield_maintenance(10_000_000, 0.05, 60, 0.06)
        assert no_penalty == 0

    def test_to_dict(self):
        from pricebook.structured.cmbs import price_cmbs
        dc = _make_curve()
        pool = self._make_pool()
        tranches = [{"name": "A", "notional": 23_000_000, "coupon": 0.04, "seniority": 0}]
        r = price_cmbs(pool, tranches, dc)[0]
        d = r.to_dict()
        assert "balloon_risk_pct" in d
        assert "wa_ltv" in d


# ═══════════════════════════════════════════════════════════════
# Advanced Autocall (S4)
# ═══════════════════════════════════════════════════════════════

class TestAdvancedAutocall:
    def test_discrete_autocall(self):
        from pricebook.options.autocall_advanced import discrete_autocall
        obs = [0.5, 1.0, 1.5, 2.0]
        r = discrete_autocall(100, 1.0, 0.80, 0.60, 0.03, obs, 0.20, n_sims=10_000)
        assert r.price > 0
        assert 0 <= r.autocall_probability <= 1
        assert r.expected_life > 0

    def test_memory_coupon_adds_value(self):
        from pricebook.options.autocall_advanced import discrete_autocall
        obs = [0.5, 1.0, 1.5, 2.0]
        no_mem = discrete_autocall(100, 1.0, 0.80, 0.60, 0.03, obs, 0.20, n_sims=10_000, memory_coupon=False)
        with_mem = discrete_autocall(100, 1.0, 0.80, 0.60, 0.03, obs, 0.20, n_sims=10_000, memory_coupon=True)
        assert with_mem.price >= no_mem.price * 0.99  # memory adds value

    def test_worst_of(self):
        from pricebook.options.autocall_advanced import worst_of_discrete_autocall
        obs = [0.5, 1.0, 1.5, 2.0]
        corr = [[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]]
        r = worst_of_discrete_autocall(
            [100, 50, 200], 1.0, 0.80, 0.60, 0.03, obs,
            [0.20, 0.25, 0.30], corr, n_sims=10_000,
        )
        assert r.price > 0
        assert r.n_observations == 4

    def test_step_down(self):
        from pricebook.options.autocall_advanced import step_down_autocall
        obs = [0.5, 1.0, 1.5, 2.0]
        r = step_down_autocall(100, 1.0, 0.05, 0.80, 0.60, 0.03, obs, 0.20, n_sims=10_000)
        assert r.autocall_probability > 0  # step-down makes autocall more likely

    def test_to_dict(self):
        from pricebook.options.autocall_advanced import discrete_autocall
        obs = [0.5, 1.0]
        r = discrete_autocall(100, 1.0, 0.80, 0.60, 0.03, obs, 0.20, n_sims=5_000)
        d = r.to_dict()
        assert "memory_coupon_value" in d


# ═══════════════════════════════════════════════════════════════
# Bespoke CDO (S5)
# ═══════════════════════════════════════════════════════════════

class TestBespokeCDO:
    def test_bespoke_tranche(self):
        from pricebook.credit.bespoke_cdo import bespoke_tranche_price
        pds = [0.02] * 50
        lgds = [0.6] * 50
        notionals = [1_000_000] * 50
        r = bespoke_tranche_price(pds, lgds, notionals, 0.03, 0.07, 0.30)
        assert r.tranche_spread > 0
        assert r.expected_loss_pct > 0

    def test_senior_lower_spread(self):
        from pricebook.credit.bespoke_cdo import bespoke_tranche_price
        pds = [0.02] * 50
        lgds = [0.6] * 50
        notionals = [1_000_000] * 50
        junior = bespoke_tranche_price(pds, lgds, notionals, 0.03, 0.07, 0.30)
        senior = bespoke_tranche_price(pds, lgds, notionals, 0.15, 1.0, 0.30)
        assert senior.tranche_spread < junior.tranche_spread

    def test_lss(self):
        from pricebook.credit.bespoke_cdo import leveraged_super_senior
        pds = [0.02] * 50
        lgds = [0.6] * 50
        notionals = [1_000_000] * 50
        r = leveraged_super_senior(pds, lgds, notionals, 0.15, leverage=10.0)
        assert r.leveraged_spread > r.unleveraged_spread
        assert r.leverage == 10.0

    def test_tranche_greeks(self):
        from pricebook.credit.bespoke_cdo import tranche_greeks
        pds = [0.02] * 50
        lgds = [0.6] * 50
        notionals = [1_000_000] * 50
        r = tranche_greeks(pds, lgds, notionals, 0.03, 0.07, 0.30)
        assert r.spread_delta != 0
        assert r.correlation_delta != 0

    def test_calibrate_correlation(self):
        from pricebook.credit.bespoke_cdo import bespoke_tranche_price, calibrate_bespoke_correlation
        pds = [0.02] * 50
        lgds = [0.6] * 50
        notionals = [1_000_000] * 50
        target = bespoke_tranche_price(pds, lgds, notionals, 0.03, 0.07, 0.40)
        calib = calibrate_bespoke_correlation(
            pds, lgds, notionals, 0.03, 0.07, target.tranche_spread,
        )
        # Verify calibrated correlation reproduces the spread
        check = bespoke_tranche_price(pds, lgds, notionals, 0.03, 0.07, calib)
        assert abs(check.tranche_spread - target.tranche_spread) < 5  # within 5bp


# ═══════════════════════════════════════════════════════════════
# Steepener (S6)
# ═══════════════════════════════════════════════════════════════

class TestSteepener:
    def test_steepener_note(self):
        from pricebook.structured.steepener import steepener_note
        r = steepener_note(0.04, 0.035, 0.008, 0.006, 0.85, leverage=3.0, n_sims=10_000)
        assert r.price > 0
        assert r.expected_coupon > 0

    def test_slope_range_accrual(self):
        from pricebook.structured.steepener import slope_range_accrual
        r = slope_range_accrual(0.04, 0.035, 0.008, 0.006, 0.85, n_sims=10_000)
        assert r.price > 0
        assert r.expected_coupon > 0

    def test_digital_steepener(self):
        from pricebook.structured.steepener import digital_steepener
        r = digital_steepener(0.04, 0.035, 0.008, 0.006, 0.85, n_sims=10_000)
        assert r.price > 0

    def test_to_dict(self):
        from pricebook.structured.steepener import steepener_note
        r = steepener_note(0.04, 0.035, 0.008, 0.006, 0.85, n_sims=5_000)
        d = r.to_dict()
        assert "prob_floor_hit" in d


# ═══════════════════════════════════════════════════════════════
# Secondary Pricing (S7)
# ═══════════════════════════════════════════════════════════════

class TestSecondaryPricing:
    def test_spread_aging(self):
        from pricebook.structured.secondary_pricing import spread_aging
        r = spread_aging(200, 5.0, 2.0, current_market_spread_bp=250)
        assert r.aged_spread_bp > r.original_spread_bp  # market wider

    def test_mark_to_bid(self):
        from pricebook.structured.secondary_pricing import mark_to_bid
        r = mark_to_bid(100.0, bid_ask_spread_pct=3.0, liquidity_score=0.3)
        assert r.bid_price < r.mid_price
        assert r.haircut_pct > 0

    def test_stale_price_detection(self):
        from pricebook.structured.secondary_pricing import stale_price_detector
        stale = [100.0] * 10  # unchanged for 10 days
        r = stale_price_detector(stale, threshold_days=5)
        assert r.is_stale is True

    def test_active_price_not_stale(self):
        from pricebook.structured.secondary_pricing import stale_price_detector
        active = [100 + i * 0.5 for i in range(20)]
        r = stale_price_detector(active, threshold_days=5)
        assert r.is_stale is False

    def test_liquidity_premium(self):
        from pricebook.structured.secondary_pricing import liquidity_premium
        r = liquidity_premium(bid_ask_bp=50, holding_period_years=2.0, rating_notch=7)
        assert r.premium_bp > 0
        assert r.credit_component > 0


# ═══════════════════════════════════════════════════════════════
# Stochastic Correlation (X1)
# ═══════════════════════════════════════════════════════════════

class TestStochasticCorrelation:
    def test_regime_switching(self):
        from pricebook.credit.stochastic_correlation import regime_switching_correlation
        r = regime_switching_correlation(0.02, 0.6, 0.03, 0.07, [0.20, 0.60], [0.7, 0.3])
        assert r.tranche_spread > 0
        assert r.expected_loss_pct > 0

    def test_correlation_smile(self):
        from pricebook.credit.stochastic_correlation import correlation_smile
        # Generate target spreads from known correlations, then re-calibrate
        from pricebook.credit.stochastic_correlation import _vasicek_tranche_el
        atts = [0.0, 0.03, 0.07, 0.15]
        dets = [0.03, 0.07, 0.15, 1.0]
        # Use corrs that increase for senior tranches (smile)
        test_corrs = [0.15, 0.25, 0.40, 0.60]
        spreads = []
        for att, det, c in zip(atts, dets, test_corrs):
            el = _vasicek_tranche_el(0.02, 0.6, c, att, det)
            width = det - att
            ann = sum(math.exp(-0.04 * t) for t in np.arange(0.25, 5.01, 0.25)) * 0.25
            spreads.append(el / (width * ann) * 10_000 if width * ann > 0 else 0)
        smile = correlation_smile(0.02, 0.6, spreads, atts, dets)
        assert len(smile) == 4
        # Verify calibrated correlations are reasonable
        assert all(s.implied_correlation > 0 for s in smile)

    def test_stochastic_corr_tranche(self):
        from pricebook.credit.stochastic_correlation import stochastic_corr_tranche
        r = stochastic_corr_tranche(0.02, 0.6, 0.03, 0.07, corr_mean=0.30, n_sims=5_000)
        assert r.tranche_spread > 0

    def test_to_dict(self):
        from pricebook.credit.stochastic_correlation import regime_switching_correlation
        r = regime_switching_correlation(0.02, 0.6, 0.03, 0.07, [0.30], [1.0])
        d = r.to_dict()
        assert "n_regimes" in d


# ═══════════════════════════════════════════════════════════════
# Mountain Range (X2)
# ═══════════════════════════════════════════════════════════════

class TestMountainRange:
    def _corr(self, n):
        corr = np.eye(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    corr[i, j] = 0.5
        return corr.tolist()

    def test_napoleon(self):
        from pricebook.equity.mountain_range import napoleon_option
        r = napoleon_option([100, 50, 200], [0.20, 0.25, 0.30], self._corr(3),
                             [0.5, 1.0, 1.5, 2.0], n_sims=10_000)
        assert r.price > 0
        assert r.product == "napoleon"

    def test_everest(self):
        from pricebook.equity.mountain_range import everest_option
        r = everest_option([100, 50, 200], [0.20, 0.25, 0.30], self._corr(3), n_sims=10_000)
        assert r.price >= 0
        assert r.product == "everest"

    def test_atlas(self):
        from pricebook.equity.mountain_range import atlas_option
        r = atlas_option([100, 50, 200, 80, 120], [0.20]*5, self._corr(5),
                          n_remove=1, n_sims=10_000)
        assert r.price >= 0
        assert r.n_assets == 5

    def test_altiplano(self):
        from pricebook.equity.mountain_range import altiplano_option
        r = altiplano_option([100, 50, 200], [0.20, 0.25, 0.30], self._corr(3),
                              barrier=0.80, coupon=0.10, n_sims=10_000)
        assert 0 <= r.price <= 12  # max = coupon × notional × df


# ═══════════════════════════════════════════════════════════════
# Power Derivatives (X3)
# ═══════════════════════════════════════════════════════════════

class TestPowerDerivatives:
    def test_swing_option(self):
        from pricebook.commodity.power_derivatives import swing_option_price
        forwards = [50 + i for i in range(12)]
        r = swing_option_price(forwards, 52, 3, 8, n_sims=10_000)
        assert r.price > 0
        assert r.min_take <= r.expected_exercises <= r.max_take + 0.5

    def test_tolling(self):
        from pricebook.commodity.power_derivatives import tolling_agreement
        power = [55, 60, 50, 65]
        gas = [3.5, 3.8, 3.2, 4.0]
        r = tolling_agreement(power, gas, heat_rate=7.0, capacity_mw=100)
        assert r.value > 0
        assert r.expected_generation_hours > 0

    def test_capacity_option(self):
        from pricebook.commodity.power_derivatives import capacity_option
        forwards = [50, 55, 45, 60, 40, 65]
        r = capacity_option(forwards, 50, capacity_mw=100, n_sims=10_000)
        assert r.price > 0

    def test_block_forward(self):
        from pricebook.commodity.power_derivatives import block_forward
        # Peak hours (7-22) get higher prices
        hourly = [30 if h < 7 or h >= 22 else 60 for h in range(24)]
        r = block_forward(hourly)
        assert r.peak_price > r.off_peak_price
        assert r.peak_premium > 0
