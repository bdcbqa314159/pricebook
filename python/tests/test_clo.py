"""Tests for CLO: waterfall, compliance, diversity, reinvestment, borrowing base."""

from __future__ import annotations

import math

import pytest

from pricebook.clo import (
    CLOTranche, CLOWaterfall,
    oc_ratio, ic_ratio, ccc_concentration,
    wal_test, warf_test, weighted_average_rating_factor,
    moody_diversity_score,
    reinvestment_capacity, break_even_default_rate,
    BorrowingBase,
    RATING_FACTORS,
)


# ---- Helper: standard CLO structure ----

def _make_standard_clo() -> CLOWaterfall:
    """Standard CLO: AAA/AA/A/BBB/BB/Equity."""
    tranches = [
        CLOTranche("AAA", 250_000_000, 0.012, 1),
        CLOTranche("AA",   40_000_000, 0.018, 2),
        CLOTranche("A",    25_000_000, 0.025, 3),
        CLOTranche("BBB",  20_000_000, 0.035, 4),
        CLOTranche("BB",   15_000_000, 0.055, 5),
        CLOTranche("Equity", 50_000_000, 0.0, 6),
    ]
    return CLOWaterfall(tranches)


# ---- CLO Tranche ----

class TestCLOTranche:

    def test_coupon_due(self):
        t = CLOTranche("AAA", 100_000_000, 0.012, 1)
        assert t.coupon_due == pytest.approx(1_200_000)

    def test_equity_zero_coupon(self):
        t = CLOTranche("Equity", 50_000_000, 0.0, 6)
        assert t.coupon_due == 0.0

    def test_to_dict(self):
        t = CLOTranche("BBB", 20_000_000, 0.035, 4)
        d = t.to_dict()
        assert d["name"] == "BBB"
        assert d["notional"] == 20_000_000
        assert d["coupon"] == 0.035
        assert d["seniority"] == 4


# ---- CLO Waterfall ----

class TestCLOWaterfall:

    def test_sorted_by_seniority(self):
        wf = _make_standard_clo()
        seniorities = [t.seniority for t in wf.tranches]
        assert seniorities == sorted(seniorities)

    def test_total_notional(self):
        wf = _make_standard_clo()
        assert wf.total_notional == pytest.approx(400_000_000)

    def test_debt_tranches(self):
        wf = _make_standard_clo()
        assert len(wf.debt_tranches) == 5
        assert all(t.name != "Equity" for t in wf.debt_tranches)

    def test_equity_tranche(self):
        wf = _make_standard_clo()
        eq = wf.equity_tranche
        assert eq is not None
        assert eq.name == "Equity"

    def test_interest_waterfall_pays_senior_first(self):
        wf = _make_standard_clo()
        # Income covers AAA coupon only
        aaa_coupon = 250_000_000 * 0.012  # 3M
        payments = wf.distribute_interest(aaa_coupon, 400_000_000)
        # After mgmt fee, AAA should get most
        assert payments["AAA"] > 0
        # Equity gets nothing (income exhausted)
        assert payments.get("Equity", 0) < aaa_coupon

    def test_interest_waterfall_equity_residual(self):
        wf = _make_standard_clo()
        # Generous income
        payments = wf.distribute_interest(20_000_000, 400_000_000)
        # All tranches should get their coupon, equity gets residual
        assert payments["Equity"] > 0

    def test_mgmt_fee_paid_first(self):
        wf = _make_standard_clo()
        payments = wf.distribute_interest(100_000, 400_000_000)
        # Only mgmt fee should be paid (income too small for coupons)
        assert payments["mgmt_fee"] > 0

    def test_principal_sequential(self):
        wf = _make_standard_clo()
        payments = wf.distribute_principal(50_000_000, sequential=True)
        # All goes to AAA (most senior)
        assert payments["AAA"] == pytest.approx(50_000_000)

    def test_principal_sequential_exceeds_senior(self):
        wf = _make_standard_clo()
        # More than AAA outstanding
        payments = wf.distribute_principal(260_000_000, sequential=True)
        assert payments["AAA"] == pytest.approx(250_000_000)
        assert payments["AA"] == pytest.approx(10_000_000)

    def test_principal_pro_rata(self):
        wf = _make_standard_clo()
        payments = wf.distribute_principal(40_000_000, sequential=False)
        # Each tranche gets proportional share
        total = wf.total_notional
        for tranche in wf.tranches:
            expected = 40_000_000 * tranche.notional / total
            assert payments[tranche.name] == pytest.approx(expected, rel=1e-6)

    def test_distribute_no_losses(self):
        wf = _make_standard_clo()
        payments = wf.distribute(15_000_000, 0, 0, 400_000_000)
        assert payments["AAA"] > 0
        assert payments["Equity"] >= 0

    def test_distribute_losses_hit_equity(self):
        wf = _make_standard_clo()
        # 30M loss, 10M recovery → 20M net loss
        payments = wf.distribute(15_000_000, 30_000_000, 10_000_000, 400_000_000)
        # Equity absorbs the loss
        assert payments["Equity"] < 0

    def test_distribute_losses_exceed_equity(self):
        wf = _make_standard_clo()
        # 60M loss, no recovery → exceeds equity (50M)
        payments = wf.distribute(15_000_000, 60_000_000, 0, 400_000_000)
        # BB also takes a hit
        assert payments["Equity"] < 0
        assert payments["BB"] < 0
        # AAA is protected
        assert payments["AAA"] >= 0

    def test_to_dict(self):
        wf = _make_standard_clo()
        d = wf.to_dict()
        assert len(d["tranches"]) == 6
        assert d["mgmt_fee"] == 0.0015


# ---- OC/IC ratios ----

class TestCompliance:

    def test_oc_overcollateralised(self):
        # 400M assets / 330M debt = ~1.21
        assert oc_ratio(400_000_000, 330_000_000) > 1.0

    def test_oc_undercollateralised(self):
        assert oc_ratio(300_000_000, 330_000_000) < 1.0

    def test_oc_zero_tranche(self):
        assert oc_ratio(400_000_000, 0) == float("inf")

    def test_ic_passes(self):
        assert ic_ratio(15_000_000, 10_000_000) > 1.0

    def test_ic_fails(self):
        assert ic_ratio(5_000_000, 10_000_000) < 1.0

    def test_ic_zero_coupon(self):
        assert ic_ratio(15_000_000, 0) == float("inf")

    def test_ccc_concentration(self):
        ratings = ["BB", "B", "CCC", "B", "CCC+"]
        notionals = [20, 30, 10, 25, 15]
        ccc_pct = ccc_concentration(ratings, notionals)
        # CCC + CCC+ = 25 out of 100 = 25%
        assert ccc_pct == pytest.approx(0.25)

    def test_ccc_no_ccc(self):
        ratings = ["BB", "B", "BB+"]
        notionals = [30, 40, 30]
        assert ccc_concentration(ratings, notionals) == 0.0

    def test_wal_test_passes(self):
        assert wal_test(4.5, 6.0) is True

    def test_wal_test_fails(self):
        assert wal_test(7.0, 6.0) is False

    def test_warf_test_passes(self):
        # All B2 → WARF = 2720, test against 3000
        ratings = ["B2", "B2", "B2"]
        notionals = [100, 100, 100]
        assert warf_test(ratings, notionals, 3000) is True

    def test_warf_test_fails(self):
        # All Caa1 → WARF = 4770, test against 3000
        ratings = ["Caa1", "Caa1"]
        notionals = [100, 100]
        assert warf_test(ratings, notionals, 3000) is False

    def test_warf_computation(self):
        ratings = ["B1", "B2"]
        notionals = [50, 50]
        warf = weighted_average_rating_factor(ratings, notionals)
        expected = (2220 + 2720) / 2
        assert warf == pytest.approx(expected)

    def test_sp_ratings_work(self):
        ratings = ["BB+", "B", "CCC"]
        notionals = [100, 100, 100]
        warf = weighted_average_rating_factor(ratings, notionals)
        expected = (940 + 2720 + 6500) / 3
        assert warf == pytest.approx(expected)


# ---- Diversity score ----

class TestDiversityScore:

    def test_single_industry(self):
        # 3 names, same industry, equal size
        industries = ["tech", "tech", "tech"]
        notionals = [100, 100, 100]
        ds = moody_diversity_score(industries, notionals)
        # Single industry, 3 equal names → HHI=1/3, eff_n=3, div=1+0.5*2=2.0
        assert ds == pytest.approx(2.0)

    def test_multiple_industries(self):
        # 3 different industries, 1 name each
        industries = ["tech", "energy", "healthcare"]
        notionals = [100, 100, 100]
        ds = moody_diversity_score(industries, notionals)
        # Each industry: 1 name → div=1.0. Total=3.0
        assert ds == pytest.approx(3.0)

    def test_diversity_increases_with_industries(self):
        # More industries → higher diversity
        ds2 = moody_diversity_score(["tech", "energy"], [100, 100])
        ds5 = moody_diversity_score(
            ["tech", "energy", "healthcare", "retail", "banking"],
            [100, 100, 100, 100, 100],
        )
        assert ds5 > ds2

    def test_concentration_reduces_diversity(self):
        # Equal vs concentrated within industry
        equal = moody_diversity_score(["tech", "tech"], [100, 100])
        concentrated = moody_diversity_score(["tech", "tech"], [190, 10])
        assert equal > concentrated

    def test_empty(self):
        assert moody_diversity_score([], []) == 0.0


# ---- Reinvestment ----

class TestReinvestment:

    def test_at_par(self):
        cap = reinvestment_capacity(10_000_000, 100.0)
        assert cap == pytest.approx(10_000_000)

    def test_at_discount(self):
        cap = reinvestment_capacity(10_000_000, 98.0)
        assert cap > 10_000_000  # buy more par at discount

    def test_at_premium(self):
        cap = reinvestment_capacity(10_000_000, 102.0)
        assert cap < 10_000_000  # buy less par at premium

    def test_zero_price(self):
        assert reinvestment_capacity(10_000_000, 0) == 0.0


# ---- Break-even default rate ----

class TestBEDR:

    def test_positive(self):
        bedr = break_even_default_rate(50_000_000, 400_000_000, 0.70)
        assert bedr > 0

    def test_higher_equity_higher_bedr(self):
        bedr1 = break_even_default_rate(30_000_000, 400_000_000, 0.70)
        bedr2 = break_even_default_rate(60_000_000, 400_000_000, 0.70)
        assert bedr2 > bedr1

    def test_higher_recovery_higher_bedr(self):
        bedr1 = break_even_default_rate(50_000_000, 400_000_000, 0.50)
        bedr2 = break_even_default_rate(50_000_000, 400_000_000, 0.80)
        assert bedr2 > bedr1

    def test_formula(self):
        bedr = break_even_default_rate(50_000_000, 400_000_000, 0.70)
        expected = 50_000_000 / (400_000_000 * 0.30)
        assert bedr == pytest.approx(expected)

    def test_full_recovery(self):
        assert break_even_default_rate(50_000_000, 400_000_000, 1.0) == 0.0


# ---- Borrowing base ----

class TestBorrowingBase:

    def test_gross_availability(self):
        bb = BorrowingBase(100_000_000, advance_rate=0.85)
        assert bb.gross_availability == pytest.approx(85_000_000)

    def test_available_draw_no_limit(self):
        bb = BorrowingBase(100_000_000, advance_rate=0.85)
        assert bb.available_draw == pytest.approx(85_000_000)

    def test_available_draw_with_limit(self):
        bb = BorrowingBase(100_000_000, advance_rate=0.85, facility_limit=50_000_000)
        assert bb.available_draw == pytest.approx(50_000_000)

    def test_concentration_excess(self):
        bb = BorrowingBase(100_000_000, concentration_limit=0.10)
        obligors = {"A": 15_000_000, "B": 5_000_000, "C": 10_000_000}
        excess = bb.concentration_excess(obligors)
        # Total = 30M, limit = 10% = 3M
        assert "A" in excess
        assert excess["A"] == pytest.approx(15_000_000 - 3_000_000)

    def test_no_concentration_excess(self):
        bb = BorrowingBase(100_000_000, concentration_limit=0.50)
        obligors = {"A": 10_000_000, "B": 10_000_000}
        excess = bb.concentration_excess(obligors)
        assert len(excess) == 0

    def test_to_dict(self):
        bb = BorrowingBase(100_000_000, advance_rate=0.85, facility_limit=50_000_000)
        d = bb.to_dict()
        assert d["eligible_receivables"] == 100_000_000
        assert d["advance_rate"] == 0.85
        assert d["facility_limit"] == 50_000_000
