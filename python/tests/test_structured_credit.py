"""Structured credit tests: SPV, fund participation, illiquid pricing."""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.discount_curve import DiscountCurve
from pricebook.spv import SPV, SPVTranche
from pricebook.fund_participation import FundParticipation
from pricebook.illiquid_pricing import (
    MatrixPricer, Comparable, LiquidityPremiumModel, PrivatePlacementPricer,
)


REF = date(2024, 7, 15)


def _disc():
    return DiscountCurve.flat(REF, 0.04)


def _spv():
    tranches = [
        SPVTranche("AAA", 70_000_000, 0.012, 1),
        SPVTranche("AA", 10_000_000, 0.018, 2),
        SPVTranche("BBB", 10_000_000, 0.035, 3),
        SPVTranche("Equity", 10_000_000, 0.0, 4),
    ]
    return SPV(100_000_000, pool_coupon=0.06, tranches=tranches,
               n_periods=20, cdr=0.02, recovery=0.40)


# ── SPV ──

class TestSPV:

    def test_projection_has_periods(self):
        proj = _spv().project()
        assert len(proj.periods) == 20

    def test_tranche_irr_keys(self):
        proj = _spv().project()
        assert "AAA" in proj.tranche_irr
        assert "Equity" in proj.tranche_irr

    def test_senior_irr_above_coupon(self):
        """AAA tranche should earn roughly SOFR + spread."""
        proj = _spv().project()
        assert proj.tranche_irr["AAA"] > 0.03  # base_rate + coupon

    def test_credit_enhancement_senior_highest(self):
        proj = _spv().project()
        assert proj.credit_enhancement["AAA"] > proj.credit_enhancement["BBB"]
        assert proj.credit_enhancement["Equity"] == 0.0

    def test_total_losses_positive(self):
        proj = _spv().project()
        assert proj.total_losses > 0

    def test_break_even_cdr(self):
        proj = _spv().project()
        assert proj.break_even_cdr > 0

    def test_pool_balance_decreases(self):
        proj = _spv().project()
        assert proj.periods[-1].pool_balance < 100_000_000

    def test_zero_default_no_losses(self):
        tranches = [SPVTranche("Senior", 80e6, 0.01, 1),
                    SPVTranche("Equity", 20e6, 0.0, 2)]
        spv = SPV(100e6, 0.05, tranches, n_periods=10, cdr=0.0, cpr=0.0)
        proj = spv.project()
        assert proj.total_losses == pytest.approx(0.0, abs=1e-6)

    def test_reinvestment_delays_amortisation(self):
        """With reinvestment, pool balance stays higher longer."""
        spv_static = SPV(100e6, 0.05, [SPVTranche("A", 100e6, 0.01, 1)],
                         n_periods=10, reinvestment_periods=0)
        spv_reinvest = SPV(100e6, 0.05, [SPVTranche("A", 100e6, 0.01, 1)],
                           n_periods=10, reinvestment_periods=5)
        bal_static = spv_static.project().periods[4].pool_balance
        bal_reinvest = spv_reinvest.project().periods[4].pool_balance
        assert bal_reinvest > bal_static


# ── Fund Participation ──

class TestFundParticipation:

    def test_metrics_moic_above_one(self):
        """10% gross return should produce MOIC > 1."""
        m = FundParticipation(50e6, gross_return=0.10).metrics()
        assert m.moic > 1.0

    def test_irr_positive(self):
        m = FundParticipation(50e6, gross_return=0.10).metrics()
        assert m.irr > 0

    def test_j_curve_trough_below_one(self):
        """J-curve: early TVPI dips below 1.0 due to fees + drawdown."""
        m = FundParticipation(50e6, gross_return=0.08).metrics()
        assert m.j_curve_trough < 1.0

    def test_dpi_less_than_tvpi(self):
        """DPI ≤ TVPI (TVPI includes unrealised NAV)."""
        m = FundParticipation(50e6, gross_return=0.10).metrics()
        assert m.dpi <= m.tvpi + 0.01

    def test_secondary_pricing(self):
        fund = FundParticipation(50e6)
        sec = fund.secondary_pricing(30e6, discount_pct=0.15)
        assert sec.secondary_price == 30e6 * 0.85
        assert sec.discount_pct == 0.15

    def test_validation(self):
        with pytest.raises(ValueError):
            FundParticipation(-10e6)

    def test_project_cashflows(self):
        cfs = FundParticipation(50e6).project()
        assert len(cfs) == 8  # default fund life
        total_calls = sum(cf.capital_call for cf in cfs)
        assert total_calls == pytest.approx(50e6, rel=0.01)


# ── Matrix Pricing ──

class TestMatrixPricer:

    def _comparables(self):
        return [
            Comparable("CORP_A", "tech", "BBB", 5.0, 150),
            Comparable("CORP_B", "tech", "BBB+", 4.0, 130),
            Comparable("CORP_C", "fin", "BBB", 6.0, 160),
            Comparable("CORP_D", "tech", "A", 5.0, 100),
        ]

    def test_fair_spread_finite(self):
        pricer = MatrixPricer(self._comparables())
        result = pricer.price("tech", "BBB", 5.0)
        assert math.isfinite(result.fair_spread_bp)
        assert result.fair_spread_bp > 0

    def test_same_sector_closer(self):
        """Same-sector comparables should have more weight."""
        pricer = MatrixPricer(self._comparables())
        result = pricer.price("tech", "BBB", 5.0)
        # CORP_A is exact match → closest
        assert result.closest_name == "CORP_A"

    def test_confidence_positive(self):
        pricer = MatrixPricer(self._comparables())
        result = pricer.price("tech", "BBB", 5.0)
        assert result.confidence_bp >= 0

    def test_empty_comparables_raises(self):
        with pytest.raises(ValueError):
            MatrixPricer([])


# ── Liquidity Premium ──

class TestLiquidityPremium:

    def test_premium_positive(self):
        model = LiquidityPremiumModel()
        result = model.estimate(bid_ask_pct=0.5, issue_size_mm=200, age_years=3)
        assert result.illiquidity_premium_bp > 0

    def test_larger_issue_less_premium(self):
        model = LiquidityPremiumModel()
        r_small = model.estimate(issue_size_mm=50)
        r_large = model.estimate(issue_size_mm=1000)
        assert r_small.illiquidity_premium_bp > r_large.illiquidity_premium_bp

    def test_wider_bid_ask_more_premium(self):
        model = LiquidityPremiumModel()
        r_tight = model.estimate(bid_ask_pct=0.1)
        r_wide = model.estimate(bid_ask_pct=2.0)
        assert r_wide.illiquidity_premium_bp > r_tight.illiquidity_premium_bp


# ── Private Placement ──

class TestPrivatePlacement:

    def test_price_below_par(self):
        """With spread, price should be below par (for par coupon = rf rate)."""
        pricer = PrivatePlacementPricer(
            coupon_rate=0.04, maturity_years=5.0,
            credit_spread_bp=150, illiquidity_premium_bp=50,
        )
        result = pricer.price(_disc())
        assert result.price_per_100 < 100  # spread → discount

    def test_total_spread_is_sum(self):
        pricer = PrivatePlacementPricer(
            coupon_rate=0.05, maturity_years=5.0,
            credit_spread_bp=150, illiquidity_premium_bp=50, complexity_premium_bp=25,
        )
        assert pricer.total_spread_bp == 225

    def test_higher_spread_lower_price(self):
        p1 = PrivatePlacementPricer(0.05, 5.0, credit_spread_bp=100).price(_disc())
        p2 = PrivatePlacementPricer(0.05, 5.0, credit_spread_bp=300).price(_disc())
        assert p2.price_per_100 < p1.price_per_100

    def test_to_dict(self):
        pricer = PrivatePlacementPricer(0.05, 5.0)
        d = pricer.price(_disc()).to_dict()
        assert "total_spread_bp" in d
        assert "credit_bp" in d
        assert "illiquidity_bp" in d
