"""Tests for repo carry breakeven + credit collateral integration."""

import pytest
from datetime import date

from pricebook.fixed_income.repo_carry import (
    carry_breakeven, xccy_repo_carry, multi_ccy_carry_comparison,
    CarryBreakevenResult, XCCYCarryResult,
)
from pricebook.fixed_income.repo_credit_collateral import (
    CreditCollateralSpec, CollateralAssetClass,
    credit_adjusted_haircut, collateral_default_loss,
    repo_price_with_collateral_credit, hazard_to_haircut_mapping,
    CreditAdjustedHaircutResult,
)

REF = date(2024, 1, 15)


# ═══════════════════════════════════════════════════════════════
# 1.2: Carry Breakeven
# ═══════════════════════════════════════════════════════════════

class TestCarryBreakeven:
    def test_positive_carry(self):
        """5% coupon funded at 4% → positive carry."""
        r = carry_breakeven(100.0, 0.05, 90, gc_rate=0.04)
        assert r.carry_gc > 0

    def test_negative_carry(self):
        """3% coupon funded at 5% → negative carry."""
        r = carry_breakeven(100.0, 0.03, 90, gc_rate=0.05)
        assert r.carry_gc < 0

    def test_special_better_carry(self):
        """Special rate < GC → better carry on special."""
        r = carry_breakeven(100.0, 0.05, 90, gc_rate=0.04, special_rate=0.02)
        assert r.carry_special > r.carry_gc

    def test_breakeven_rate(self):
        """Breakeven rate should make carry = 0."""
        r = carry_breakeven(100.0, 0.05, 90, gc_rate=0.04)
        # At breakeven rate, carry should be ~0
        r2 = carry_breakeven(100.0, 0.05, 90, gc_rate=r.breakeven_gc_rate)
        assert abs(r2.carry_gc) < 0.01

    def test_term_vs_on(self):
        r = carry_breakeven(100.0, 0.05, 90, gc_rate=0.04, on_rate=0.045)
        assert r.term_vs_on_pickup_bp != 0

    def test_to_dict(self):
        d = carry_breakeven(100.0, 0.05, 90, gc_rate=0.04).to_dict()
        assert "carry_gc" in d


class TestXCCYCarry:
    def test_basic(self):
        r = xccy_repo_carry(100.0, 0.05, 90, 0.04, 0.03, xccy_basis_bp=20)
        assert isinstance(r, XCCYCarryResult)
        assert r.domestic_carry != r.foreign_carry

    def test_basis_cost_reduces_carry(self):
        """Positive xccy basis should reduce foreign carry."""
        r = xccy_repo_carry(100.0, 0.05, 90, 0.04, 0.03, xccy_basis_bp=50)
        assert r.xccy_basis_cost > 0


class TestMultiCCYComparison:
    def test_ranking(self):
        rates = {"USD": 0.05, "EUR": 0.035, "JPY": 0.005}
        basis = {"EUR": 20, "JPY": -10}
        results = multi_ccy_carry_comparison(100.0, 0.04, 90, rates, basis)
        assert len(results) == 3
        # JPY lowest rate → best carry (even with negative basis)
        assert results[0]["carry"] > results[-1]["carry"]


# ═══════════════════════════════════════════════════════════════
# 1.5: Credit Collateral Integration
# ═══════════════════════════════════════════════════════════════

class TestCreditAdjustedHaircut:
    def test_ig_vs_hy(self):
        """HY collateral should have higher haircut than IG."""
        ig = CreditCollateralSpec(CollateralAssetClass.IG_CORPORATE, "ACME", "BBB",
                                   cds_spread_bp=100, hazard_rate=0.01, recovery=0.40, duration=5)
        hy = CreditCollateralSpec(CollateralAssetClass.HY_CORPORATE, "JUNK", "B",
                                   cds_spread_bp=500, hazard_rate=0.08, recovery=0.30, duration=4)
        h_ig = credit_adjusted_haircut(ig)
        h_hy = credit_adjusted_haircut(hy)
        assert h_hy.total_haircut > h_ig.total_haircut

    def test_sovereign_lowest(self):
        sov = CreditCollateralSpec(CollateralAssetClass.SOVEREIGN, "UST", "AAA",
                                    cds_spread_bp=10, hazard_rate=0.001, recovery=0.50, duration=7)
        h = credit_adjusted_haircut(sov)
        assert h.base_haircut == 0.02  # sovereign base

    def test_credit_add_on_positive(self):
        spec = CreditCollateralSpec(CollateralAssetClass.IG_CORPORATE, "X", "BBB",
                                     cds_spread_bp=200, hazard_rate=0.03, recovery=0.40, duration=5)
        h = credit_adjusted_haircut(spec)
        assert h.credit_add_on > 0
        assert h.spread_add_on > 0

    def test_higher_hazard_higher_haircut(self):
        low = CreditCollateralSpec(CollateralAssetClass.IG_CORPORATE, "A", "A",
                                    cds_spread_bp=50, hazard_rate=0.005, recovery=0.40, duration=5)
        high = CreditCollateralSpec(CollateralAssetClass.IG_CORPORATE, "B", "BBB-",
                                     cds_spread_bp=300, hazard_rate=0.05, recovery=0.40, duration=5)
        assert credit_adjusted_haircut(high).total_haircut > credit_adjusted_haircut(low).total_haircut

    def test_at1_t2_high_base(self):
        at1 = CreditCollateralSpec(CollateralAssetClass.BANK_AT1_T2, "DB", "BB+",
                                    cds_spread_bp=400, hazard_rate=0.05, recovery=0.30, duration=3)
        h = credit_adjusted_haircut(at1)
        assert h.base_haircut == 0.15

    def test_to_dict(self):
        spec = CreditCollateralSpec(CollateralAssetClass.IG_CORPORATE, "X", "BBB",
                                     cds_spread_bp=100, hazard_rate=0.01, recovery=0.40, duration=5)
        d = credit_adjusted_haircut(spec).to_dict()
        assert "total_haircut" in d


class TestCollateralDefaultLoss:
    def test_positive(self):
        spec = CreditCollateralSpec(CollateralAssetClass.HY_CORPORATE, "X", "B",
                                     cds_spread_bp=500, hazard_rate=0.08, recovery=0.30, duration=4)
        loss = collateral_default_loss(spec, 90, 1e6)
        assert loss > 0

    def test_sovereign_near_zero(self):
        spec = CreditCollateralSpec(CollateralAssetClass.SOVEREIGN, "UST", "AAA",
                                     cds_spread_bp=5, hazard_rate=0.0005, recovery=0.50, duration=7)
        loss = collateral_default_loss(spec, 90, 1e6)
        assert loss < 100  # negligible


class TestRepoPriceWithCredit:
    def test_basic(self):
        spec = CreditCollateralSpec(CollateralAssetClass.IG_CORPORATE, "ACME", "BBB",
                                     cds_spread_bp=150, hazard_rate=0.02, recovery=0.40, duration=5)
        result = repo_price_with_collateral_credit(0.05, 1e6, 90, spec)
        assert result["interest_income"] > 0
        assert result["collateral_credit_charge"] > 0
        assert result["net_income"] < result["interest_income"]

    def test_wrong_way_risk(self):
        """High correlation → higher cost."""
        spec = CreditCollateralSpec(CollateralAssetClass.IG_CORPORATE, "X", "BBB",
                                     cds_spread_bp=200, hazard_rate=0.03, recovery=0.40, duration=5)
        r_zero = repo_price_with_collateral_credit(0.05, 1e6, 90, spec, 0.02, correlation=0.0)
        r_high = repo_price_with_collateral_credit(0.05, 1e6, 90, spec, 0.02, correlation=0.80)
        assert r_high["wrong_way_risk"] > r_zero["wrong_way_risk"]

    def test_breakeven_rate(self):
        spec = CreditCollateralSpec(CollateralAssetClass.IG_CORPORATE, "X", "BBB",
                                     cds_spread_bp=100, hazard_rate=0.01, recovery=0.40, duration=5)
        result = repo_price_with_collateral_credit(0.05, 1e6, 90, spec)
        assert result["breakeven_rate"] > 0


class TestHazardToHaircut:
    def test_mapping(self):
        hazards = [0.005, 0.01, 0.03, 0.05, 0.10]
        durations = [5.0] * 5
        mapping = hazard_to_haircut_mapping(hazards, durations)
        assert len(mapping) == 5
        # Higher hazard → higher haircut
        assert mapping[-1]["total_haircut"] > mapping[0]["total_haircut"]
