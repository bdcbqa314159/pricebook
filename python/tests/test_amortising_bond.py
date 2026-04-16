"""Tests for amortising and sinker bonds."""

import math

import numpy as np
import pytest

from pricebook.amortising_bond import (
    AmortisingBond,
    AmortisingBondResult,
    PrepaymentBondResult,
    SinkerComparisonResult,
    average_life,
    cpr_to_smm,
    prepayment_bond_price,
    psa_schedule,
    sinker_vs_bullet,
    weighted_average_maturity,
)


# ---- Amortising bond ----

class TestAmortisingBond:
    def test_mortgage_schedule(self):
        bond = AmortisingBond(1000, coupon_rate=0.05, maturity_years=10, n_payments=120)
        schedule = bond.schedule()
        assert len(schedule) == 120
        # Sum of principals ≈ notional
        total_principal = sum(p for _, p, _ in schedule)
        assert total_principal == pytest.approx(1000, rel=0.01)

    def test_linear_schedule(self):
        bond = AmortisingBond(1200, 0.05, 10, 12, "linear")
        schedule = bond.schedule()
        principals = [p for _, p, _ in schedule]
        # Equal principal each period
        assert all(p == pytest.approx(100) for p in principals)

    def test_price_basic(self):
        bond = AmortisingBond(1000, 0.05, 10, 120)
        result = bond.price(rate=0.04)
        assert isinstance(result, AmortisingBondResult)
        assert result.price > 0

    def test_average_life_less_than_maturity(self):
        """Average life of amortising bond < maturity."""
        bond = AmortisingBond(1000, 0.05, 10, 120)
        result = bond.price(0.04)
        assert result.average_life < 10

    def test_duration_less_than_maturity(self):
        bond = AmortisingBond(1000, 0.05, 10, 120)
        result = bond.price(0.04)
        assert result.duration < 10

    def test_dv01_positive(self):
        bond = AmortisingBond(1000, 0.05, 10, 120)
        result = bond.price(0.04)
        assert result.dv01 > 0

    def test_invalid_type(self):
        bond = AmortisingBond(1000, 0.05, 10, 120, "invalid")
        with pytest.raises(ValueError):
            bond.schedule()

    def test_zero_coupon_linear(self):
        """Zero coupon, linear amortisation."""
        bond = AmortisingBond(1200, 0.0, 10, 12, "linear")
        schedule = bond.schedule()
        # All interest = 0, equal principal
        assert all(i == 0 for _, _, i in schedule)


# ---- Prepayment ----

class TestCprToSmm:
    def test_zero_cpr(self):
        assert cpr_to_smm(0) == 0.0

    def test_one_cpr(self):
        assert cpr_to_smm(1.0) == 1.0

    def test_monotone(self):
        assert cpr_to_smm(0.06) > cpr_to_smm(0.03)

    def test_consistency(self):
        """CPR 6% → SMM ≈ 0.00514."""
        smm = cpr_to_smm(0.06)
        assert smm == pytest.approx(0.00514, abs=0.0001)


class TestPSASchedule:
    def test_100_psa(self):
        smm = psa_schedule(1.0, 360)
        # After month 30, should be 6% CPR ≈ 0.514% SMM
        assert smm[30] == pytest.approx(cpr_to_smm(0.06))
        assert smm[100] == pytest.approx(cpr_to_smm(0.06))

    def test_200_psa_double(self):
        smm_100 = psa_schedule(1.0, 360)
        smm_200 = psa_schedule(2.0, 360)
        # Higher PSA → higher prepayment
        assert smm_200[100] > smm_100[100]

    def test_ramp(self):
        smm = psa_schedule(1.0, 360)
        # Early months should be lower than plateau
        assert smm[5] < smm[50]


class TestPrepaymentBondPrice:
    def test_basic(self):
        result = prepayment_bond_price(
            notional=1000, coupon_rate=0.05, maturity_years=10,
            psa_speed=100, discount_rate=0.05,
        )
        assert isinstance(result, PrepaymentBondResult)
        assert result.price > 0

    def test_higher_psa_shorter_life(self):
        slow = prepayment_bond_price(1000, 0.05, 10, psa_speed=50,
                                      discount_rate=0.05)
        fast = prepayment_bond_price(1000, 0.05, 10, psa_speed=300,
                                      discount_rate=0.05)
        assert fast.average_life < slow.average_life

    def test_higher_psa_more_prepayments(self):
        slow = prepayment_bond_price(1000, 0.05, 10, 50, 0.05)
        fast = prepayment_bond_price(1000, 0.05, 10, 300, 0.05)
        assert fast.projected_prepayments_pct > slow.projected_prepayments_pct


# ---- Portfolio measures ----

class TestAverageLife:
    def test_basic(self):
        cfs = [(1, 100), (2, 100), (3, 100)]
        al = average_life(cfs)
        assert al == pytest.approx(2.0)

    def test_empty_cashflows(self):
        assert average_life([]) == 0.0

    def test_weighted(self):
        cfs = [(1, 900), (10, 100)]   # mostly near-term
        al = average_life(cfs)
        # = (1×900 + 10×100) / 1000 = (900+1000)/1000 = 1.9
        assert al == pytest.approx(1.9)


class TestWeightedAverageMaturity:
    def test_basic(self):
        bonds = [(5, 50), (10, 50)]
        wam = weighted_average_maturity(bonds)
        assert wam == 7.5

    def test_single_bond(self):
        assert weighted_average_maturity([(7, 100)]) == 7.0

    def test_zero_weights(self):
        assert weighted_average_maturity([(5, 0), (10, 0)]) == 0.0


# ---- Sinker vs bullet ----

class TestSinkerVsBullet:
    def test_basic(self):
        result = sinker_vs_bullet(1000, 0.05, 10, 10, 0.04)
        assert isinstance(result, SinkerComparisonResult)
        assert result.sinker_price > 0
        assert result.bullet_price > 0

    def test_sinker_shorter_duration(self):
        """Sinker duration < bullet duration (cashflows returned earlier)."""
        result = sinker_vs_bullet(1000, 0.05, 10, 10, 0.04)
        assert result.sinker_duration < result.bullet_duration

    def test_sinker_lower_dv01(self):
        """Sinker DV01 < bullet DV01 (shorter effective maturity)."""
        result = sinker_vs_bullet(1000, 0.05, 10, 10, 0.04)
        assert result.sinker_dv01 < result.bullet_dv01
