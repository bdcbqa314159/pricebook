"""Tests for exotic credit payoffs."""

import math

import pytest

from pricebook.credit_exotic import (
    CappedCouponBondResult,
    CreditLinkedLoanResult,
    CreditRangeAccrualResult,
    DigitalCDSResult,
    capped_coupon_bond,
    credit_linked_loan,
    credit_range_accrual,
    digital_cds,
)


# ---- Capped coupon bond ----

class TestCappedCouponBond:
    def test_risky_less_than_risk_free(self):
        result = capped_coupon_bond(
            notional=100, floating_rate=0.04, spread=0.02, cap=0.08,
            maturity_years=5, flat_rate=0.05, flat_hazard=0.02,
        )
        assert result.dirty_price < result.risk_free_price

    def test_cap_binds(self):
        """When floating + spread > cap, coupon is capped."""
        uncapped = capped_coupon_bond(
            100, 0.06, 0.03, cap=1.0,  # cap won't bind
            maturity_years=5,
        )
        capped = capped_coupon_bond(
            100, 0.06, 0.03, cap=0.07,  # cap binds (0.09 > 0.07)
            maturity_years=5,
        )
        assert capped.dirty_price < uncapped.dirty_price

    def test_zero_hazard_matches_risk_free(self):
        result = capped_coupon_bond(
            100, 0.04, 0.02, cap=0.10,
            maturity_years=5, flat_hazard=0.0,
        )
        assert result.dirty_price == pytest.approx(result.risk_free_price, rel=0.01)

    def test_recovery_contributes(self):
        result = capped_coupon_bond(
            100, 0.04, 0.02, cap=0.10,
            maturity_years=5, flat_hazard=0.05,
        )
        assert result.recovery_pv > 0

    def test_positive_price(self):
        result = capped_coupon_bond(100, 0.04, 0.02, 0.08, 5)
        assert result.dirty_price > 0


# ---- Digital CDS ----

class TestDigitalCDS:
    def test_par_spread_positive(self):
        result = digital_cds(1_000_000, digital_payout=1_000_000,
                             spread=0.01, maturity_years=5)
        assert result.par_spread > 0

    def test_at_par_pv_near_zero(self):
        """At par spread, PV ≈ 0."""
        result1 = digital_cds(1_000_000, 1_000_000, 0.01, 5)
        result2 = digital_cds(1_000_000, 1_000_000,
                              result1.par_spread, 5)
        assert result2.pv == pytest.approx(0.0, abs=100)

    def test_higher_payout_higher_par_spread(self):
        low = digital_cds(1_000_000, 500_000, 0.01, 5)
        high = digital_cds(1_000_000, 1_000_000, 0.01, 5)
        assert high.par_spread > low.par_spread

    def test_digital_vs_standard_cds(self):
        """Digital CDS with payout = notional × (1−R) ≈ standard CDS."""
        recovery = 0.4
        digital = digital_cds(1_000_000, 1_000_000 * (1 - recovery),
                              0.01, 5, flat_hazard=0.02)
        # Par spread should be close to the hazard-rate-based CDS spread
        # spread ≈ hazard × (1 − R) = 0.02 × 0.6 = 0.012
        assert digital.par_spread == pytest.approx(0.012, rel=0.15)


# ---- Credit range accrual ----

class TestCreditRangeAccrual:
    def test_spread_in_range_full_accrual(self):
        """If spread is well within range, accrual fraction ≈ 1."""
        result = credit_range_accrual(
            100, coupon_rate=0.05,
            lower_spread=0.005, upper_spread=0.050,
            maturity_years=1.0, current_spread=0.020,
            spread_vol=0.002,  # very low vol → stays in range
        )
        assert result.accrual_fraction > 0.8

    def test_spread_outside_range_low_accrual(self):
        """If spread is outside range, accrual fraction is low."""
        result = credit_range_accrual(
            100, 0.05,
            lower_spread=0.005, upper_spread=0.010,
            maturity_years=1.0, current_spread=0.030,
            spread_vol=0.005,
        )
        assert result.accrual_fraction < 0.3

    def test_positive_pv(self):
        result = credit_range_accrual(
            100, 0.05, 0.01, 0.05, 1.0, 0.02, 0.005,
        )
        assert result.pv > 0


# ---- Credit-linked loan ----

class TestCreditLinkedLoan:
    def test_loan_pv_less_than_principal(self):
        """Risky loan PV < principal (credit loss)."""
        result = credit_linked_loan(
            principal=1_000_000, base_rate=0.05, margin=0.02,
            maturity_years=5, flat_hazard=0.03,
        )
        assert result.pv < 1_000_000

    def test_expected_loss_positive(self):
        result = credit_linked_loan(
            1_000_000, 0.05, 0.02, 5, flat_hazard=0.03,
        )
        assert result.expected_loss > 0

    def test_covenant_breach(self):
        result = credit_linked_loan(
            1_000_000, 0.05, 0.02, 5,
            leverage_ratio=6.0, max_leverage=5.0,
        )
        assert result.covenant_breached

    def test_no_covenant_breach(self):
        result = credit_linked_loan(
            1_000_000, 0.05, 0.02, 5,
            leverage_ratio=3.0, max_leverage=5.0,
        )
        assert not result.covenant_breached

    def test_margin_grid(self):
        """Margin grid: higher leverage → higher margin."""
        low_lev = credit_linked_loan(
            1_000_000, 0.05, 0.02, 5, leverage_ratio=2.0,
            margin_grid=[(3.0, 0.015), (4.0, 0.025), (5.0, 0.035)],
        )
        high_lev = credit_linked_loan(
            1_000_000, 0.05, 0.02, 5, leverage_ratio=4.5,
            margin_grid=[(3.0, 0.015), (4.0, 0.025), (5.0, 0.035)],
        )
        assert high_lev.effective_spread > low_lev.effective_spread

    def test_zero_hazard(self):
        """Zero default risk: PV = PV of all cashflows."""
        result = credit_linked_loan(
            1_000_000, 0.05, 0.02, 5, flat_hazard=0.0,
        )
        assert result.expected_loss == pytest.approx(0.0)
