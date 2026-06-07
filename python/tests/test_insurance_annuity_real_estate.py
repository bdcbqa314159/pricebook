"""Tests for pricebook.structured.insurance_annuity and real_estate_derivative."""

import pytest

from pricebook.structured.insurance_annuity import gmab, gmwb, ratchet_gmab
from pricebook.structured.real_estate_derivative import (
    property_total_return_swap,
    property_option,
    housing_affordability,
)

# ---- Common parameters -------------------------------------------------------

INV   = 100_000.0   # initial investment
VOL   = 0.20
R     = 0.04
T     = 10.0


# ---- GMAB -------------------------------------------------------------------

class TestGMAB:
    def test_guarantee_cost_positive(self):
        res = gmab(INV, guarantee_rate=0.0, fee_rate=0.015, vol=VOL, risk_free_rate=R, T=T, n_paths=10_000, seed=0)
        assert res.guarantee_cost > 0.0

    def test_higher_vol_higher_cost(self):
        low_vol = gmab(INV, 0.0, 0.015, vol=0.10, risk_free_rate=R, T=T, n_paths=10_000, seed=0)
        high_vol = gmab(INV, 0.0, 0.015, vol=0.30, risk_free_rate=R, T=T, n_paths=10_000, seed=0)
        assert high_vol.guarantee_cost > low_vol.guarantee_cost

    def test_ratchet_more_expensive_than_plain(self):
        plain = gmab(INV, guarantee_rate=0.0, fee_rate=0.015, vol=VOL, risk_free_rate=R, T=T, n_paths=10_000, seed=42)
        ratchet = ratchet_gmab(INV, fee_rate=0.015, vol=VOL, risk_free_rate=R, T=T, n_paths=10_000, seed=42)
        assert ratchet.guarantee_cost > plain.guarantee_cost

    def test_prob_in_the_money_in_unit_interval(self):
        res = gmab(INV, 0.0, 0.015, VOL, R, T, n_paths=5_000, seed=1)
        assert 0.0 <= res.prob_in_the_money <= 1.0

    def test_guarantee_value_matches_formula(self):
        import math
        res = gmab(INV, guarantee_rate=0.02, fee_rate=0.0, vol=VOL, risk_free_rate=R, T=T, n_paths=5_000, seed=0)
        expected_g = INV * math.exp(0.02 * T)
        assert res.guarantee_value == pytest.approx(expected_g, rel=1e-9)


# ---- GMWB -------------------------------------------------------------------

class TestGMWB:
    def test_guarantee_cost_positive(self):
        res = gmwb(INV, withdrawal_rate=0.07, fee_rate=0.015, vol=VOL, risk_free_rate=R, T=T, n_paths=10_000, seed=42)
        assert res.guarantee_cost > 0.0

    def test_ruin_probability_positive(self):
        """With a high withdrawal rate relative to growth, ruin should occur."""
        res = gmwb(INV, withdrawal_rate=0.10, fee_rate=0.015, vol=VOL, risk_free_rate=R, T=T, n_paths=10_000, seed=42)
        assert res.ruin_probability > 0.0

    def test_ruin_probability_in_unit_interval(self):
        res = gmwb(INV, withdrawal_rate=0.07, fee_rate=0.015, vol=VOL, risk_free_rate=R, T=T, n_paths=5_000, seed=0)
        assert 0.0 <= res.ruin_probability <= 1.0

    def test_expected_withdrawals_positive(self):
        res = gmwb(INV, withdrawal_rate=0.07, fee_rate=0.015, vol=VOL, risk_free_rate=R, T=T, n_paths=5_000, seed=0)
        assert res.expected_withdrawals > 0.0


# ---- Property Total Return Swap ---------------------------------------------

class TestPropertyTRS:
    def test_pv_positive_when_floating_exceeds_fixed(self):
        res = property_total_return_swap(
            notional=1_000_000.0, fixed_rate=0.02,
            expected_appreciation=0.04, rental_yield=0.03,
            risk_free_rate=0.03, T=5.0,
        )
        assert res.pv > 0.0

    def test_pv_negative_when_fixed_high(self):
        res = property_total_return_swap(
            notional=1_000_000.0, fixed_rate=0.15,
            expected_appreciation=0.02, rental_yield=0.03,
            risk_free_rate=0.03, T=5.0,
        )
        assert res.pv < 0.0

    def test_pv_changes_sign_with_fixed_rate(self):
        low = property_total_return_swap(1_000_000.0, 0.02, 0.04, 0.03, 0.03, 5.0)
        high = property_total_return_swap(1_000_000.0, 0.12, 0.04, 0.03, 0.03, 5.0)
        assert low.pv * high.pv < 0.0


# ---- Property Option --------------------------------------------------------

class TestPropertyOption:
    def test_call_positive(self):
        res = property_option(100.0, 100.0, vol=0.15, T=1.0, risk_free_rate=0.03, rental_yield=0.04, option_type="call")
        assert res.price > 0.0

    def test_put_positive(self):
        res = property_option(100.0, 100.0, vol=0.15, T=1.0, risk_free_rate=0.03, rental_yield=0.04, option_type="put")
        assert res.price > 0.0

    def test_put_call_parity_approximate(self):
        """C - P ≈ disc_factor * (F - K); check within 5% for ATM."""
        import math
        S, K, vol, t, r, y = 100.0, 100.0, 0.15, 1.0, 0.03, 0.04
        call = property_option(S, K, vol, t, r, y, "call", illiquidity_premium=0.01)
        put  = property_option(S, K, vol, t, r, y, "put",  illiquidity_premium=0.01)
        F = S * math.exp((r + 0.01 - y) * t)
        parity_rhs = math.exp(-r * t) * (F - K)
        assert (call.price - put.price) == pytest.approx(parity_rhs, rel=0.05)

    def test_higher_vol_higher_option_price(self):
        low = property_option(100.0, 100.0, 0.10, 1.0, 0.03, 0.04, "call")
        high = property_option(100.0, 100.0, 0.30, 1.0, 0.03, 0.04, "call")
        assert high.price > low.price


# ---- Housing Affordability --------------------------------------------------

class TestHousingAffordability:
    def test_payment_to_income_positive(self):
        res = housing_affordability(400_000.0, 80_000.0, mortgage_rate=0.065)
        assert res.payment_to_income > 0.0

    def test_higher_rate_higher_ratio(self):
        low = housing_affordability(400_000.0, 80_000.0, mortgage_rate=0.03)
        high = housing_affordability(400_000.0, 80_000.0, mortgage_rate=0.08)
        assert high.payment_to_income > low.payment_to_income

    def test_price_to_income_positive(self):
        res = housing_affordability(400_000.0, 80_000.0, mortgage_rate=0.065)
        assert res.price_to_income > 0.0

    def test_max_affordable_price_positive(self):
        res = housing_affordability(400_000.0, 80_000.0, mortgage_rate=0.065)
        assert res.max_affordable_price > 0.0

    def test_lower_price_improves_affordability(self):
        cheap = housing_affordability(200_000.0, 80_000.0, mortgage_rate=0.065)
        expensive = housing_affordability(600_000.0, 80_000.0, mortgage_rate=0.065)
        assert cheap.payment_to_income < expensive.payment_to_income
