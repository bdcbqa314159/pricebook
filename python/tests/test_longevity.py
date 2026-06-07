"""Tests for pricebook.structured.longevity."""

import pytest
import numpy as np

from pricebook.structured.longevity import (
    q_forward,
    survivor_index,
    lee_carter_forecast,
    value_of_life_annuity,
    mortality_bond_price,
)


class TestQForward:
    def test_pv_changes_sign_with_risk_premium(self):
        """With zero risk_premium, PV = 0; positive risk_premium makes PV > 0."""
        res_zero = q_forward(
            reference_age=65, term_years=10,
            current_mortality_rate=0.02, mortality_improvement_rate=0.015,
            risk_free_rate=0.03, risk_premium=0.0,
        )
        res_pos = q_forward(
            reference_age=65, term_years=10,
            current_mortality_rate=0.02, mortality_improvement_rate=0.015,
            risk_free_rate=0.03, risk_premium=0.05,
        )
        assert res_zero.pv == pytest.approx(0.0, abs=1e-10)
        assert res_pos.pv > 0.0

    def test_fixed_mortality_above_forward(self):
        """With positive risk_premium, fixed rate should exceed forward rate."""
        res = q_forward(65, 10, 0.02, 0.015, 0.03, risk_premium=0.05)
        assert res.fixed_mortality > res.forward_mortality

    def test_notional_at_risk_positive(self):
        res = q_forward(65, 10, 0.02, 0.015, 0.03)
        assert res.notional_at_risk > 0.0

    def test_longer_term_lower_forward_mortality(self):
        """Mortality improvement → longer term yields lower projected mortality."""
        short = q_forward(65, 5, 0.02, 0.02, 0.03)
        long_ = q_forward(65, 20, 0.02, 0.02, 0.03)
        assert long_.forward_mortality < short.forward_mortality

    def test_negative_risk_premium_makes_pv_negative(self):
        res = q_forward(65, 10, 0.02, 0.015, 0.03, risk_premium=-0.05)
        assert res.pv < 0.0


class TestSurvivorIndex:
    def test_starts_at_initial_population(self):
        idx = survivor_index(1000.0, [0.01], improvement_rate=0.01, n_years=10)
        assert idx[0] == pytest.approx(1000.0)

    def test_monotonically_decreasing(self):
        idx = survivor_index(1000.0, [0.02], improvement_rate=0.01, n_years=20)
        diffs = np.diff(idx)
        assert np.all(diffs <= 0.0)

    def test_positive_values(self):
        idx = survivor_index(100.0, [0.01], improvement_rate=0.01, n_years=10)
        assert np.all(idx > 0.0)

    def test_length_is_n_years_plus_one(self):
        idx = survivor_index(100.0, [0.01], improvement_rate=0.01, n_years=15)
        assert len(idx) == 16


class TestLeeCarter:
    def setup_method(self):
        rng = np.random.default_rng(0)
        # Synthetic log-mortality: 5 ages × 20 historical years
        self.log_m = -4.0 + 0.05 * np.arange(5)[:, None] - 0.01 * np.arange(20)[None, :] + rng.normal(0, 0.02, (5, 20))

    def test_output_shape(self):
        forecast = lee_carter_forecast(self.log_m, n_forecast_years=10)
        assert forecast.shape == (5, 10)

    def test_mortality_rates_positive(self):
        forecast = lee_carter_forecast(self.log_m, n_forecast_years=10)
        assert np.all(forecast > 0.0)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            lee_carter_forecast(np.array([1.0, 2.0, 3.0]), n_forecast_years=5)

    def test_more_forecast_years_larger_output(self):
        f10 = lee_carter_forecast(self.log_m, n_forecast_years=10)
        f20 = lee_carter_forecast(self.log_m, n_forecast_years=20)
        assert f20.shape[1] == 20
        assert f10.shape[1] == 10


class TestValueOfLifeAnnuity:
    def test_positive_pv(self):
        pv = value_of_life_annuity(
            annual_payment=10_000.0, age=65,
            mortality_rates=[0.01 + i * 0.002 for i in range(35)],
            improvement_rate=0.01, risk_free_rate=0.03,
        )
        assert pv > 0.0

    def test_higher_mortality_lower_pv(self):
        low_mort = value_of_life_annuity(10_000.0, 65, [0.005] * 35, 0.01, 0.03)
        high_mort = value_of_life_annuity(10_000.0, 65, [0.05] * 35, 0.01, 0.03)
        assert high_mort < low_mort

    def test_higher_discount_lower_pv(self):
        low_r = value_of_life_annuity(10_000.0, 65, [0.01] * 35, 0.01, 0.02)
        high_r = value_of_life_annuity(10_000.0, 65, [0.01] * 35, 0.01, 0.08)
        assert high_r < low_r

    def test_payment_proportional(self):
        pv1 = value_of_life_annuity(10_000.0, 65, [0.01] * 35, 0.01, 0.03)
        pv2 = value_of_life_annuity(20_000.0, 65, [0.01] * 35, 0.01, 0.03)
        assert pv2 == pytest.approx(2.0 * pv1, rel=1e-9)


class TestMortalityBondPrice:
    def test_price_positive(self):
        result = mortality_bond_price(
            notional=1_000.0, coupon=0.05, risk_free_rate=0.03,
            attachment=1.2, exhaustion=1.5,
            expected_mortality=1.0, mortality_vol=0.15, T=5.0,
        )
        assert result["price"] > 0.0

    def test_price_has_expected_keys(self):
        result = mortality_bond_price(
            1_000.0, 0.05, 0.03, 1.2, 1.5, 1.0, 0.15, 5.0,
        )
        assert "price" in result
        assert "expected_loss" in result
        assert "spread" in result

    def test_expected_loss_in_unit_interval(self):
        result = mortality_bond_price(1_000.0, 0.05, 0.03, 1.2, 1.5, 1.0, 0.15, 5.0)
        assert 0.0 <= result["expected_loss"] <= 1.0
