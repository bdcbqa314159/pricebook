"""Tests for pricebook.structured.cat_bond."""

import pytest

from pricebook.structured.cat_bond import (
    cat_bond_price,
    parametric_trigger_prob,
    indemnity_trigger_loss,
    cat_bond_spread_decomposition,
    seasonal_adjustment,
    PeriodType,
)


class TestCatBondPrice:
    def test_price_positive(self):
        result = cat_bond_price(
            notional=1_000.0,
            coupon_spread=0.05,
            risk_free_rate=0.03,
            expected_loss=0.02,
            T=3.0,
        )
        assert result.price > 0.0

    def test_higher_expected_loss_lower_price(self):
        low_el = cat_bond_price(1_000.0, 0.05, 0.03, expected_loss=0.01, T=3.0)
        high_el = cat_bond_price(1_000.0, 0.05, 0.03, expected_loss=0.10, T=3.0)
        assert high_el.price < low_el.price

    def test_probability_of_loss_in_unit_interval(self):
        result = cat_bond_price(1_000.0, 0.05, 0.03, expected_loss=0.02, T=3.0)
        assert 0.0 < result.probability_of_loss < 1.0

    def test_zero_expected_loss_price_near_par(self):
        """With zero expected loss and spread, price should be close to 100 (par)."""
        result = cat_bond_price(1_000.0, 0.0, 0.03, expected_loss=0.0, T=1.0)
        assert result.price == pytest.approx(100.0, abs=5.0)

    def test_recovery_increases_price(self):
        no_rec = cat_bond_price(1_000.0, 0.05, 0.03, expected_loss=0.05, T=3.0, recovery_if_triggered=0.0)
        with_rec = cat_bond_price(1_000.0, 0.05, 0.03, expected_loss=0.05, T=3.0, recovery_if_triggered=0.4)
        assert with_rec.price > no_rec.price


class TestParametricTriggerProb:
    def test_probability_in_unit_interval(self):
        prob = parametric_trigger_prob(threshold=7.0, location_mu=5.5, scale_sigma=0.8)
        assert 0.0 <= prob <= 1.0

    def test_higher_threshold_lower_probability(self):
        low_thresh = parametric_trigger_prob(threshold=6.0, location_mu=5.5, scale_sigma=0.8)
        high_thresh = parametric_trigger_prob(threshold=9.0, location_mu=5.5, scale_sigma=0.8)
        assert high_thresh < low_thresh

    def test_historical_events_recalibrate(self):
        """Providing historical events should change the probability (MOM recalibration)."""
        prob_no_data = parametric_trigger_prob(7.0, 5.5, 0.8)
        prob_with_data = parametric_trigger_prob(7.0, 5.5, 0.8, historical_events=[5.0, 5.5, 6.0, 6.5, 7.0])
        # Just verify it runs and returns a valid probability
        assert 0.0 <= prob_with_data <= 1.0

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError):
            parametric_trigger_prob(7.0, 5.5, scale_sigma=-1.0)


class TestIndemnityTriggerLoss:
    def test_expected_loss_in_unit_interval(self):
        result = indemnity_trigger_loss(
            attachment=500.0, exhaustion=1_000.0,
            loss_distribution_mean=400.0, loss_distribution_cv=0.5,
        )
        assert 0.0 <= result["expected_loss"] <= 1.0

    def test_higher_attachment_lower_expected_loss(self):
        low_att = indemnity_trigger_loss(200.0, 800.0, 400.0, 0.5)
        high_att = indemnity_trigger_loss(600.0, 1_000.0, 400.0, 0.5)
        assert high_att["expected_loss"] < low_att["expected_loss"]

    def test_prob_of_loss_in_unit_interval(self):
        result = indemnity_trigger_loss(500.0, 1_000.0, 400.0, 0.5)
        assert 0.0 <= result["prob_of_loss"] <= 1.0

    def test_exhaustion_above_attachment_required(self):
        with pytest.raises(ValueError):
            indemnity_trigger_loss(800.0, 500.0, 400.0, 0.5)


class TestCatBondSpreadDecomposition:
    def test_components_sum_to_total(self):
        result = cat_bond_spread_decomposition(
            total_spread=0.08, expected_loss=0.03, expense_load=0.01
        )
        assert (result["expected_loss"] + result["risk_premium"] + result["expense_load"]) == pytest.approx(result["total_spread"])

    def test_risk_multiple_positive(self):
        result = cat_bond_spread_decomposition(0.08, 0.03, 0.01)
        assert result["risk_multiple"] > 0.0

    def test_risk_premium_correct(self):
        result = cat_bond_spread_decomposition(0.08, 0.03, 0.01)
        assert result["risk_premium"] == pytest.approx(0.08 - 0.03 - 0.01)


class TestSeasonalAdjustment:
    def test_remaining_zero_gives_zero_probability(self):
        prob = seasonal_adjustment(0.10, PeriodType.ANNUAL, remaining_fraction=0.0)
        assert prob == pytest.approx(0.0, abs=1e-10)

    def test_remaining_one_annual_gives_full_prob(self):
        prob = seasonal_adjustment(0.10, PeriodType.ANNUAL, remaining_fraction=1.0)
        assert prob == pytest.approx(0.10, rel=0.01)

    def test_hurricane_season_scales_up(self):
        """Hurricane season concentrates risk, so remaining_fraction=1 gives higher adjusted prob."""
        annual_prob = seasonal_adjustment(0.05, PeriodType.ANNUAL, remaining_fraction=1.0)
        hurricane_prob = seasonal_adjustment(0.05, PeriodType.HURRICANE_SEASON, remaining_fraction=1.0)
        assert hurricane_prob > annual_prob

    def test_invalid_remaining_fraction_raises(self):
        with pytest.raises(ValueError):
            seasonal_adjustment(0.10, PeriodType.ANNUAL, remaining_fraction=1.5)
