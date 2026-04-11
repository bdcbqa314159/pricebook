"""Tests for FX carry strategies."""

import pytest

from pricebook.fx_carry import (
    CarryRanking,
    CarrySignal,
    NDFCarry,
    carry_adjusted_forward,
    carry_signal,
    carry_volatility_ratio,
    g10_carry_ranking,
    ndf_carry,
)


# ---- Step 1: G10 carry ----

class TestCarrySignal:
    def test_positive_carry_when_rate_diff_positive(self):
        """Step 1 test: positive carry when rate differential is positive."""
        # Forward below spot → base at discount → foreign rate higher
        # Carry trade = long base (collect the higher foreign rate)
        sig = carry_signal("AUD/USD", spot=0.6500, forward=0.6480, days_to_maturity=90)
        # annualised = (0.648 − 0.65) / 0.65 × 365/90 ≈ −0.1247
        # Negative carry on the forward → long_base direction (positive carry trade)
        assert sig.annualised_carry < 0
        assert sig.carry_direction == "long_base"

    def test_negative_carry(self):
        # Forward above spot → base at premium → domestic rate higher
        sig = carry_signal("USD/JPY", spot=148.0, forward=148.5, days_to_maturity=90)
        assert sig.annualised_carry > 0
        assert sig.carry_direction == "short_base"

    def test_forward_points(self):
        sig = carry_signal("EUR/USD", spot=1.085, forward=1.087, days_to_maturity=90)
        assert sig.forward_points == pytest.approx(0.002)

    def test_zero_days(self):
        sig = carry_signal("EUR/USD", 1.085, 1.087, 0)
        assert sig.annualised_carry == 0.0


class TestCarryAdjustedForward:
    def test_break_even(self):
        be = carry_adjusted_forward(1.085, 1.087, 90)
        assert be == pytest.approx(0.002)


class TestG10CarryRanking:
    def test_ranking(self):
        pairs = [
            ("AUD/USD", 0.65, 0.648, 90),  # negative carry → rank 1
            ("EUR/USD", 1.085, 1.087, 90),  # positive → rank last
            ("USD/JPY", 148.0, 147.5, 90),  # slightly negative
        ]
        ranked = g10_carry_ranking(pairs)
        assert len(ranked) == 3
        assert ranked[0].rank == 1
        # Most negative annualised carry is rank 1
        assert ranked[0].annualised_carry <= ranked[1].annualised_carry
        assert ranked[1].annualised_carry <= ranked[2].annualised_carry


# ---- Step 2: EM / NDF carry ----

class TestNDFCarry:
    def test_rate_differential(self):
        """Step 2 test: NDF carry matches rate differential."""
        result = ndf_carry("USD/BRL", spot=5.0, domestic_rate=0.05,
                           foreign_rate=0.12, days=90)
        assert result.rate_differential == pytest.approx(0.07)
        assert result.annualised_carry == pytest.approx(0.07)

    def test_ndf_points(self):
        result = ndf_carry("USD/BRL", spot=5.0, domestic_rate=0.05,
                           foreign_rate=0.12, days=365)
        # NDF = 5 × 1.05 / 1.12 ≈ 4.6875
        assert result.ndf_points == pytest.approx(5 * 1.05 / 1.12 - 5)

    def test_zero_days(self):
        result = ndf_carry("USD/BRL", 5.0, 0.05, 0.12, 0)
        assert result.annualised_carry == 0.0


class TestCarryVolatilityRatio:
    def test_positive_ratio(self):
        ratio = carry_volatility_ratio(0.05, 0.10)
        assert ratio == pytest.approx(0.5)

    def test_high_carry_low_vol(self):
        ratio = carry_volatility_ratio(0.08, 0.10)
        assert ratio == pytest.approx(0.8)

    def test_zero_vol(self):
        assert carry_volatility_ratio(0.05, 0.0) == 0.0

    def test_negative_carry_uses_abs(self):
        ratio = carry_volatility_ratio(-0.05, 0.10)
        assert ratio == pytest.approx(0.5)
