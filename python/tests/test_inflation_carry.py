"""Tests for inflation carry."""
import pytest
from pricebook.inflation_carry import real_yield_rolldown, linker_carry_decomposition, inflation_carry_vs_vol

class TestRealYieldRolldown:
    def test_positive_rolldown(self):
        """Normal upward-sloping real curve → positive roll-down."""
        r = real_yield_rolldown([2, 5, 10], [0.5, 1.0, 1.5], position_tenor=5, horizon=1.0)
        assert r.rolldown_bps > 0
    def test_flat_curve_zero(self):
        r = real_yield_rolldown([2, 5, 10], [1.0, 1.0, 1.0], 5, 1.0)
        assert r.rolldown_bps == pytest.approx(0)

class TestLinkerCarry:
    def test_positive_carry(self):
        r = linker_carry_decomposition(real_yield=0.015, breakeven=0.025, repo_rate=0.01, duration=7)
        assert r.total_carry_bps > 0
    def test_negative_carry(self):
        """Negative real yield + high financing → negative carry."""
        r = linker_carry_decomposition(-0.01, 0.02, 0.05, 7)
        assert r.total_carry_bps < 0
    def test_decomposition_sums(self):
        r = linker_carry_decomposition(0.015, 0.025, 0.01, 7)
        assert r.total_carry_bps == pytest.approx(r.real_yield_carry + r.breakeven_carry - r.financing_cost)

class TestCarryVsVol:
    def test_strong_buy(self):
        r = inflation_carry_vs_vol(carry_bps=20, breakeven_vol_bps=10)
        assert r.sharpe > 1
        assert r.signal == "strong_buy"
    def test_sell(self):
        r = inflation_carry_vs_vol(-10, 5)
        assert r.signal == "sell"
    def test_neutral(self):
        r = inflation_carry_vs_vol(1, 10)
        assert r.signal == "neutral"
