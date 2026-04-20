"""Tests for structural credit."""
import math, pytest
from pricebook.structural_credit import merton_equity_credit, kmv_distance_to_default, black_cox_first_passage, implied_credit_from_equity

class TestMerton:
    def test_basic(self):
        r = merton_equity_credit(150, 100, 0.30, 0.05, 5.0)
        assert r.equity_value > 0; assert r.debt_value > 0
        assert r.default_probability >= 0
    def test_high_leverage_high_spread(self):
        low = merton_equity_credit(200, 100, 0.30, 0.05, 5.0)
        high = merton_equity_credit(120, 100, 0.30, 0.05, 5.0)
        assert high.credit_spread_bps > low.credit_spread_bps
    def test_asset_decomposition(self):
        r = merton_equity_credit(150, 100, 0.30, 0.05, 5.0)
        assert r.equity_value + r.debt_value == pytest.approx(150, rel=0.01)

class TestKMV:
    def test_basic(self):
        r = kmv_distance_to_default(50, 0.30, 40, 60, 0.05)
        assert r.distance_to_default > 0
        assert 0 <= r.default_probability <= 1
    def test_more_debt_higher_default_point(self):
        """More debt → higher default point."""
        low = kmv_distance_to_default(100, 0.30, 20, 30, 0.05)
        high = kmv_distance_to_default(100, 0.30, 80, 120, 0.05)
        assert high.default_point > low.default_point

class TestBlackCox:
    def test_basic(self):
        r = black_cox_first_passage(150, 80, 0.25, 0.05, 5.0)
        assert 0 <= r.default_probability <= 1
    def test_higher_barrier_higher_pd(self):
        low = black_cox_first_passage(150, 60, 0.25, 0.05, 5.0)
        high = black_cox_first_passage(150, 120, 0.25, 0.05, 5.0)
        assert high.default_probability > low.default_probability
    def test_already_defaulted(self):
        r = black_cox_first_passage(50, 80, 0.25, 0.05, 5.0)
        assert r.default_probability == 1.0

class TestImpliedCredit:
    def test_basic(self):
        r = implied_credit_from_equity(100, 0.30, 80, 0.05)
        assert r.implied_spread_bps > 0
        assert r.leverage > 0
