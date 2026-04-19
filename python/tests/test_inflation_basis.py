"""Tests for inflation basis."""
import pytest
from pricebook.inflation_basis import zc_yoy_basis, cross_market_inflation_basis, inflation_basis_trade

class TestZCYoYBasis:
    def test_basic(self):
        r = zc_yoy_basis(0.025, 0.024, 5.0)
        assert r.basis_bps == pytest.approx(10, abs=1)  # 0.001 = 10 bps
    def test_zero_basis(self):
        r = zc_yoy_basis(0.025, 0.025, 5.0)
        assert r.basis_bps == pytest.approx(0)

class TestCrossMarket:
    def test_basic(self):
        r = cross_market_inflation_basis("CPI_USD", 0.025, "HICP_EUR", 0.020)
        assert r.basis_bps == pytest.approx(50)
    def test_z_score(self):
        r = cross_market_inflation_basis("CPI", 0.025, "HICP", 0.020, [40, 45, 50, 55, 60])
        assert isinstance(r.z_score, float)

class TestBasisTrade:
    def test_expected_pnl(self):
        r = inflation_basis_trade(0.025, 0.024, 1e6, 50, target_basis_bps=5)
        assert r.basis_bps == pytest.approx(10, abs=1)
        assert r.expected_pnl == pytest.approx((10 - 5) * 50)
    def test_zero_target(self):
        r = inflation_basis_trade(0.025, 0.025, 1e6, 50)
        assert r.expected_pnl == pytest.approx(0)
