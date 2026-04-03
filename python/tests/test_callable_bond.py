"""Tests for callable and puttable bonds."""

import pytest
import math
from datetime import date

from pricebook.callable_bond import (
    callable_bond_price,
    puttable_bond_price,
    oas,
    _straight_bond_hw,
)
from pricebook.hull_white import HullWhite
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


@pytest.fixture
def hw():
    curve = make_flat_curve(REF, 0.05)
    return HullWhite(a=0.1, sigma=0.01, curve=curve)


@pytest.fixture
def hw_low_vol():
    curve = make_flat_curve(REF, 0.05)
    return HullWhite(a=0.1, sigma=0.001, curve=curve)


class TestCallableBond:
    def test_callable_leq_straight(self, hw):
        """Callable ≤ straight (call option value to issuer)."""
        straight = _straight_bond_hw(hw, 0.05, 10.0)
        call = callable_bond_price(hw, 0.05, 10.0)
        assert call <= straight + 0.5

    def test_callable_positive(self, hw):
        price = callable_bond_price(hw, 0.05, 10.0)
        assert price > 0

    def test_low_vol_approaches_straight(self, hw_low_vol):
        """Near-zero vol → callable ≈ straight."""
        straight = _straight_bond_hw(hw_low_vol, 0.05, 5.0)
        call = callable_bond_price(hw_low_vol, 0.05, 5.0)
        assert call == pytest.approx(straight, rel=0.05)

    def test_higher_coupon_higher_price(self, hw):
        p_low = callable_bond_price(hw, 0.03, 10.0)
        p_high = callable_bond_price(hw, 0.07, 10.0)
        assert p_high > p_low

    def test_callable_capped_at_call_price(self, hw):
        """Callable bond price ≤ call price (approximately, at any node)."""
        price = callable_bond_price(hw, 0.10, 5.0, call_price=100.0)
        # High coupon → bond wants to trade above par → gets called
        assert price <= 105  # some flexibility for accrued


class TestPuttableBond:
    def test_puttable_geq_straight(self, hw):
        """Puttable ≥ straight (put option value to investor)."""
        straight = _straight_bond_hw(hw, 0.05, 10.0)
        put = puttable_bond_price(hw, 0.05, 10.0)
        assert put >= straight - 0.5

    def test_puttable_positive(self, hw):
        price = puttable_bond_price(hw, 0.05, 10.0)
        assert price > 0

    def test_low_vol_approaches_straight(self, hw_low_vol):
        straight = _straight_bond_hw(hw_low_vol, 0.05, 5.0)
        put = puttable_bond_price(hw_low_vol, 0.05, 5.0)
        assert put == pytest.approx(straight, rel=0.05)

    def test_puttable_floored_at_put_price(self, hw):
        """Puttable ≥ put price (approximately)."""
        price = puttable_bond_price(hw, 0.01, 10.0, put_price=100.0)
        # Low coupon → bond trades below par → investor puts
        assert price >= 95  # some flexibility


class TestOAS:
    def test_oas_round_trip(self, hw):
        """Price → OAS → reprice recovers market price."""
        market = callable_bond_price(hw, 0.05, 5.0)
        spread = oas(hw, market, 0.05, 5.0, is_callable=True)
        assert spread == pytest.approx(0.0, abs=0.005)

    def test_oas_positive_below_model(self, hw):
        """Market price below model → positive OAS."""
        model_price = callable_bond_price(hw, 0.05, 5.0)
        spread = oas(hw, model_price - 2.0, 0.05, 5.0, is_callable=True)
        assert spread > 0

    def test_puttable_oas(self, hw):
        market = puttable_bond_price(hw, 0.05, 5.0)
        spread = oas(hw, market, 0.05, 5.0, is_callable=False)
        assert spread == pytest.approx(0.0, abs=0.005)
