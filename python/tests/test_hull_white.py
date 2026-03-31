"""Tests for Hull-White model and rate tree."""

import pytest
import math
from datetime import date

from pricebook.hull_white import HullWhite
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


@pytest.fixture
def hw():
    curve = make_flat_curve(REF, 0.05)
    return HullWhite(a=0.1, sigma=0.01, curve=curve)


class TestConstruction:
    def test_basic(self, hw):
        assert hw.a == 0.1
        assert hw.sigma == 0.01

    def test_negative_a_raises(self):
        with pytest.raises(ValueError, match="positive"):
            HullWhite(a=-0.1, sigma=0.01, curve=make_flat_curve(REF, 0.05))

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="positive"):
            HullWhite(a=0.1, sigma=-0.01, curve=make_flat_curve(REF, 0.05))


class TestAnalytical:
    def test_B_at_zero(self, hw):
        assert hw.B(0, 0) == 0.0

    def test_B_positive(self, hw):
        assert hw.B(0, 5) > 0

    def test_B_increases_with_maturity(self, hw):
        assert hw.B(0, 10) > hw.B(0, 5)

    def test_zcb_price_at_zero(self, hw):
        """P(0, 0) = 1."""
        assert hw.zcb_price(0, 0, 0.05) == pytest.approx(1.0)

    def test_zcb_price_positive(self, hw):
        p = hw.zcb_price(0, 5, 0.05)
        assert 0 < p < 1

    def test_zcb_decreasing_with_maturity(self, hw):
        p5 = hw.zcb_price(0, 5, 0.05)
        p10 = hw.zcb_price(0, 10, 0.05)
        assert p10 < p5

    def test_zcb_matches_curve_at_r0(self, hw):
        """At r = forward_rate(0), analytical ZCB ≈ market discount factor."""
        r0 = hw._forward_rate(0.0)
        d5 = date.fromordinal(REF.toordinal() + int(5 * 365))
        market_df = hw.curve.df(d5)
        model_df = hw.zcb_price(0, 5, r0)
        assert model_df == pytest.approx(market_df, rel=0.02)


class TestRateTree:
    def test_tree_zcb_positive(self, hw):
        p = hw.tree_zcb(T=5.0, n_steps=50)
        assert p > 0

    def test_tree_zcb_less_than_one(self, hw):
        p = hw.tree_zcb(T=5.0, n_steps=50)
        assert p < 1

    def test_tree_matches_curve(self, hw):
        """Tree ZCB price ≈ market discount factor (calibration test)."""
        d5 = date.fromordinal(REF.toordinal() + int(5 * 365))
        market_df = hw.curve.df(d5)
        tree_df = hw.tree_zcb(T=5.0, n_steps=100)
        assert tree_df == pytest.approx(market_df, rel=0.02)

    def test_tree_matches_analytical(self, hw):
        """Tree price ≈ analytical price."""
        r0 = hw._forward_rate(0.0)
        analytical = hw.zcb_price(0, 5, r0)
        tree = hw.tree_zcb(T=5.0, n_steps=100)
        assert tree == pytest.approx(analytical, rel=0.02)

    def test_longer_bond_lower_price(self, hw):
        p5 = hw.tree_zcb(T=5.0, n_steps=50)
        p10 = hw.tree_zcb(T=10.0, n_steps=100)
        assert p10 < p5


class TestSwaption:
    def test_payer_swaption_positive(self, hw):
        pv = hw.tree_european_swaption(
            expiry_T=1.0, swap_end_T=6.0, strike=0.05, n_steps=50,
        )
        assert pv > 0

    def test_receiver_swaption_positive(self, hw):
        pv = hw.tree_european_swaption(
            expiry_T=1.0, swap_end_T=6.0, strike=0.05, n_steps=50,
            is_payer=False,
        )
        assert pv > 0

    def test_higher_vol_higher_swaption(self):
        curve = make_flat_curve(REF, 0.05)
        hw_low = HullWhite(a=0.1, sigma=0.005, curve=curve)
        hw_high = HullWhite(a=0.1, sigma=0.02, curve=curve)
        pv_low = hw_low.tree_european_swaption(1.0, 6.0, 0.05, n_steps=50)
        pv_high = hw_high.tree_european_swaption(1.0, 6.0, 0.05, n_steps=50)
        assert pv_high > pv_low
