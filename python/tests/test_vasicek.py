"""Tests for Vasicek and G2++ models."""

import pytest
import math
import numpy as np

from pricebook.vasicek import Vasicek, G2PlusPlus
from tests.conftest import make_flat_curve
from datetime import date


REF = date(2024, 1, 15)


class TestVasicek:
    def test_zcb_positive(self):
        v = Vasicek(a=0.5, b=0.05, sigma=0.01)
        assert 0 < v.zcb_price(0.05, 5.0) < 1

    def test_zcb_at_zero(self):
        v = Vasicek(a=0.5, b=0.05, sigma=0.01)
        assert v.zcb_price(0.05, 0.0) == pytest.approx(1.0)

    def test_zcb_decreasing(self):
        v = Vasicek(a=0.5, b=0.05, sigma=0.01)
        assert v.zcb_price(0.05, 10.0) < v.zcb_price(0.05, 5.0)

    def test_higher_rate_lower_price(self):
        v = Vasicek(a=0.5, b=0.05, sigma=0.01)
        assert v.zcb_price(0.08, 5.0) < v.zcb_price(0.03, 5.0)

    def test_mean(self):
        v = Vasicek(a=0.5, b=0.05, sigma=0.01)
        assert v.mean(0.10, 10.0) == pytest.approx(0.05, rel=0.01)

    def test_variance(self):
        v = Vasicek(a=0.5, b=0.05, sigma=0.01)
        var = v.variance(10.0)
        assert var > 0
        # Stationary: sigma^2/(2a) = 0.0001
        assert var == pytest.approx(0.0001, rel=0.01)

    def test_simulate_mean(self):
        v = Vasicek(a=0.5, b=0.05, sigma=0.01)
        paths = v.simulate(r0=0.10, T=10.0, n_steps=100, n_paths=50_000)
        assert paths[:, -1].mean() == pytest.approx(v.mean(0.10, 10.0), rel=0.03)

    def test_yield_curve(self):
        v = Vasicek(a=0.5, b=0.05, sigma=0.01)
        yields = v.yield_curve(0.05, [1, 2, 5, 10])
        assert all(y > 0 for y in yields)

    def test_caplet_positive(self):
        v = Vasicek(a=0.5, b=0.05, sigma=0.01)
        cap = v.caplet_price(r=0.05, strike=0.05, T_option=1.0, T_pay=1.25)
        assert cap > 0

    def test_caplet_zero_vol(self):
        v = Vasicek(a=0.5, b=0.05, sigma=1e-10)
        cap = v.caplet_price(r=0.05, strike=0.05, T_option=1.0, T_pay=1.25)
        assert cap == pytest.approx(0.0, abs=1e-6)


class TestG2PlusPlus:
    @pytest.fixture
    def g2(self):
        curve = make_flat_curve(REF, 0.05)
        return G2PlusPlus(a=0.5, b=0.1, sigma1=0.01, sigma2=0.008,
                          rho=-0.5, curve=curve)

    def test_zcb_at_origin(self, g2):
        """P(x=0, y=0, T) = P_market(T) * exp(0.5*V(T))."""
        p = g2.zcb_price(0.0, 0.0, 5.0)
        assert p > 0

    def test_zcb_matches_market_at_origin(self, g2):
        """At x=0, y=0 the model should be close to market (phi calibrated)."""
        d5 = date.fromordinal(REF.toordinal() + int(5 * 365))
        mkt = g2.curve.df(d5)
        model = g2.zcb_price(0.0, 0.0, 5.0)
        # Not exactly equal because of V(T) term, but close
        assert model == pytest.approx(mkt, rel=0.05)

    def test_sigma2_zero_one_factor(self, g2):
        """sigma2=0 → G2++ collapses to one-factor HW."""
        curve = make_flat_curve(REF, 0.05)
        g2_1f = G2PlusPlus(a=0.5, b=0.1, sigma1=0.01, sigma2=0.0001,
                            rho=0.0, curve=curve)
        # y has negligible effect
        p1 = g2_1f.zcb_price(0.01, 0.0, 5.0)
        p2 = g2_1f.zcb_price(0.01, 0.001, 5.0)
        assert p1 == pytest.approx(p2, rel=0.01)

    def test_higher_x_lower_price(self, g2):
        """Higher x (= higher rate) → lower bond price."""
        assert g2.zcb_price(0.02, 0.0, 5.0) < g2.zcb_price(0.0, 0.0, 5.0)

    def test_simulate_shape(self, g2):
        x, y = g2.simulate(T=5.0, n_steps=50, n_paths=100)
        assert x.shape == (100, 51)
        assert y.shape == (100, 51)

    def test_simulate_mean_reversion(self, g2):
        """x and y should revert to 0."""
        x, y = g2.simulate(T=20.0, n_steps=200, n_paths=50_000)
        assert x[:, -1].mean() == pytest.approx(0.0, abs=0.002)
        assert y[:, -1].mean() == pytest.approx(0.0, abs=0.002)

    def test_mc_discount_factor(self, g2):
        """MC discount factor ≈ market discount factor."""
        df_mc = g2.discount_factor_mc(T=5.0, n_steps=50, n_paths=50_000)
        d5 = date.fromordinal(REF.toordinal() + int(5 * 365))
        mkt_df = g2.curve.df(d5)
        assert df_mc.mean() == pytest.approx(mkt_df, rel=0.05)

    def test_V_positive(self, g2):
        assert g2._V(5.0) > 0

    def test_V_zero_at_zero(self, g2):
        assert g2._V(0.0) == pytest.approx(0.0, abs=1e-10)
