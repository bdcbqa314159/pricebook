"""Tests for Black-76 and Bachelier models."""

import math
import pytest

from pricebook.black76 import (
    black76_price, bachelier_price, OptionType, _norm_cdf,
    black76_delta, black76_gamma, black76_vega, black76_theta,
)


class TestBlack76:

    def test_call_positive(self):
        price = black76_price(100.0, 100.0, 0.20, 1.0, 0.95)
        assert price > 0

    def test_put_positive(self):
        price = black76_price(100.0, 100.0, 0.20, 1.0, 0.95, OptionType.PUT)
        assert price > 0

    def test_put_call_parity(self):
        """C - P = df * (F - K)."""
        F, K, vol, T, df = 100.0, 95.0, 0.25, 0.5, 0.98
        call = black76_price(F, K, vol, T, df, OptionType.CALL)
        put = black76_price(F, K, vol, T, df, OptionType.PUT)
        assert call - put == pytest.approx(df * (F - K), rel=1e-10)

    def test_deep_itm_call_near_intrinsic(self):
        """Deep ITM call ≈ df * (F - K)."""
        price = black76_price(150.0, 50.0, 0.20, 0.01, 0.99)
        assert price == pytest.approx(0.99 * 100.0, rel=1e-3)

    def test_deep_otm_call_near_zero(self):
        price = black76_price(50.0, 150.0, 0.20, 0.01, 0.99)
        assert price < 0.01

    def test_higher_vol_higher_price(self):
        low = black76_price(100.0, 100.0, 0.10, 1.0, 0.95)
        high = black76_price(100.0, 100.0, 0.40, 1.0, 0.95)
        assert high > low

    def test_longer_expiry_higher_price(self):
        short = black76_price(100.0, 100.0, 0.20, 0.25, 0.95)
        long = black76_price(100.0, 100.0, 0.20, 2.0, 0.95)
        assert long > short

    def test_atm_call_approx_formula(self):
        """ATM call ≈ df * F * vol * sqrt(T) * 0.4 (rough approximation)."""
        F, vol, T, df = 100.0, 0.20, 1.0, 1.0
        price = black76_price(F, F, vol, T, df)
        approx = df * F * vol * math.sqrt(T) * 0.3989  # N'(0) = 1/sqrt(2pi)
        assert price == pytest.approx(approx, rel=0.05)

    def test_zero_vol_returns_intrinsic(self):
        call = black76_price(110.0, 100.0, 0.0, 1.0, 0.95)
        assert call == pytest.approx(0.95 * 10.0, rel=1e-10)
        put = black76_price(90.0, 100.0, 0.0, 1.0, 0.95, OptionType.PUT)
        assert put == pytest.approx(0.95 * 10.0, rel=1e-10)

    def test_expired_returns_intrinsic(self):
        call = black76_price(110.0, 100.0, 0.20, 0.0, 0.95)
        assert call == pytest.approx(0.95 * 10.0, rel=1e-10)


class TestBachelier:

    def test_call_positive(self):
        price = bachelier_price(100.0, 100.0, 20.0, 1.0, 0.95)
        assert price > 0

    def test_put_call_parity(self):
        """C - P = df * (F - K) for Bachelier too."""
        F, K, vol_n, T, df = 100.0, 95.0, 15.0, 0.5, 0.98
        call = bachelier_price(F, K, vol_n, T, df, OptionType.CALL)
        put = bachelier_price(F, K, vol_n, T, df, OptionType.PUT)
        assert call - put == pytest.approx(df * (F - K), rel=1e-10)

    def test_handles_negative_forward(self):
        """Bachelier works with negative forwards (rates)."""
        price = bachelier_price(-0.005, -0.003, 0.005, 1.0, 1.0, OptionType.PUT)
        assert price > 0

    def test_higher_vol_higher_price(self):
        low = bachelier_price(100.0, 100.0, 10.0, 1.0, 0.95)
        high = bachelier_price(100.0, 100.0, 30.0, 1.0, 0.95)
        assert high > low

    def test_atm_call_approx(self):
        """ATM Bachelier call ≈ df * sigma_n * sqrt(T) / sqrt(2*pi)."""
        vol_n, T, df = 20.0, 1.0, 1.0
        price = bachelier_price(100.0, 100.0, vol_n, T, df)
        approx = df * vol_n * math.sqrt(T) / math.sqrt(2 * math.pi)
        assert price == pytest.approx(approx, rel=1e-6)

    def test_zero_vol_returns_intrinsic(self):
        call = bachelier_price(110.0, 100.0, 0.0, 1.0, 0.95)
        assert call == pytest.approx(0.95 * 10.0, rel=1e-10)

    def test_expired_returns_intrinsic(self):
        put = bachelier_price(90.0, 100.0, 20.0, 0.0, 0.95, OptionType.PUT)
        assert put == pytest.approx(0.95 * 10.0, rel=1e-10)


class TestGreeks:
    """Analytical Greeks vs bump-and-reprice."""

    F, K, vol, T, df = 100.0, 100.0, 0.20, 1.0, 0.95

    def test_delta_call_atm_near_half(self):
        d = black76_delta(self.F, self.K, self.vol, self.T, self.df, OptionType.CALL)
        assert 0.4 * self.df < d < 0.6 * self.df

    def test_delta_put_call_relation(self):
        """delta_call - delta_put = df."""
        dc = black76_delta(self.F, self.K, self.vol, self.T, self.df, OptionType.CALL)
        dp = black76_delta(self.F, self.K, self.vol, self.T, self.df, OptionType.PUT)
        assert dc - dp == pytest.approx(self.df, rel=1e-10)

    def test_delta_matches_bump(self):
        bump = 0.01
        p_up = black76_price(self.F + bump, self.K, self.vol, self.T, self.df)
        p_dn = black76_price(self.F - bump, self.K, self.vol, self.T, self.df)
        numerical = (p_up - p_dn) / (2 * bump)
        analytical = black76_delta(self.F, self.K, self.vol, self.T, self.df)
        assert analytical == pytest.approx(numerical, rel=1e-4)

    def test_gamma_positive(self):
        g = black76_gamma(self.F, self.K, self.vol, self.T, self.df)
        assert g > 0

    def test_gamma_matches_bump(self):
        bump = 0.01
        d_up = black76_delta(self.F + bump, self.K, self.vol, self.T, self.df)
        d_dn = black76_delta(self.F - bump, self.K, self.vol, self.T, self.df)
        numerical = (d_up - d_dn) / (2 * bump)
        analytical = black76_gamma(self.F, self.K, self.vol, self.T, self.df)
        assert analytical == pytest.approx(numerical, rel=1e-3)

    def test_vega_positive(self):
        v = black76_vega(self.F, self.K, self.vol, self.T, self.df)
        assert v > 0

    def test_vega_matches_bump(self):
        bump = 0.0001
        p_up = black76_price(self.F, self.K, self.vol + bump, self.T, self.df)
        p_dn = black76_price(self.F, self.K, self.vol - bump, self.T, self.df)
        numerical = (p_up - p_dn) / (2 * bump)
        analytical = black76_vega(self.F, self.K, self.vol, self.T, self.df)
        assert analytical == pytest.approx(numerical, rel=1e-4)

    def test_vega_maximised_atm(self):
        """Vega is highest at ATM."""
        v_atm = black76_vega(100.0, 100.0, 0.20, 1.0, 0.95)
        v_otm = black76_vega(100.0, 130.0, 0.20, 1.0, 0.95)
        assert v_atm > v_otm

    def test_theta_negative_for_long(self):
        t = black76_theta(self.F, self.K, self.vol, self.T, self.df)
        assert t < 0
