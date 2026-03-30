"""Tests for equity option pricing and Greeks."""

import pytest
import math

from pricebook.equity_option import (
    equity_option_price,
    equity_delta,
    equity_gamma,
    equity_vega,
    equity_theta,
    equity_rho,
)
from pricebook.black76 import OptionType


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
BUMP = 1e-4


class TestPricing:
    def test_call_positive(self):
        p = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert p > 0

    def test_put_positive(self):
        p = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        assert p > 0

    def test_put_call_parity(self):
        """C - P = S*exp(-q*T) - K*exp(-r*T)."""
        q = 0.02
        c = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        p = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT, q)
        expected = SPOT * math.exp(-q * T) - STRIKE * math.exp(-RATE * T)
        assert c - p == pytest.approx(expected, abs=1e-10)

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S*exp(-q*T) - K*exp(-r*T)."""
        c = equity_option_price(SPOT, 50.0, RATE, 0.01, T, OptionType.CALL)
        intrinsic = SPOT - 50.0 * math.exp(-RATE * T)
        assert c == pytest.approx(intrinsic, rel=0.01)

    def test_with_dividend_yield(self):
        """Dividend yield reduces call price."""
        c_no_div = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        c_div = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, 0.03)
        assert c_div < c_no_div


class TestDelta:
    def test_atm_forward_call_delta_near_half(self):
        """ATM-forward (strike = forward) gives delta ≈ 0.5 * exp(-q*T)."""
        fwd_strike = SPOT * math.exp(RATE * T)
        d = equity_delta(SPOT, fwd_strike, RATE, VOL, T, OptionType.CALL)
        assert d == pytest.approx(0.5, abs=0.05)

    def test_call_delta_positive(self):
        d = equity_delta(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert 0 < d < 1

    def test_put_delta_negative(self):
        d = equity_delta(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        assert -1 < d < 0

    def test_call_put_delta_relation(self):
        """Delta_call - Delta_put = exp(-q*T)."""
        q = 0.02
        dc = equity_delta(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        dp = equity_delta(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT, q)
        assert dc - dp == pytest.approx(math.exp(-q * T), abs=1e-10)

    def test_delta_vs_bump(self):
        """Analytical delta matches bump-and-reprice."""
        d = equity_delta(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        p_up = equity_option_price(SPOT + BUMP, STRIKE, RATE, VOL, T, OptionType.CALL)
        p_dn = equity_option_price(SPOT - BUMP, STRIKE, RATE, VOL, T, OptionType.CALL)
        bump_delta = (p_up - p_dn) / (2 * BUMP)
        assert d == pytest.approx(bump_delta, rel=1e-4)


class TestGamma:
    def test_gamma_positive(self):
        g = equity_gamma(SPOT, STRIKE, RATE, VOL, T)
        assert g > 0

    def test_gamma_vs_bump(self):
        g = equity_gamma(SPOT, STRIKE, RATE, VOL, T)
        d_up = equity_delta(SPOT + BUMP, STRIKE, RATE, VOL, T, OptionType.CALL)
        d_dn = equity_delta(SPOT - BUMP, STRIKE, RATE, VOL, T, OptionType.CALL)
        bump_gamma = (d_up - d_dn) / (2 * BUMP)
        assert g == pytest.approx(bump_gamma, rel=1e-3)

    def test_gamma_maximised_atm(self):
        """Gamma highest near ATM."""
        g_atm = equity_gamma(SPOT, 100, RATE, VOL, T)
        g_itm = equity_gamma(SPOT, 80, RATE, VOL, T)
        g_otm = equity_gamma(SPOT, 120, RATE, VOL, T)
        assert g_atm > g_itm
        assert g_atm > g_otm


class TestVega:
    def test_vega_positive(self):
        v = equity_vega(SPOT, STRIKE, RATE, VOL, T)
        assert v > 0

    def test_vega_vs_bump(self):
        v = equity_vega(SPOT, STRIKE, RATE, VOL, T)
        p_up = equity_option_price(SPOT, STRIKE, RATE, VOL + BUMP, T, OptionType.CALL)
        p_dn = equity_option_price(SPOT, STRIKE, RATE, VOL - BUMP, T, OptionType.CALL)
        bump_vega = (p_up - p_dn) / (2 * BUMP)
        assert v == pytest.approx(bump_vega, rel=1e-3)


class TestTheta:
    def test_theta_negative_for_call(self):
        """Long options lose value over time."""
        th = equity_theta(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert th < 0

    def test_theta_vs_bump(self):
        """Theta via bump: hold forward and df constant, change only T."""
        dt = 1e-4
        forward = SPOT * math.exp(RATE * T)
        df = math.exp(-RATE * T)
        from pricebook.black76 import black76_price
        p1 = black76_price(forward, STRIKE, VOL, T, df, OptionType.CALL)
        p2 = black76_price(forward, STRIKE, VOL, T - dt, df, OptionType.CALL)
        bump_theta = (p2 - p1) / dt
        th = equity_theta(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert th == pytest.approx(bump_theta, rel=0.01)


class TestRho:
    def test_call_rho_positive(self):
        """Call rho is positive (higher rates help calls)."""
        r = equity_rho(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert r > 0

    def test_put_rho_negative(self):
        r = equity_rho(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        assert r < 0

    def test_rho_vs_bump(self):
        dr = 1e-4
        p_up = equity_option_price(SPOT, STRIKE, RATE + dr, VOL, T, OptionType.CALL)
        p_dn = equity_option_price(SPOT, STRIKE, RATE - dr, VOL, T, OptionType.CALL)
        bump_rho = (p_up - p_dn) / (2 * dr)
        r = equity_rho(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert r == pytest.approx(bump_rho, rel=0.01)
