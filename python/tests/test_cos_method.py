"""Tests for COS method."""

import pytest
import math

from pricebook.cos_method import cos_price, bs_char_func, heston_char_func_cos
from pricebook.equity_option import equity_option_price
from pricebook.heston import heston_price
from pricebook.black76 import OptionType


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
BS_CALL = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
BS_PUT = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)


class TestCOSBlackScholes:
    def test_call_matches_bs(self):
        phi = bs_char_func(RATE, 0.0, VOL, T)
        p = cos_price(phi, SPOT, STRIKE, RATE, T, OptionType.CALL, N=128)
        assert p == pytest.approx(BS_CALL, rel=0.001)

    def test_put_matches_bs(self):
        phi = bs_char_func(RATE, 0.0, VOL, T)
        p = cos_price(phi, SPOT, STRIKE, RATE, T, OptionType.PUT, N=128)
        assert p == pytest.approx(BS_PUT, rel=0.001)

    def test_otm_call(self):
        bs = equity_option_price(SPOT, 120.0, RATE, VOL, T, OptionType.CALL)
        phi = bs_char_func(RATE, 0.0, VOL, T)
        p = cos_price(phi, SPOT, 120.0, RATE, T, OptionType.CALL, N=128)
        assert p == pytest.approx(bs, rel=0.01)

    def test_itm_put(self):
        bs = equity_option_price(SPOT, 80.0, RATE, VOL, T, OptionType.PUT)
        phi = bs_char_func(RATE, 0.0, VOL, T)
        p = cos_price(phi, SPOT, 80.0, RATE, T, OptionType.PUT, N=128)
        assert p == pytest.approx(bs, rel=0.01)

    def test_with_dividend(self):
        q = 0.02
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, q)
        phi = bs_char_func(RATE, q, VOL, T)
        p = cos_price(phi, SPOT, STRIKE, RATE, T, OptionType.CALL, div_yield=q, N=128)
        assert p == pytest.approx(bs, rel=0.005)

    def test_put_call_parity(self):
        phi = bs_char_func(RATE, 0.0, VOL, T)
        c = cos_price(phi, SPOT, STRIKE, RATE, T, OptionType.CALL, N=128)
        p = cos_price(phi, SPOT, STRIKE, RATE, T, OptionType.PUT, N=128)
        expected = SPOT - STRIKE * math.exp(-RATE * T)
        assert c - p == pytest.approx(expected, rel=0.005)


class TestConvergence:
    def test_more_terms_more_accurate(self):
        phi = bs_char_func(RATE, 0.0, VOL, T)
        err_32 = abs(cos_price(phi, SPOT, STRIKE, RATE, T, N=32) - BS_CALL)
        err_128 = abs(cos_price(phi, SPOT, STRIKE, RATE, T, N=128) - BS_CALL)
        assert err_128 < err_32

    def test_high_N_very_accurate(self):
        phi = bs_char_func(RATE, 0.0, VOL, T)
        p = cos_price(phi, SPOT, STRIKE, RATE, T, N=256)
        assert p == pytest.approx(BS_CALL, rel=1e-6)


class TestCOSHeston:
    def test_matches_semi_analytical(self):
        """COS Heston matches the Gauss-Legendre Heston pricer."""
        v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7
        heston_ref = heston_price(SPOT, STRIKE, RATE, T, v0, kappa, theta, xi, rho)

        phi = heston_char_func_cos(RATE, 0.0, v0, kappa, theta, xi, rho, T)
        cos_p = cos_price(phi, SPOT, STRIKE, RATE, T, OptionType.CALL, N=128, L=12)
        assert cos_p == pytest.approx(heston_ref, rel=0.02)

    def test_heston_put(self):
        v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7
        heston_ref = heston_price(SPOT, STRIKE, RATE, T, v0, kappa, theta, xi, rho,
                                  OptionType.PUT)
        phi = heston_char_func_cos(RATE, 0.0, v0, kappa, theta, xi, rho, T)
        cos_p = cos_price(phi, SPOT, STRIKE, RATE, T, OptionType.PUT, N=128, L=12)
        assert cos_p == pytest.approx(heston_ref, rel=0.02)


class TestCharFuncProtocol:
    def test_bs_callable(self):
        phi = bs_char_func(RATE, 0.0, VOL, T)
        assert callable(phi)
        val = phi(1.0)
        assert isinstance(val, complex)

    def test_bs_at_zero(self):
        """φ(0) = 1 for any valid char func."""
        phi = bs_char_func(RATE, 0.0, VOL, T)
        assert abs(phi(0)) == pytest.approx(1.0)
