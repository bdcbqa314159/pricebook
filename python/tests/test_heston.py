"""Tests for Heston model."""

import pytest
import math

from pricebook.heston import heston_price, _heston_f, heston_calibrate
from pricebook.equity_option import equity_option_price
from pricebook.black76 import OptionType


SPOT, STRIKE, RATE, T = 100.0, 100.0, 0.05, 1.0
V0, KAPPA, THETA, XI, RHO = 0.04, 2.0, 0.04, 0.3, -0.7


class TestCharFunction:
    def test_at_zero(self):
        """f_j(0) should be finite and well-defined."""
        x = math.log(SPOT)
        f = _heston_f(0.001, T, RATE, 0.0, V0, KAPPA, THETA, XI, RHO, x, 2)
        assert abs(f) > 0

    def test_returns_complex(self):
        x = math.log(SPOT)
        f = _heston_f(1.0, T, RATE, 0.0, V0, KAPPA, THETA, XI, RHO, x, 2)
        assert isinstance(f, complex)


class TestHestonPrice:
    def test_call_positive(self):
        p = heston_price(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO)
        assert p > 0

    def test_put_positive(self):
        p = heston_price(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO,
                         OptionType.PUT)
        assert p > 0

    def test_put_call_parity(self):
        c = heston_price(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO,
                         OptionType.CALL)
        p = heston_price(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO,
                         OptionType.PUT)
        expected = SPOT - STRIKE * math.exp(-RATE * T)
        assert c - p == pytest.approx(expected, rel=0.01)

    def test_zero_xi_matches_bs(self):
        """Zero vol-of-vol → Black-Scholes."""
        vol = math.sqrt(V0)
        bs = equity_option_price(SPOT, STRIKE, RATE, vol, T, OptionType.CALL)
        heston = heston_price(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, 0.001, 0.0)
        assert heston == pytest.approx(bs, rel=0.02)

    def test_otm_call(self):
        p = heston_price(SPOT, 120.0, RATE, T, V0, KAPPA, THETA, XI, RHO)
        assert p > 0
        assert p < heston_price(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO)

    def test_negative_rho_fatter_left_tail(self):
        """Negative rho → fatter left tail → higher OTM put price."""
        p_neg = heston_price(SPOT, 80.0, RATE, T, V0, KAPPA, THETA, XI, -0.7,
                             OptionType.PUT)
        p_zero = heston_price(SPOT, 80.0, RATE, T, V0, KAPPA, THETA, XI, 0.0,
                              OptionType.PUT)
        assert p_neg > p_zero

    def test_with_dividend(self):
        p = heston_price(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO,
                         div_yield=0.02)
        assert p > 0

    def test_higher_v0_higher_price(self):
        p_low = heston_price(SPOT, STRIKE, RATE, T, 0.01, KAPPA, THETA, XI, RHO)
        p_high = heston_price(SPOT, STRIKE, RATE, T, 0.09, KAPPA, THETA, XI, RHO)
        assert p_high > p_low


class TestHestonCalibration:
    def test_recovers_known_prices(self):
        """Calibrate to Heston-generated prices, recover params approximately."""
        strikes = [SPOT * m for m in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]]
        prices = [
            heston_price(SPOT, k, RATE, T, V0, KAPPA, THETA, XI, RHO)
            for k in strikes
        ]

        result = heston_calibrate(SPOT, strikes, prices, RATE, T)

        # Reprice with calibrated params
        for k, mp in zip(strikes, prices):
            model = heston_price(
                SPOT, k, RATE, T,
                result["v0"], result["kappa"], result["theta"],
                result["xi"], result["rho"],
            )
            assert model == pytest.approx(mp, rel=0.02)

    def test_rmse_low(self):
        strikes = [SPOT * m for m in [0.9, 1.0, 1.1]]
        prices = [
            heston_price(SPOT, k, RATE, T, V0, KAPPA, THETA, XI, RHO)
            for k in strikes
        ]
        result = heston_calibrate(SPOT, strikes, prices, RATE, T)
        assert result["rmse"] < 0.1
