"""Tests for pricebook.credit.tranche_option."""

import math
import pytest

from pricebook.credit.tranche_option import (
    tranche_option_black,
    tranche_option_bachelier,
    tranche_option_greeks,
    tranche_straddle,
    tranche_forward_spread,
)

# Common parameters
SPREAD = 0.0100      # 100 bps forward spread
STRIKE = 0.0100      # ATM strike
VOL    = 0.60        # log-normal vol
T      = 1.0
R      = 0.05
ANNUITY = 4.2


class TestTrancheOptionBlack:
    def test_payer_price_positive(self):
        res = tranche_option_black(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=True)
        assert res.price > 0.0

    def test_receiver_price_positive(self):
        res = tranche_option_black(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=False)
        assert res.price > 0.0

    def test_payer_receiver_parity(self):
        """ATM: payer ≈ receiver (put-call parity with zero moneyness)."""
        payer = tranche_option_black(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=True)
        receiver = tranche_option_black(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=False)
        assert payer.price == pytest.approx(receiver.price, rel=0.05)

    def test_payer_delta_positive(self):
        res = tranche_option_black(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=True)
        assert res.delta > 0.0

    def test_receiver_delta_negative(self):
        res = tranche_option_black(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=False)
        assert res.delta < 0.0

    def test_vega_positive(self):
        res = tranche_option_black(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=True)
        assert res.vega > 0.0

    def test_otm_payer_cheaper_than_atm(self):
        atm = tranche_option_black(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=True)
        otm = tranche_option_black(SPREAD, 0.0150, VOL, T, R, ANNUITY, is_payer=True)
        assert otm.price < atm.price

    def test_zero_expiry_intrinsic_payer(self):
        """At T=0, price should equal discounted intrinsic (zero for ATM)."""
        res = tranche_option_black(0.0120, 0.0100, VOL, 0.0, R, ANNUITY, is_payer=True)
        expected = ANNUITY * (0.0120 - 0.0100)
        assert res.price == pytest.approx(expected, rel=1e-6)


class TestTrancheOptionBachelier:
    def test_bachelier_payer_positive(self):
        res = tranche_option_bachelier(SPREAD, STRIKE, 0.002, T, R, ANNUITY, is_payer=True)
        assert res.price > 0.0

    def test_bachelier_receiver_positive(self):
        """ATM receiver should be positive."""
        vol_n = VOL * SPREAD  # ATM normal vol from lognormal
        res = tranche_option_bachelier(SPREAD, SPREAD, vol_n, T, R, ANNUITY, is_payer=False)
        assert res.price > 0.0

    def test_bachelier_atm_close_to_black(self):
        """For ATM with matched vols, Bachelier and Black should give similar results."""
        black_res = tranche_option_black(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=True)
        # Convert log-normal vol to approximate normal vol: sigma_N ≈ sigma_LN * F
        vol_normal = VOL * SPREAD
        bach_res = tranche_option_bachelier(SPREAD, STRIKE, vol_normal, T, R, ANNUITY, is_payer=True)
        assert bach_res.price == pytest.approx(black_res.price, rel=0.05)

    def test_bachelier_vega_positive(self):
        res = tranche_option_bachelier(SPREAD, STRIKE, 0.002, T, R, ANNUITY, is_payer=True)
        assert res.vega > 0.0


class TestTrancheOptionGreeks:
    def test_spread_delta_positive_payer(self):
        greeks = tranche_option_greeks(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=True)
        assert greeks["spread_delta"] > 0.0

    def test_vega_positive(self):
        greeks = tranche_option_greeks(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=True)
        assert greeks["vega"] > 0.0

    def test_spread_gamma_positive(self):
        greeks = tranche_option_greeks(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=True)
        assert greeks["spread_gamma"] > 0.0


class TestTrancheStraddle:
    def test_straddle_price_equals_payer_plus_receiver(self):
        straddle = tranche_straddle(SPREAD, STRIKE, VOL, T, R, ANNUITY)
        payer = tranche_option_black(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=True)
        receiver = tranche_option_black(SPREAD, STRIKE, VOL, T, R, ANNUITY, is_payer=False)
        assert straddle.price == pytest.approx(payer.price + receiver.price, rel=1e-9)

    def test_max_loss_equals_price(self):
        straddle = tranche_straddle(SPREAD, STRIKE, VOL, T, R, ANNUITY)
        assert straddle.max_loss == pytest.approx(straddle.price, rel=1e-9)

    def test_breakeven_up_above_strike(self):
        straddle = tranche_straddle(SPREAD, STRIKE, VOL, T, R, ANNUITY)
        assert straddle.breakeven_up > STRIKE


class TestTrancheForwardSpread:
    def test_short_T_zero_losses_approx_current(self):
        """For very short T and no expected losses, forward ≈ current spread."""
        fwd = tranche_forward_spread(SPREAD, 0.05, 0.01, 0.0)
        assert fwd == pytest.approx(SPREAD, rel=0.01)

    def test_losses_increase_forward_spread(self):
        """Expected losses reduce survival factor, raising forward spread."""
        fwd_no_loss = tranche_forward_spread(SPREAD, 0.05, 1.0, 0.0)
        fwd_with_loss = tranche_forward_spread(SPREAD, 0.05, 1.0, 0.10)
        assert fwd_with_loss > fwd_no_loss
