"""Tests for FX option pricing and delta conventions."""

import pytest
import math

from pricebook.fx_option import (
    fx_forward,
    fx_option_price,
    fx_spot_delta,
    fx_forward_delta,
    fx_premium_adjusted_delta,
    fx_vega,
    strike_from_delta,
)
from pricebook.black76 import OptionType


SPOT = 1.0850  # EUR/USD
R_D = 0.05     # USD rate (domestic = quote)
R_F = 0.03     # EUR rate (foreign = base)
VOL = 0.08
T = 1.0


class TestFXForward:
    def test_cip_forward(self):
        f = fx_forward(SPOT, R_D, R_F, T)
        expected = SPOT * math.exp((R_D - R_F) * T)
        assert f == pytest.approx(expected)

    def test_forward_above_spot_when_rd_gt_rf(self):
        assert fx_forward(SPOT, R_D, R_F, T) > SPOT


class TestFXOptionPrice:
    def test_call_positive(self):
        p = fx_option_price(SPOT, SPOT, R_D, R_F, VOL, T, OptionType.CALL)
        assert p > 0

    def test_put_positive(self):
        p = fx_option_price(SPOT, SPOT, R_D, R_F, VOL, T, OptionType.PUT)
        assert p > 0

    def test_put_call_parity(self):
        """C - P = S*exp(-r_f*T) - K*exp(-r_d*T)."""
        K = 1.10
        c = fx_option_price(SPOT, K, R_D, R_F, VOL, T, OptionType.CALL)
        p = fx_option_price(SPOT, K, R_D, R_F, VOL, T, OptionType.PUT)
        expected = SPOT * math.exp(-R_F * T) - K * math.exp(-R_D * T)
        assert c - p == pytest.approx(expected, abs=1e-10)

    def test_higher_vol_higher_price(self):
        p1 = fx_option_price(SPOT, SPOT, R_D, R_F, 0.05, T, OptionType.CALL)
        p2 = fx_option_price(SPOT, SPOT, R_D, R_F, 0.15, T, OptionType.CALL)
        assert p2 > p1


class TestDeltaConventions:
    def test_spot_delta_call_positive(self):
        d = fx_spot_delta(SPOT, SPOT, R_D, R_F, VOL, T, OptionType.CALL)
        assert 0 < d < 1

    def test_spot_delta_put_negative(self):
        d = fx_spot_delta(SPOT, SPOT, R_D, R_F, VOL, T, OptionType.PUT)
        assert -1 < d < 0

    def test_call_put_spot_delta_relation(self):
        """Delta_call - Delta_put = exp(-r_f*T)."""
        dc = fx_spot_delta(SPOT, SPOT, R_D, R_F, VOL, T, OptionType.CALL)
        dp = fx_spot_delta(SPOT, SPOT, R_D, R_F, VOL, T, OptionType.PUT)
        assert dc - dp == pytest.approx(math.exp(-R_F * T), abs=1e-10)

    def test_forward_delta_call_positive(self):
        d = fx_forward_delta(SPOT, SPOT, R_D, R_F, VOL, T, OptionType.CALL)
        assert 0 < d < 1

    def test_forward_delta_call_put_sum(self):
        """Forward delta: call - put = 1."""
        dc = fx_forward_delta(SPOT, SPOT, R_D, R_F, VOL, T, OptionType.CALL)
        dp = fx_forward_delta(SPOT, SPOT, R_D, R_F, VOL, T, OptionType.PUT)
        assert dc - dp == pytest.approx(1.0, abs=1e-10)

    def test_premium_adjusted_delta_call(self):
        d = fx_premium_adjusted_delta(SPOT, SPOT, R_D, R_F, VOL, T, OptionType.CALL)
        assert 0 < d < 1

    def test_spot_delta_vs_bump(self):
        bump = 1e-4
        p_up = fx_option_price(SPOT + bump, SPOT, R_D, R_F, VOL, T, OptionType.CALL)
        p_dn = fx_option_price(SPOT - bump, SPOT, R_D, R_F, VOL, T, OptionType.CALL)
        bump_delta = (p_up - p_dn) / (2 * bump)
        analytical = fx_spot_delta(SPOT, SPOT, R_D, R_F, VOL, T, OptionType.CALL)
        assert analytical == pytest.approx(bump_delta, rel=1e-4)


class TestStrikeFromDelta:
    def test_forward_delta_round_trip(self):
        """strike_from_delta → fx_forward_delta → recover original delta."""
        target_delta = 0.25
        K = strike_from_delta(SPOT, target_delta, R_D, R_F, VOL, T,
                              delta_type="forward", option_type=OptionType.CALL)
        recovered = fx_forward_delta(SPOT, K, R_D, R_F, VOL, T, OptionType.CALL)
        assert recovered == pytest.approx(target_delta, abs=1e-8)

    def test_spot_delta_round_trip(self):
        target_delta = 0.25
        K = strike_from_delta(SPOT, target_delta, R_D, R_F, VOL, T,
                              delta_type="spot", option_type=OptionType.CALL)
        recovered = fx_spot_delta(SPOT, K, R_D, R_F, VOL, T, OptionType.CALL)
        assert recovered == pytest.approx(target_delta, abs=1e-8)

    def test_premium_adjusted_round_trip(self):
        target_delta = 0.25
        K = strike_from_delta(SPOT, target_delta, R_D, R_F, VOL, T,
                              delta_type="premium_adjusted", option_type=OptionType.CALL)
        recovered = fx_premium_adjusted_delta(SPOT, K, R_D, R_F, VOL, T, OptionType.CALL)
        assert recovered == pytest.approx(target_delta, abs=1e-8)

    def test_put_delta_round_trip(self):
        target_delta = -0.25
        K = strike_from_delta(SPOT, target_delta, R_D, R_F, VOL, T,
                              delta_type="forward", option_type=OptionType.PUT)
        recovered = fx_forward_delta(SPOT, K, R_D, R_F, VOL, T, OptionType.PUT)
        assert recovered == pytest.approx(target_delta, abs=1e-8)

    def test_atm_forward_delta_strike_equals_forward(self):
        """ATM forward (50 delta) strike ≈ forward."""
        K = strike_from_delta(SPOT, 0.5, R_D, R_F, VOL, T,
                              delta_type="forward", option_type=OptionType.CALL)
        fwd = fx_forward(SPOT, R_D, R_F, T)
        # Not exactly equal due to vol adjustment, but close
        assert K == pytest.approx(fwd, rel=0.01)


class TestVega:
    def test_vega_positive(self):
        v = fx_vega(SPOT, SPOT, R_D, R_F, VOL, T)
        assert v > 0

    def test_vega_vs_bump(self):
        bump = 1e-4
        p_up = fx_option_price(SPOT, SPOT, R_D, R_F, VOL + bump, T, OptionType.CALL)
        p_dn = fx_option_price(SPOT, SPOT, R_D, R_F, VOL - bump, T, OptionType.CALL)
        bump_vega = (p_up - p_dn) / (2 * bump)
        analytical = fx_vega(SPOT, SPOT, R_D, R_F, VOL, T)
        assert analytical == pytest.approx(bump_vega, rel=1e-3)
