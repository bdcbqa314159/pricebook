"""
Slice 13 round-trip validation: FX options.

1. ATM-DNS strike has zero straddle delta
2. RR and BF recovered from surface
3. Put-call parity with FX discount factors
4. Synthetic forward from options matches CIP forward
5. Delta round-trips (all conventions)
"""

import pytest
import math
from datetime import date

from pricebook.fx_option import (
    fx_forward,
    fx_option_price,
    fx_spot_delta,
    fx_forward_delta,
    fx_premium_adjusted_delta,
    strike_from_delta,
)
from pricebook.fx_vol_surface import FXVolSurface, FXVolQuote
from pricebook.black76 import OptionType


REF = date(2024, 1, 15)
EXP = date(2025, 1, 15)
SPOT = 1.0850
R_D = 0.05
R_F = 0.03
VOL = 0.08
T = (EXP - REF).days / 365.0


class TestATMDeltaNeutral:
    def test_atm_dns_straddle_delta_near_zero(self):
        """ATM-DNS: straddle (call + put) has near-zero forward delta."""
        # ATM-DNS strike: K such that call_delta + put_delta ≈ 0
        # For forward delta: call_delta + put_delta = N(d1) + N(d1) - 1 = 2*N(d1) - 1
        # Zero when d1 = 0, i.e. K = F * exp(0.5 * vol^2 * T)
        fwd = fx_forward(SPOT, R_D, R_F, T)
        k_dns = fwd * math.exp(0.5 * VOL**2 * T)

        dc = fx_forward_delta(SPOT, k_dns, R_D, R_F, VOL, T, OptionType.CALL)
        dp = fx_forward_delta(SPOT, k_dns, R_D, R_F, VOL, T, OptionType.PUT)
        straddle_delta = dc + dp
        assert straddle_delta == pytest.approx(0.0, abs=0.001)


class TestRRBFRecovery:
    def test_rr_from_surface(self):
        """Risk reversal: vol at 25D call strike - vol at 25D put strike ≈ RR input."""
        rr = 0.015
        bf = 0.005
        q = FXVolQuote(EXP, atm=VOL, rr25=rr, bf25=bf)
        s = FXVolSurface(SPOT, R_D, R_F, [q], reference_date=REF)

        vol_25c = VOL + bf + 0.5 * rr
        vol_25p = VOL + bf - 0.5 * rr

        k_25c = strike_from_delta(SPOT, 0.25, R_D, R_F, vol_25c, T,
                                   delta_type="forward", option_type=OptionType.CALL)
        k_25p = strike_from_delta(SPOT, -0.25, R_D, R_F, vol_25p, T,
                                   delta_type="forward", option_type=OptionType.PUT)

        recovered_rr = s.vol(EXP, k_25c) - s.vol(EXP, k_25p)
        assert recovered_rr == pytest.approx(rr, abs=0.003)

    def test_bf_from_surface(self):
        """Butterfly: 0.5*(vol_25c + vol_25p) - ATM ≈ BF input."""
        rr = 0.01
        bf = 0.008
        q = FXVolQuote(EXP, atm=VOL, rr25=rr, bf25=bf)
        s = FXVolSurface(SPOT, R_D, R_F, [q], reference_date=REF)

        vol_25c = VOL + bf + 0.5 * rr
        vol_25p = VOL + bf - 0.5 * rr

        k_25c = strike_from_delta(SPOT, 0.25, R_D, R_F, vol_25c, T,
                                   delta_type="forward", option_type=OptionType.CALL)
        k_25p = strike_from_delta(SPOT, -0.25, R_D, R_F, vol_25p, T,
                                   delta_type="forward", option_type=OptionType.PUT)

        fwd = fx_forward(SPOT, R_D, R_F, T)
        k_atm = fwd * math.exp(0.5 * VOL**2 * T)

        vol_c = s.vol(EXP, k_25c)
        vol_p = s.vol(EXP, k_25p)
        vol_a = s.vol(EXP, k_atm)
        recovered_bf = 0.5 * (vol_c + vol_p) - vol_a
        assert recovered_bf == pytest.approx(bf, abs=0.003)


class TestPutCallParityFX:
    @pytest.mark.parametrize("K", [1.00, 1.05, 1.10, 1.15, 1.20])
    def test_parity(self, K):
        """C - P = S*exp(-r_f*T) - K*exp(-r_d*T)."""
        c = fx_option_price(SPOT, K, R_D, R_F, VOL, T, OptionType.CALL)
        p = fx_option_price(SPOT, K, R_D, R_F, VOL, T, OptionType.PUT)
        expected = SPOT * math.exp(-R_F * T) - K * math.exp(-R_D * T)
        assert c - p == pytest.approx(expected, abs=1e-10)


class TestSyntheticForward:
    def test_synthetic_forward_matches_cip(self):
        """Synthetic forward from options = CIP forward.

        F_synthetic = K + exp(r_d*T) * (C - P)
        """
        K = 1.10
        c = fx_option_price(SPOT, K, R_D, R_F, VOL, T, OptionType.CALL)
        p = fx_option_price(SPOT, K, R_D, R_F, VOL, T, OptionType.PUT)
        synthetic_fwd = K + math.exp(R_D * T) * (c - p)
        cip_fwd = fx_forward(SPOT, R_D, R_F, T)
        assert synthetic_fwd == pytest.approx(cip_fwd, abs=1e-8)


class TestDeltaRoundTrips:
    @pytest.mark.parametrize("delta_type", ["spot", "forward", "premium_adjusted"])
    @pytest.mark.parametrize("target", [0.10, 0.25, 0.40])
    def test_call_delta_round_trip(self, delta_type, target):
        K = strike_from_delta(SPOT, target, R_D, R_F, VOL, T,
                              delta_type=delta_type, option_type=OptionType.CALL)
        if delta_type == "spot":
            recovered = fx_spot_delta(SPOT, K, R_D, R_F, VOL, T, OptionType.CALL)
        elif delta_type == "forward":
            recovered = fx_forward_delta(SPOT, K, R_D, R_F, VOL, T, OptionType.CALL)
        else:
            recovered = fx_premium_adjusted_delta(SPOT, K, R_D, R_F, VOL, T, OptionType.CALL)
        assert recovered == pytest.approx(target, abs=1e-8)

    @pytest.mark.parametrize("delta_type", ["spot", "forward", "premium_adjusted"])
    def test_put_delta_round_trip(self, delta_type):
        target = -0.25
        K = strike_from_delta(SPOT, target, R_D, R_F, VOL, T,
                              delta_type=delta_type, option_type=OptionType.PUT)
        if delta_type == "spot":
            recovered = fx_spot_delta(SPOT, K, R_D, R_F, VOL, T, OptionType.PUT)
        elif delta_type == "forward":
            recovered = fx_forward_delta(SPOT, K, R_D, R_F, VOL, T, OptionType.PUT)
        else:
            recovered = fx_premium_adjusted_delta(SPOT, K, R_D, R_F, VOL, T, OptionType.PUT)
        assert recovered == pytest.approx(target, abs=1e-8)
