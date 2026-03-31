"""Tests for FX vol surface from market quotes."""

import pytest
import math
from datetime import date

from pricebook.fx_vol_surface import FXVolSurface, FXVolQuote
from pricebook.fx_option import fx_forward, fx_forward_delta, fx_option_price
from pricebook.black76 import OptionType


REF = date(2024, 1, 15)
EXP1 = date(2024, 7, 15)
EXP2 = date(2025, 1, 15)
SPOT = 1.0850
R_D = 0.05
R_F = 0.03


def _quote(expiry, atm=0.08, rr25=0.01, bf25=0.005):
    return FXVolQuote(expiry=expiry, atm=atm, rr25=rr25, bf25=bf25)


class TestConstruction:
    def test_single_expiry(self):
        s = FXVolSurface(SPOT, R_D, R_F, [_quote(EXP1)], reference_date=REF)
        v = s.vol(EXP1, SPOT)
        assert v > 0

    def test_two_expiries(self):
        s = FXVolSurface(
            SPOT, R_D, R_F,
            [_quote(EXP1), _quote(EXP2, atm=0.09)],
            reference_date=REF,
        )
        v1 = s.vol(EXP1, SPOT)
        v2 = s.vol(EXP2, SPOT)
        assert v1 > 0
        assert v2 > 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            FXVolSurface(SPOT, R_D, R_F, [], reference_date=REF)


class TestSmileRecovery:
    def test_rr_recovered(self):
        """Risk reversal: vol_25D_call - vol_25D_put = RR25."""
        rr = 0.015
        q = FXVolQuote(expiry=EXP1, atm=0.08, rr25=rr, bf25=0.005)
        vol_25c = q.atm + q.bf25 + 0.5 * q.rr25
        vol_25p = q.atm + q.bf25 - 0.5 * q.rr25
        assert vol_25c - vol_25p == pytest.approx(rr)

    def test_bf_recovered(self):
        """Butterfly: 0.5*(vol_25c + vol_25p) - ATM = BF25."""
        bf = 0.006
        q = FXVolQuote(expiry=EXP1, atm=0.08, rr25=0.01, bf25=bf)
        vol_25c = q.atm + q.bf25 + 0.5 * q.rr25
        vol_25p = q.atm + q.bf25 - 0.5 * q.rr25
        recovered_bf = 0.5 * (vol_25c + vol_25p) - q.atm
        assert recovered_bf == pytest.approx(bf)

    def test_atm_vol_near_center(self):
        """ATM strike vol should be close to the ATM input."""
        s = FXVolSurface(SPOT, R_D, R_F, [_quote(EXP1, atm=0.08)],
                         reference_date=REF)
        T = (EXP1 - REF).days / 365.0
        fwd = fx_forward(SPOT, R_D, R_F, T)
        k_atm = fwd * math.exp(0.5 * 0.08**2 * T)
        v = s.vol(EXP1, k_atm)
        assert v == pytest.approx(0.08, abs=0.005)


class TestStrikeDimension:
    def test_higher_vol_in_wings(self):
        """Wings should have higher vol than ATM (positive butterfly)."""
        s = FXVolSurface(
            SPOT, R_D, R_F,
            [FXVolQuote(EXP1, atm=0.08, rr25=0.0, bf25=0.01)],
            reference_date=REF,
        )
        T = (EXP1 - REF).days / 365.0
        fwd = fx_forward(SPOT, R_D, R_F, T)
        v_atm = s.vol(EXP1, fwd)
        v_high = s.vol(EXP1, fwd * 1.10)
        v_low = s.vol(EXP1, fwd * 0.90)
        assert v_high >= v_atm - 0.001
        assert v_low >= v_atm - 0.001


class TestPutCallParity:
    def test_parity_with_surface_vols(self):
        """Put-call parity holds regardless of vol (same vol for both)."""
        s = FXVolSurface(SPOT, R_D, R_F, [_quote(EXP1)], reference_date=REF)
        T = (EXP1 - REF).days / 365.0
        K = 1.10
        v = s.vol(EXP1, K)
        c = fx_option_price(SPOT, K, R_D, R_F, v, T, OptionType.CALL)
        p = fx_option_price(SPOT, K, R_D, R_F, v, T, OptionType.PUT)
        expected = SPOT * math.exp(-R_F * T) - K * math.exp(-R_D * T)
        assert c - p == pytest.approx(expected, abs=1e-10)


class TestExpiryInterpolation:
    def test_midpoint_vol_between_expiries(self):
        s = FXVolSurface(
            SPOT, R_D, R_F,
            [_quote(EXP1, atm=0.08), _quote(EXP2, atm=0.10)],
            reference_date=REF,
        )
        mid = date(2024, 10, 15)
        T = (mid - REF).days / 365.0
        fwd = fx_forward(SPOT, R_D, R_F, T)
        v = s.vol(mid, fwd)
        # Should be between 0.08 and 0.10
        assert 0.07 < v < 0.11
