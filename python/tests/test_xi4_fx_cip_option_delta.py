"""XI4: FX CIP → Option → Delta Round-trip integration chain.

Two curves → CIP forward → GK option → put-call parity → spot delta →
strike_from_delta → verify round-trip. Verify forward from curves matches
S × exp((rd-rf)T).

Bug hotspots:
- FXForward.forward_rate() uses df ratio, fx_option_price() uses exp formula
- Must extract rates from same curves for consistency
- Delta conventions (spot, forward, premium-adjusted) must round-trip
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.bootstrap import bootstrap
from pricebook.black76 import OptionType
from pricebook.discount_curve import DiscountCurve
from pricebook.fx_forward import FXForward
from pricebook.fx_option import (
    fx_forward,
    fx_option_price,
    fx_spot_delta,
    fx_forward_delta,
    strike_from_delta,
)


# ---- Helpers ----

REF = date(2026, 4, 25)


def _usd_curve(ref: date) -> DiscountCurve:
    deposits = [(ref + timedelta(days=91), 0.045), (ref + timedelta(days=182), 0.044)]
    swaps = [
        (ref + timedelta(days=365), 0.043),
        (ref + timedelta(days=730), 0.042),
        (ref + timedelta(days=1825), 0.040),
    ]
    return bootstrap(ref, deposits, swaps)


def _eur_curve(ref: date) -> DiscountCurve:
    deposits = [(ref + timedelta(days=91), 0.030), (ref + timedelta(days=182), 0.029)]
    swaps = [
        (ref + timedelta(days=365), 0.028),
        (ref + timedelta(days=730), 0.027),
        (ref + timedelta(days=1825), 0.025),
    ]
    return bootstrap(ref, deposits, swaps)


SPOT = 1.0850  # EURUSD
VOL = 0.08


def _rates_from_curves(dom_curve, for_curve, ref, T_days):
    """Extract continuous rates from curves for GK pricing."""
    mat = ref + timedelta(days=T_days)
    T = T_days / 365.0
    df_d = dom_curve.df(mat)
    df_f = for_curve.df(mat)
    r_d = -math.log(df_d) / T
    r_f = -math.log(df_f) / T
    return r_d, r_f, T


# ---- R1: CIP forward consistency ----

class TestXI4R1CIPConsistency:
    """Forward from curves must match S × exp((rd-rf)T)."""

    def test_cip_forward_matches_exp_formula(self):
        """FXForward.forward_rate (df ratio) ≈ S × exp((rd-rf)T)."""
        usd = _usd_curve(REF)
        eur = _eur_curve(REF)
        mat = REF + timedelta(days=365)

        # CIP via df ratio: F = S × df_base / df_quote
        # EURUSD: base=EUR (foreign), quote=USD (domestic)
        fwd_cip = FXForward.forward_rate(SPOT, mat, eur, usd)

        # Continuous rate formula
        r_d, r_f, T = _rates_from_curves(usd, eur, REF, 365)
        fwd_exp = fx_forward(SPOT, r_d, r_f, T)

        assert fwd_cip == pytest.approx(fwd_exp, rel=1e-6)

    def test_cip_forward_multiple_tenors(self):
        """CIP consistency at 3M, 1Y, 5Y."""
        usd = _usd_curve(REF)
        eur = _eur_curve(REF)

        for days in [91, 365, 1825]:
            mat = REF + timedelta(days=days)
            fwd_cip = FXForward.forward_rate(SPOT, mat, eur, usd)
            r_d, r_f, T = _rates_from_curves(usd, eur, REF, days)
            fwd_exp = fx_forward(SPOT, r_d, r_f, T)
            assert fwd_cip == pytest.approx(fwd_exp, rel=1e-4), f"Mismatch at {days}d"

    def test_higher_domestic_rate_forward_above_spot(self):
        """USD rate > EUR rate → EURUSD forward > spot."""
        usd = _usd_curve(REF)
        eur = _eur_curve(REF)
        mat = REF + timedelta(days=365)
        fwd = FXForward.forward_rate(SPOT, mat, eur, usd)
        assert fwd > SPOT

    def test_fx_forward_pv_at_market_forward(self):
        """FX forward struck at market forward has PV ≈ 0."""
        usd = _usd_curve(REF)
        eur = _eur_curve(REF)
        mat = REF + timedelta(days=365)
        fwd = FXForward.forward_rate(SPOT, mat, eur, usd)

        from pricebook.currency import CurrencyPair
        pair = CurrencyPair("EUR", "USD")
        contract = FXForward(pair, mat, strike=fwd)
        pv = contract.pv(SPOT, eur, usd)
        assert pv == pytest.approx(0.0, abs=1.0)


# ---- R2: Put-call parity ----

class TestXI4R2PutCallParity:
    """C - P = S×exp(-rf×T) - K×exp(-rd×T)."""

    def _check_parity(self, strike, T_days):
        usd = _usd_curve(REF)
        eur = _eur_curve(REF)
        r_d, r_f, T = _rates_from_curves(usd, eur, REF, T_days)

        call = fx_option_price(SPOT, strike, r_d, r_f, VOL, T, OptionType.CALL)
        put = fx_option_price(SPOT, strike, r_d, r_f, VOL, T, OptionType.PUT)

        # Put-call parity: C - P = S×exp(-rf×T) - K×exp(-rd×T)
        lhs = call - put
        rhs = SPOT * math.exp(-r_f * T) - strike * math.exp(-r_d * T)
        assert lhs == pytest.approx(rhs, rel=1e-6), (
            f"Parity failed: K={strike}, T={T_days}d"
        )

    def test_parity_atm(self):
        r_d, r_f, T = _rates_from_curves(_usd_curve(REF), _eur_curve(REF), REF, 365)
        fwd = fx_forward(SPOT, r_d, r_f, T)
        self._check_parity(fwd, 365)

    def test_parity_itm_call(self):
        self._check_parity(1.05, 365)

    def test_parity_otm_call(self):
        self._check_parity(1.15, 365)

    def test_parity_short_tenor(self):
        self._check_parity(1.08, 91)

    def test_parity_long_tenor(self):
        self._check_parity(1.10, 1825)


# ---- R3: Delta round-trip ----

class TestXI4R3DeltaRoundTrip:
    """strike_from_delta → delta → verify round-trip."""

    def _check_round_trip(self, delta_val, delta_type, option_type, T_days=365):
        r_d, r_f, T = _rates_from_curves(
            _usd_curve(REF), _eur_curve(REF), REF, T_days)

        K = strike_from_delta(SPOT, delta_val, r_d, r_f, VOL, T,
                              delta_type=delta_type, option_type=option_type)
        assert K > 0, f"Invalid strike: {K}"

        if delta_type == "spot":
            recovered = fx_spot_delta(SPOT, K, r_d, r_f, VOL, T, option_type)
        elif delta_type == "forward":
            recovered = fx_forward_delta(SPOT, K, r_d, r_f, VOL, T, option_type)
        else:
            pytest.skip("premium_adjusted not tested here")

        assert recovered == pytest.approx(delta_val, abs=1e-6), (
            f"Round-trip failed: delta_type={delta_type}, target={delta_val}, "
            f"recovered={recovered}"
        )

    def test_spot_delta_25_call(self):
        self._check_round_trip(0.25, "spot", OptionType.CALL)

    def test_spot_delta_10_call(self):
        self._check_round_trip(0.10, "spot", OptionType.CALL)

    def test_spot_delta_25_put(self):
        self._check_round_trip(-0.25, "spot", OptionType.PUT)

    def test_spot_delta_10_put(self):
        self._check_round_trip(-0.10, "spot", OptionType.PUT)

    def test_forward_delta_25_call(self):
        self._check_round_trip(0.25, "forward", OptionType.CALL)

    def test_forward_delta_25_put(self):
        self._check_round_trip(-0.25, "forward", OptionType.PUT)

    def test_spot_delta_50_call_near_atm(self):
        """50-delta call strike should be near ATM forward."""
        r_d, r_f, T = _rates_from_curves(
            _usd_curve(REF), _eur_curve(REF), REF, 365)
        fwd = fx_forward(SPOT, r_d, r_f, T)

        K50 = strike_from_delta(SPOT, 0.50, r_d, r_f, VOL, T,
                                delta_type="forward", option_type=OptionType.CALL)
        # 50-delta forward strike ≈ ATM forward (exactly at F for forward delta)
        assert K50 == pytest.approx(fwd, rel=0.02)


# ---- R4: Edge cases ----

class TestXI4R4EdgeCases:
    """Edge cases for FX integration."""

    def test_equal_rates_forward_equals_spot(self):
        """If domestic = foreign rate, forward = spot."""
        flat = DiscountCurve.flat(REF, 0.03)
        mat = REF + timedelta(days=365)
        fwd = FXForward.forward_rate(SPOT, mat, flat, flat)
        assert fwd == pytest.approx(SPOT, rel=1e-6)

    def test_zero_vol_call_equals_intrinsic(self):
        """At zero vol, call = max(F-K, 0) × df."""
        r_d, r_f, T = _rates_from_curves(
            _usd_curve(REF), _eur_curve(REF), REF, 365)
        fwd = fx_forward(SPOT, r_d, r_f, T)
        K = fwd - 0.01  # ITM call

        call = fx_option_price(SPOT, K, r_d, r_f, 1e-8, T, OptionType.CALL)
        intrinsic = max(fwd - K, 0.0) * math.exp(-r_d * T)
        assert call == pytest.approx(intrinsic, rel=0.01)

    def test_higher_vol_higher_price(self):
        """Higher vol → higher option price."""
        r_d, r_f, T = _rates_from_curves(
            _usd_curve(REF), _eur_curve(REF), REF, 365)
        low = fx_option_price(SPOT, 1.10, r_d, r_f, 0.05, T, OptionType.CALL)
        high = fx_option_price(SPOT, 1.10, r_d, r_f, 0.15, T, OptionType.CALL)
        assert high > low

    def test_call_delta_between_0_and_1(self):
        """Spot delta of a call must be in (0, 1)."""
        r_d, r_f, T = _rates_from_curves(
            _usd_curve(REF), _eur_curve(REF), REF, 365)
        delta = fx_spot_delta(SPOT, 1.10, r_d, r_f, VOL, T, OptionType.CALL)
        assert 0 < delta < 1

    def test_put_delta_between_neg1_and_0(self):
        """Spot delta of a put must be in (-1, 0)."""
        r_d, r_f, T = _rates_from_curves(
            _usd_curve(REF), _eur_curve(REF), REF, 365)
        delta = fx_spot_delta(SPOT, 1.10, r_d, r_f, VOL, T, OptionType.PUT)
        assert -1 < delta < 0
