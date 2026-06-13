"""Regression for L2 Wave-2 audit — `fx_charm` (∂Δ/∂t) returned 0 in
the ``vol <= 0 or T <= 0`` branch, silently dropping the deterministic
``±r_f·exp(-r_f T)`` discount-decay contribution.

At vol=0, T>0 the FX spot delta has the closed-form deterministic value
``±exp(-r_f T)·I(forward vs strike)``.  Its derivative w.r.t. calendar
time (charm) is:

    ∂Δ/∂t = (∂/∂t)[±exp(-r_f (T_expiry - t))·indicator]
           = ±r_f·exp(-r_f T)·indicator

Pre-fix returned 0 across all four ITM/OTM/call/put cases.  Post-fix:

  - ITM call  → +r_f·exp(-r_f T)
  - ITM put   → -r_f·exp(-r_f T)
  - OTM (both) → 0
  - ATM       → ±0.5·r_f·exp(-r_f T) (one-sided half-limit of indicator)

T=0 path preserved (returns 0 — boundary).
"""

from __future__ import annotations

import math

import pytest

from pricebook.fx.fx_greeks import fx_charm


class TestFxCharmVolZeroTPositive:
    def test_itm_call_positive_decay(self):
        # spot=120, strike=100, rd=0.05, rf=0.02 → fwd=120·exp(0.03)≈123.7 > 100.
        charm = fx_charm(spot=120.0, strike=100.0, rd=0.05, rf=0.02,
                         vol=0.0, T=1.0, is_call=True)
        expected = 0.02 * math.exp(-0.02)
        assert charm == pytest.approx(expected, abs=1e-12)
        assert charm > 0.0  # pre-fix returned 0

    def test_itm_put_negative_decay(self):
        # spot=80, strike=100 → fwd < 100 → ITM put.
        charm = fx_charm(spot=80.0, strike=100.0, rd=0.05, rf=0.02,
                         vol=0.0, T=1.0, is_call=False)
        expected = -0.02 * math.exp(-0.02)
        assert charm == pytest.approx(expected, abs=1e-12)

    def test_otm_call_zero(self):
        # spot=80, fwd < 100 → OTM call.
        charm = fx_charm(spot=80.0, strike=100.0, rd=0.05, rf=0.02,
                         vol=0.0, T=1.0, is_call=True)
        assert charm == 0.0

    def test_otm_put_zero(self):
        charm = fx_charm(spot=120.0, strike=100.0, rd=0.05, rf=0.02,
                         vol=0.0, T=1.0, is_call=False)
        assert charm == 0.0

    def test_atm_call_half(self):
        # rd == rf → forward = spot, so ATM at strike=spot.
        charm = fx_charm(spot=100.0, strike=100.0, rd=0.03, rf=0.03,
                         vol=0.0, T=1.0, is_call=True)
        expected = 0.5 * 0.03 * math.exp(-0.03)
        assert charm == pytest.approx(expected, abs=1e-12)

    def test_zero_rf_returns_zero(self):
        # r_f = 0 → discount-decay term vanishes regardless of moneyness.
        charm = fx_charm(spot=120.0, strike=100.0, rd=0.05, rf=0.0,
                         vol=0.0, T=1.0, is_call=True)
        assert charm == 0.0


class TestFxCharmTZero:
    def test_T_zero_returns_zero(self):
        # T=0 boundary: no time to act → 0.
        for is_call in [True, False]:
            for spot in [80.0, 100.0, 120.0]:
                charm = fx_charm(spot=spot, strike=100.0, rd=0.05, rf=0.02,
                                 vol=0.2, T=0.0, is_call=is_call)
                assert charm == 0.0


class TestFxCharmInteriorUnchanged:
    def test_interior_finite(self):
        # Interior path (vol>0, T>0): formula unchanged.
        charm = fx_charm(spot=100.0, strike=100.0, rd=0.05, rf=0.02,
                         vol=0.15, T=1.0, is_call=True)
        assert math.isfinite(charm)
