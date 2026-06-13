"""Regression for L2 Wave-2 audit — `fx_lookback_floating` returned 0
unconditionally at `vol <= 0 or T <= 0`, dropping the deterministic
non-zero value when the spot path drifts in or out of the running
extreme.

At vol=0, T>0, the spot path is the deterministic exponential
``S_t = spot·exp(a·t)`` where ``a = rate_dom - rate_for``.  Over [0, T]:

  - If a >= 0: monotone non-decreasing → min = spot, max = spot_T.
  - If a <  0: monotone decreasing     → min = spot_T, max = spot.

The floating-strike call payoff is ``S_T − running_min``; the put
is ``running_max − S_T``.  Discounted by exp(-r_d·T).
"""

from __future__ import annotations

import math

import pytest

from pricebook.fx.fx_exotic import fx_lookback_floating


class TestFxLookbackVolZeroTPositiveCall:
    def test_positive_drift_call_pays_forward_minus_spot(self):
        # a = rd - rf = 5% - 0% = +5%.  forward = 100·exp(0.05) ≈ 105.13.
        # No running_extreme → effective_min = spot = 100.
        # Payoff = forward − 100; PV = df_d · payoff.
        res = fx_lookback_floating(spot=100.0, rate_dom=0.05, rate_for=0.0,
                                   vol=0.0, T=1.0, is_call=True)
        forward = 100.0 * math.exp(0.05)
        df_d = math.exp(-0.05)
        expected = df_d * (forward - 100.0)
        assert res.price == pytest.approx(expected, abs=1e-12)
        assert res.price > 0.0  # pre-fix returned 0

    def test_negative_drift_call_pays_zero_without_running_extreme(self):
        # a < 0 → path falls → min = spot_T = forward.
        # running_extreme = spot (default).  effective_min = min(spot, forward) = forward.
        # Payoff = spot_T − forward = 0.
        res = fx_lookback_floating(spot=100.0, rate_dom=0.0, rate_for=0.05,
                                   vol=0.0, T=1.0, is_call=True)
        assert res.price == pytest.approx(0.0, abs=1e-12)

    def test_negative_drift_call_with_low_running_extreme_pays(self):
        # a < 0, but running_extreme < forward means we already observed a low.
        # forward = 100·exp(-0.05) ≈ 95.12.
        # running_extreme = 90 < forward → effective_min = 90.
        # Payoff = forward − 90.
        res = fx_lookback_floating(spot=100.0, rate_dom=0.0, rate_for=0.05,
                                   vol=0.0, T=1.0, is_call=True,
                                   running_extreme=90.0)
        forward = 100.0 * math.exp(-0.05)
        expected = math.exp(0.0) * (forward - 90.0)
        assert res.price == pytest.approx(expected, abs=1e-12)


class TestFxLookbackVolZeroTPositivePut:
    def test_negative_drift_put_pays_spot_minus_forward(self):
        # a < 0 → path falls.  max = spot = 100, S_T = forward < 100.
        # Payoff = max − S_T = 100 − forward.
        res = fx_lookback_floating(spot=100.0, rate_dom=0.0, rate_for=0.05,
                                   vol=0.0, T=1.0, is_call=False)
        forward = 100.0 * math.exp(-0.05)
        df_d = math.exp(0.0)
        expected = df_d * (100.0 - forward)
        assert res.price == pytest.approx(expected, abs=1e-12)
        assert res.price > 0.0  # pre-fix returned 0

    def test_positive_drift_put_zero_without_running_max(self):
        # a > 0 → max = spot_T = forward.  running_extreme = spot (default).
        # effective_max = max(spot, forward) = forward.
        # Payoff = forward − S_T = 0.
        res = fx_lookback_floating(spot=100.0, rate_dom=0.05, rate_for=0.0,
                                   vol=0.0, T=1.0, is_call=False)
        assert res.price == pytest.approx(0.0, abs=1e-12)


class TestFxLookbackTZero:
    def test_T_zero_returns_zero(self):
        # T=0 boundary: no horizon, no lookback value.
        for is_call in [True, False]:
            res = fx_lookback_floating(spot=100.0, rate_dom=0.05, rate_for=0.02,
                                       vol=0.2, T=0.0, is_call=is_call)
            assert res.price == 0.0


class TestFxLookbackInteriorUnchanged:
    def test_interior_finite(self):
        # Interior path (vol>0, T>0): closed form unchanged.
        res = fx_lookback_floating(spot=100.0, rate_dom=0.05, rate_for=0.02,
                                   vol=0.20, T=1.0, is_call=True)
        # Lookback call > vanilla call > 0.
        assert res.price > 0.0
        assert math.isfinite(res.price)
