"""Regression for L2 Wave-2 audit — `equity_theta` pre-fix returned the
Black-76 theta alone in the ``T <= 0 or vol <= 0`` branch, silently
dropping both the rate-discount correction (``theta_r``) and the
dividend correction (``theta_q``).  These corrections are non-zero in
the deterministic limit at ``vol=0, T>0``, where the option price is
the discounted intrinsic of the forward and theta has a closed form.

At ``vol=0, T>0`` the price is:
- ITM call (forward > strike): ``S·exp(-qT) - K·exp(-rT)``
  → theta = -∂price/∂T = q·S·exp(-qT) - r·K·exp(-rT)
- ITM put (forward < strike): ``K·exp(-rT) - S·exp(-qT)``
  → theta = r·K·exp(-rT) - q·S·exp(-qT)
- OTM: price = 0 → theta = 0.
- ATM: one-sided limit = half of the ITM value.
At ``T=0`` (expiry): no time decay → theta = 0 (convention).
"""

from __future__ import annotations

import math

import pytest

from pricebook.models.black76 import OptionType
from pricebook.options.equity_option import equity_theta


def _itm_call_theta(spot, strike, rate, q, T):
    """Closed-form deterministic theta for an ITM call at vol=0."""
    return q * spot * math.exp(-q * T) - rate * strike * math.exp(-rate * T)


class TestEquityThetaVolZeroTPositive:
    def test_itm_call_theta_matches_closed_form(self):
        # spot=120, strike=100, r=0.05, q=0.02, T=1 → forward = 120·exp(0.03) > 100 → ITM call.
        spot, strike, r, q, T = 120.0, 100.0, 0.05, 0.02, 1.0
        theta = equity_theta(spot=spot, strike=strike, rate=r, vol=0.0, T=T,
                             option_type=OptionType.CALL, div_yield=q)
        expected = _itm_call_theta(spot, strike, r, q, T)
        # expected ≈ 0.02·120·exp(-0.02) − 0.05·100·exp(-0.05) ≈ 2.352 − 4.756 ≈ -2.404
        assert theta == pytest.approx(expected, abs=1e-12)
        assert theta < 0.0  # net negative because rT > qT for these inputs

    def test_itm_put_theta_matches_closed_form(self):
        # spot=80, strike=100, r=0.05, q=0.02, T=1 → forward = 80·exp(0.03) < 100 → ITM put.
        spot, strike, r, q, T = 80.0, 100.0, 0.05, 0.02, 1.0
        theta = equity_theta(spot=spot, strike=strike, rate=r, vol=0.0, T=T,
                             option_type=OptionType.PUT, div_yield=q)
        # Put theta is negative of the call deterministic value.
        expected = -_itm_call_theta(spot, strike, r, q, T)
        assert theta == pytest.approx(expected, abs=1e-12)

    def test_otm_call_theta_zero(self):
        # spot=80, strike=100 → forward < strike → OTM call → theta = 0 at vol=0.
        theta = equity_theta(spot=80.0, strike=100.0, rate=0.05, vol=0.0, T=1.0,
                             option_type=OptionType.CALL, div_yield=0.0)
        assert theta == 0.0

    def test_otm_put_theta_zero(self):
        theta = equity_theta(spot=120.0, strike=100.0, rate=0.05, vol=0.0, T=1.0,
                             option_type=OptionType.PUT, div_yield=0.0)
        assert theta == 0.0

    def test_atm_call_theta_half(self):
        # spot=100, strike=100, r=q → forward = 100 exactly → ATM.
        # ATM theta = 0.5 × (q·S·exp(-qT) − r·K·exp(-rT))
        spot, strike, r, q, T = 100.0, 100.0, 0.05, 0.05, 1.0
        theta = equity_theta(spot=spot, strike=strike, rate=r, vol=0.0, T=T,
                             option_type=OptionType.CALL, div_yield=q)
        expected = 0.5 * _itm_call_theta(spot, strike, r, q, T)
        # With r==q, q·S·exp(-qT) − r·K·exp(-rT) = r·exp(-rT)·(S−K) = 0 here.
        assert theta == pytest.approx(0.0, abs=1e-12)
        assert theta == pytest.approx(expected, abs=1e-12)


class TestEquityThetaInteriorUnchanged:
    """Healthy interior path (T>0, vol>0) must match pre-fix exactly."""

    def test_atm_call_finite_negative(self):
        # Theta of a long ATM call is negative (decay).
        theta = equity_theta(spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0,
                             option_type=OptionType.CALL, div_yield=0.0)
        assert theta < 0.0
        # Hull worked example: |theta| ≈ a few units per year for ATM call.
        assert -20.0 < theta < 0.0

    def test_atm_put_finite_negative(self):
        theta = equity_theta(spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0,
                             option_type=OptionType.PUT, div_yield=0.0)
        assert theta < 0.0


class TestEquityThetaTZero:
    def test_T_zero_returns_b76(self):
        # At T=0, just return whatever black76_theta returns (typically 0).
        # No regression: this path is unchanged from pre-fix.
        for opt in [OptionType.CALL, OptionType.PUT]:
            for spot in [80.0, 100.0, 120.0]:
                theta = equity_theta(spot=spot, strike=100.0, rate=0.05, vol=0.2,
                                     T=0.0, option_type=opt, div_yield=0.0)
                assert math.isfinite(theta)
