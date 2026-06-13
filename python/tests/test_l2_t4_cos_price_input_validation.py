"""Regression for L2 Wave-2 audit — `cos_price` had three degenerate
inputs that crashed with opaque exceptions deep inside the formula.

Pre-fix:
- ``L=0``: ``b - a = 0`` (because the truncation half-width
  ``L·sqrt(c2) = 0``), then ``ZeroDivisionError`` inside the V_k
  recursion's ``2.0 / (b - a)`` factor.
- ``spot <= 0``: ``math.log(spot / strike)`` raised ``math domain
  error`` with no pointer at which input was bad.
- ``strike <= 0``: ``spot / strike`` raised ``ZeroDivisionError`` (zero)
  or wrong sign (negative), depending on the value.

Post-fix all three raise ``ValueError`` upfront with a clear message.
"""

from __future__ import annotations

import pytest

from pricebook.models.black76 import OptionType
from pricebook.models.cos_method import bs_char_func, cos_price


class TestSpotValidation:
    def test_zero_spot_raises(self):
        with pytest.raises(ValueError, match="spot must be > 0"):
            cos_price(
                char_func=bs_char_func(0.05, 0.0, 0.2, 1.0),
                spot=0.0, strike=100.0, rate=0.05, T=1.0,
            )

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="spot must be > 0"):
            cos_price(
                char_func=bs_char_func(0.05, 0.0, 0.2, 1.0),
                spot=-10.0, strike=100.0, rate=0.05, T=1.0,
            )


class TestStrikeValidation:
    def test_zero_strike_raises(self):
        with pytest.raises(ValueError, match="strike must be > 0"):
            cos_price(
                char_func=bs_char_func(0.05, 0.0, 0.2, 1.0),
                spot=100.0, strike=0.0, rate=0.05, T=1.0,
            )

    def test_negative_strike_raises(self):
        with pytest.raises(ValueError, match="strike must be > 0"):
            cos_price(
                char_func=bs_char_func(0.05, 0.0, 0.2, 1.0),
                spot=100.0, strike=-50.0, rate=0.05, T=1.0,
            )


class TestLValidation:
    def test_zero_L_raises(self):
        with pytest.raises(ValueError, match="L must be > 0"):
            cos_price(
                char_func=bs_char_func(0.05, 0.0, 0.2, 1.0),
                spot=100.0, strike=100.0, rate=0.05, T=1.0, L=0.0,
            )

    def test_negative_L_raises(self):
        with pytest.raises(ValueError, match="L must be > 0"):
            cos_price(
                char_func=bs_char_func(0.05, 0.0, 0.2, 1.0),
                spot=100.0, strike=100.0, rate=0.05, T=1.0, L=-5.0,
            )


class TestHealthyPathUnchanged:
    def test_atm_call_prices_correctly(self):
        # ATM call on BS: should price near 10.45 for r=0.05, vol=0.2, T=1.
        p = cos_price(
            char_func=bs_char_func(0.05, 0.0, 0.2, 1.0),
            spot=100.0, strike=100.0, rate=0.05, T=1.0,
            option_type=OptionType.CALL, N=128,
        )
        assert p == pytest.approx(10.45, abs=0.05)
