"""Regression for L2 Wave-2 audit — `equity_rho` returned 0 in the
``T <= 0 or vol <= 0`` branch, silently dropping the deterministic
limit at vol=0 with T>0.

At vol=0, T>0 the price is the discounted intrinsic of the FORWARD:
- ITM call (forward > strike): ``S·exp(-qT) − K·exp(-rT)``
  → ``rho = T·K·exp(-rT)`` (positive)
- ITM put (forward < strike): ``K·exp(-rT) − S·exp(-qT)``
  → ``rho = -T·K·exp(-rT)`` (negative)
- OTM (either side): price = 0 → rho = 0.
- ATM (forward == strike): one-sided limit ``±0.5·T·K·exp(-rT)``.

Pre-fix all four cases returned 0.  Post-fix preserves T=0 (still 0,
no time for rate to act) but handles the vol=0/T>0 case correctly.
"""

from __future__ import annotations

import math

import pytest

from pricebook.models.black76 import OptionType
from pricebook.options.equity_option import equity_rho


class TestEquityRhoVolZeroTPositive:
    def test_itm_call_rho_positive(self):
        # spot=120, strike=100, r=0.05, q=0, T=1 → forward=120·e^0.05≈126.2 > 100
        rho = equity_rho(spot=120.0, strike=100.0, rate=0.05, vol=0.0, T=1.0,
                         option_type=OptionType.CALL, div_yield=0.0)
        # rho = T·K·exp(-rT) = 1 · 100 · exp(-0.05) ≈ 95.12
        expected = 1.0 * 100.0 * math.exp(-0.05)
        assert rho == pytest.approx(expected, abs=1e-12)

    def test_itm_put_rho_negative(self):
        # spot=80, strike=100, r=0.05, q=0, T=1 → forward≈84.1 < 100
        rho = equity_rho(spot=80.0, strike=100.0, rate=0.05, vol=0.0, T=1.0,
                         option_type=OptionType.PUT, div_yield=0.0)
        expected = -1.0 * 100.0 * math.exp(-0.05)
        assert rho == pytest.approx(expected, abs=1e-12)

    def test_otm_call_rho_zero(self):
        # spot=80, strike=100 → forward<strike → OTM call → rho = 0
        rho = equity_rho(spot=80.0, strike=100.0, rate=0.05, vol=0.0, T=1.0,
                         option_type=OptionType.CALL, div_yield=0.0)
        assert rho == 0.0

    def test_otm_put_rho_zero(self):
        rho = equity_rho(spot=120.0, strike=100.0, rate=0.05, vol=0.0, T=1.0,
                         option_type=OptionType.PUT, div_yield=0.0)
        assert rho == 0.0

    def test_atm_call_rho_half(self):
        # spot=100, strike=100, r=q=0 → forward=100 exactly → ATM
        rho = equity_rho(spot=100.0, strike=100.0, rate=0.05, vol=0.0, T=1.0,
                         option_type=OptionType.CALL, div_yield=0.05)
        # forward = 100·exp(0) = 100 == strike
        expected = 0.5 * 1.0 * 100.0 * math.exp(-0.05)
        assert rho == pytest.approx(expected, abs=1e-12)


class TestEquityRhoTZero:
    def test_T_zero_returns_zero(self):
        """At expiry, no time for rate to act → rho = 0."""
        for opt in [OptionType.CALL, OptionType.PUT]:
            for spot in [80.0, 100.0, 120.0]:
                rho = equity_rho(spot=spot, strike=100.0, rate=0.05, vol=0.2,
                                 T=0.0, option_type=opt, div_yield=0.0)
                assert rho == 0.0


class TestEquityRhoInteriorUnchanged:
    """Healthy interior path (T>0, vol>0) must be identical to pre-fix."""

    def test_atm_call_finite_positive(self):
        rho = equity_rho(spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0,
                         option_type=OptionType.CALL, div_yield=0.0)
        # Black-Scholes rho ATM call with these params is ~50.
        assert 40.0 < rho < 60.0
