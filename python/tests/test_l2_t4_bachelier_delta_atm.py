"""Regression for L2 Wave-2 audit — `bachelier_delta` at exactly ATM
with ``T <= 0 or vol <= 0`` returned 0 (for both call and put) instead
of the standard ``±0.5·df`` one-sided limit.

Pre-fix:

    if time_to_expiry <= 0 or vol_normal <= 0:
        if option_type == OptionType.CALL:
            return df if forward > strike else 0.0
        return -df if forward < strike else 0.0

The ``else 0.0`` branch caught both ``F < K`` (correct for a call OTM)
AND ``F == K`` (incorrect — ATM-at-expiry should be 0.5·df).  Symmetric
bug for puts.

This was inconsistent with ``black76_delta`` which handles the ATM case
explicitly.  Post-fix `bachelier_delta` now matches the same convention.
"""

from __future__ import annotations

import math

import pytest

from pricebook.models.black76 import (
    OptionType,
    bachelier_delta,
    black76_delta,
)


class TestBachelierDeltaATMAtExpiry:
    def test_call_atm_t_zero_is_half_df(self):
        d = bachelier_delta(forward=100.0, strike=100.0, vol_normal=0.0,
                            time_to_expiry=0.0, df=0.95,
                            option_type=OptionType.CALL)
        assert d == pytest.approx(0.5 * 0.95)

    def test_put_atm_t_zero_is_minus_half_df(self):
        d = bachelier_delta(forward=100.0, strike=100.0, vol_normal=0.0,
                            time_to_expiry=0.0, df=0.95,
                            option_type=OptionType.PUT)
        assert d == pytest.approx(-0.5 * 0.95)

    def test_call_atm_vol_zero(self):
        d = bachelier_delta(forward=100.0, strike=100.0, vol_normal=0.0,
                            time_to_expiry=1.0, df=0.95,
                            option_type=OptionType.CALL)
        assert d == pytest.approx(0.5 * 0.95)


class TestConsistencyWithBlack76Delta:
    """At ATM-at-expiry, `bachelier_delta` and `black76_delta` should
    agree on the ±0.5·df limit since they share the same convention."""

    def test_call_agrees(self):
        params = dict(forward=100.0, strike=100.0, vol=0.0, vol_normal=0.0,
                       time_to_expiry=0.0, df=0.95)
        bach = bachelier_delta(forward=params["forward"], strike=params["strike"],
                                vol_normal=params["vol_normal"],
                                time_to_expiry=params["time_to_expiry"],
                                df=params["df"], option_type=OptionType.CALL)
        b76 = black76_delta(forward=params["forward"], strike=params["strike"],
                             vol=params["vol"],
                             time_to_expiry=params["time_to_expiry"],
                             df=params["df"], option_type=OptionType.CALL)
        assert bach == pytest.approx(b76)

    def test_put_agrees(self):
        bach = bachelier_delta(forward=100.0, strike=100.0, vol_normal=0.0,
                                time_to_expiry=0.0, df=0.95,
                                option_type=OptionType.PUT)
        b76 = black76_delta(forward=100.0, strike=100.0, vol=0.0,
                             time_to_expiry=0.0, df=0.95,
                             option_type=OptionType.PUT)
        assert bach == pytest.approx(b76)


class TestNonATMUnaffected:
    """Non-ATM degenerate cases must remain unchanged from pre-fix."""

    def test_itm_call_returns_df(self):
        d = bachelier_delta(forward=110.0, strike=100.0, vol_normal=0.0,
                            time_to_expiry=0.0, df=0.95,
                            option_type=OptionType.CALL)
        assert d == pytest.approx(0.95)

    def test_otm_call_returns_zero(self):
        d = bachelier_delta(forward=90.0, strike=100.0, vol_normal=0.0,
                            time_to_expiry=0.0, df=0.95,
                            option_type=OptionType.CALL)
        assert d == 0.0

    def test_itm_put_returns_minus_df(self):
        d = bachelier_delta(forward=90.0, strike=100.0, vol_normal=0.0,
                            time_to_expiry=0.0, df=0.95,
                            option_type=OptionType.PUT)
        assert d == pytest.approx(-0.95)
