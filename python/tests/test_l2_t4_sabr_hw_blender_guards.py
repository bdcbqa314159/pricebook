"""Regression for L2 Wave-2 audit — `price_swaption_sabr_hw` guards.

Two issues:

1. `blend_half_life = 0` divided by zero deep inside the pricer
   (``math.exp(-T / 0.0)`` raises ``ZeroDivisionError`` with no diagnostic
   context).  Post-fix this raises a clear ``ValueError`` upfront.

2. When BOTH SABR and HW vols returned non-positive values, the pre-fix
   fallback silently substituted a hard-coded 1% volatility and produced
   an essentially arbitrary price.  Post-fix this raises a clear
   ``ValueError`` so the caller can diagnose the upstream calibration
   failure.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.schedule import Frequency
from pricebook.options.swaption import (
    Swaption,
    SwaptionType,
    price_swaption_sabr_hw,
)


REF = date(2024, 1, 1)


def _flat_curve(rate: float = 0.04) -> DiscountCurve:
    tenors = [0.5, 1, 2, 5, 10, 20]
    dates = [REF + timedelta(days=int(t * 365)) for t in tenors]
    return DiscountCurve(REF, dates, [math.exp(-rate * t) for t in tenors])


def _make_swaption() -> Swaption:
    return Swaption(
        expiry=REF + timedelta(days=365 * 2),
        swap_end=REF + timedelta(days=365 * 7),
        strike=0.04,
        swaption_type=SwaptionType.PAYER,
        notional=1_000_000.0,
        fixed_frequency=Frequency.SEMI_ANNUAL,
        float_frequency=Frequency.QUARTERLY,
        fixed_day_count=DayCountConvention.THIRTY_360,
        float_day_count=DayCountConvention.ACT_360,
    )


class _SABRCubeReturningZero:
    def vol(self, *_a, **_kw) -> float:
        return 0.0


class _SABRCubeReturningPositive:
    def vol(self, *_a, **_kw) -> float:
        return 0.20


class _DummyHWModelFails:
    """A HW model whose calibrated params yield zero vol on the test strike."""
    a = 1e-6  # near-zero mean reversion + tiny sigma → near-zero implied vol
    sigma = 1e-10


class _DummyHWModelOK:
    a = 0.05
    sigma = 0.01


def _make_deep_otm_swaption() -> Swaption:
    """Strike so far OTM that HW pricer returns ≤ 0 → implied vol = 0."""
    return Swaption(
        expiry=REF + timedelta(days=365 * 2),
        swap_end=REF + timedelta(days=365 * 7),
        strike=10.0,  # 1000% strike — deeply OTM payer
        swaption_type=SwaptionType.PAYER,
        notional=1_000_000.0,
        fixed_frequency=Frequency.SEMI_ANNUAL,
        float_frequency=Frequency.QUARTERLY,
        fixed_day_count=DayCountConvention.THIRTY_360,
        float_day_count=DayCountConvention.ACT_360,
    )


class TestBlendHalfLifeGuard:
    def test_zero_blend_half_life_raises_value_error(self):
        swn = _make_swaption()
        with pytest.raises(ValueError, match="blend_half_life must be > 0"):
            price_swaption_sabr_hw(
                swn, _SABRCubeReturningPositive(), _DummyHWModelOK(),
                _flat_curve(), blend_half_life=0.0,
            )

    def test_negative_blend_half_life_raises_value_error(self):
        swn = _make_swaption()
        with pytest.raises(ValueError, match="blend_half_life must be > 0"):
            price_swaption_sabr_hw(
                swn, _SABRCubeReturningPositive(), _DummyHWModelOK(),
                _flat_curve(), blend_half_life=-1.0,
            )


class TestBothVolsFailRaises:
    def test_both_vols_non_positive_raises_value_error(self):
        """Pre-fix the function silently used 1% vol and returned a wrong
        price.  Post-fix it raises a clear ValueError.

        Force both vols to ≤ 0 by combining a SABR cube returning 0 with a
        deeply OTM strike that makes the HW pricer return 0 → implied vol 0.
        """
        swn = _make_deep_otm_swaption()
        with pytest.raises(ValueError, match="both SABR and HW vols"):
            price_swaption_sabr_hw(
                swn, _SABRCubeReturningZero(), _DummyHWModelOK(),
                _flat_curve(), blend_half_life=5.0,
            )


class TestHappyPathStillWorks:
    def test_positive_sabr_only(self):
        """If SABR returns a positive vol but HW fails, fall back to SABR
        alone (no error)."""
        swn = _make_swaption()
        price = price_swaption_sabr_hw(
            swn, _SABRCubeReturningPositive(), _DummyHWModelFails(),
            _flat_curve(), blend_half_life=5.0,
        )
        assert math.isfinite(price)
        assert price > 0
