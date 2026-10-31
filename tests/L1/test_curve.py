"""The curve's own checks — the pieces the tranche-1 closed form exercises only at a flat curve.

A flat curve cannot discriminate an interpolation bug (every rate is the same rate), which is the
fixture trap the predecessor hit twice. So the interpolation is tested on a SLOPED curve, where a
wrong scheme produces a different number.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.foundation import Currency, DayCountConvention, TimeMeasure
from pricebook.market import DiscountCurve

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)


def test_a_flat_curve_is_exp_minus_rt_exactly() -> None:
    curve = DiscountCurve.flat(VAL, 0.04, DC)
    for days in (1, 180, 365, 3650):
        assert curve.df(VAL + timedelta(days=days)) == math.exp(-0.04 * days / 365.0)


def test_the_zero_rate_interpolates_LINEARLY_on_a_SLOPED_curve() -> None:
    """A flat curve returns the same rate everywhere, so it cannot tell linear interpolation from
    any other scheme — or from none. Sloped pillars are what make this a test."""
    curve = DiscountCurve(TM, (0.0, 1.0, 3.0), (0.02, 0.04, 0.05))
    assert curve.zero_rate(0.5) == pytest.approx(0.03)          # midpoint of 2% and 4%
    assert curve.zero_rate(2.0) == pytest.approx(0.045)         # midpoint of 4% and 5%
    assert curve.zero_rate(1.0) == pytest.approx(0.04)          # on the pillar
    assert curve.zero_rate(0.0) == pytest.approx(0.02)


def test_beyond_the_pillars_the_rate_is_held_FLAT_not_extrapolated() -> None:
    curve = DiscountCurve(TM, (0.0, 1.0, 3.0), (0.02, 0.04, 0.05))
    assert curve.zero_rate(10.0) == pytest.approx(0.05)
    assert curve.zero_rate(-1.0) == pytest.approx(0.02)


def test_a_bumped_curve_is_a_NEW_curve_and_the_base_is_untouched() -> None:
    base = DiscountCurve(TM, (0.0, 2.0), (0.03, 0.035))
    bumped = base.bumped(1e-4)
    assert base.zero_rates == (0.03, 0.035)
    assert bumped.zero_rates == pytest.approx((0.0301, 0.0351))
    assert bumped.df(VAL + timedelta(days=365)) < base.df(VAL + timedelta(days=365))


def test_a_malformed_curve_is_REFUSED_at_construction() -> None:
    with pytest.raises(ValueError, match="equal length"):
        DiscountCurve(TM, (0.0, 1.0), (0.02,))
    with pytest.raises(ValueError, match="t=0.0"):
        DiscountCurve(TM, (0.5, 1.0), (0.02, 0.03))
    with pytest.raises(ValueError, match="ascending"):
        DiscountCurve(TM, (0.0, 1.0, 1.0), (0.02, 0.03, 0.04))


def test_the_currency_is_open_not_an_enum() -> None:
    """The scope contract puts EM currencies in scope, so an unlisted code must simply work —
    the predecessor's note that an enum would silently drop BRL is the reason this is a test."""
    assert Currency("BRL").code == "BRL"
    assert Currency("JPY", minor_units=0).minor_units == 0
    assert Currency("USD") == Currency("USD")  # keyed by value
