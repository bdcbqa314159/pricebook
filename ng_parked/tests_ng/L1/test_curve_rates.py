"""General-curve rate accessors oracle (L1) — CP-2b #1, unblocks general-curve HW.

Both discount curves gain `zero_rate` (continuously-compounded `-ln P(0,t)/t`) and
`instantaneous_forward` (`f(0,t) = -d/dt ln P(0,t)`) — the capability a general-curve
Hull-White needs where the flat curve used a constant `r0`. Log-linear-in-DF gives a
piecewise-CONSTANT forward per segment (exact, not finite-difference), which integrates
back to `-ln df` at the pillars. Toward parity with quarry `core/discount_curve`.

Oracles: flat curve → constant `rate` for both; a curve built from constant-rate pillars
→ that same constant forward everywhere; a rising curve's segment forward equals the
analytic log-DF slope and its running integral reconstructs `-ln df` at each pillar.
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.calibration.discount_curve import bootstrap_discount_curve
from pricebook_ng.market.discount_curve import DepositQuote, DiscountCurve
from pricebook_ng.market.snapshot import FlatDiscountCurve

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
ACT360 = DC.ACT_360
RATE = 0.03


def _t(d):
    return year_fraction(D0, d, ACT365)


def test_flat_curve_forward_and_zero_are_the_constant_rate():
    flat = FlatDiscountCurve(RATE, D0, ACT365)
    for d in (date(2027, 4, 1), date(2030, 7, 1)):
        assert flat.instantaneous_forward(d) == pytest.approx(RATE)
        assert flat.zero_rate(d) == pytest.approx(RATE)


def test_constant_rate_pillars_give_constant_forward():
    dates = [D0, date(2027, 1, 15), date(2029, 1, 15), date(2031, 1, 15)]
    curve = DiscountCurve(D0, tuple((d, math.exp(-RATE * _t(d))) for d in dates))
    for d in (date(2026, 7, 1), date(2028, 5, 1), date(2030, 9, 1)):
        assert curve.instantaneous_forward(d) == pytest.approx(RATE, abs=1e-12)
        assert curve.zero_rate(d) == pytest.approx(RATE, abs=1e-12)


def _rising_curve():
    deposits = [
        DepositQuote(date(2027, 1, 15), 0.030, ACT360),
        DepositQuote(date(2028, 1, 15), 0.035, ACT360),
        DepositQuote(date(2029, 1, 15), 0.040, ACT360),
    ]
    return bootstrap_discount_curve(D0, deposits, [])


def test_segment_forward_equals_log_df_slope():
    curve = _rising_curve()
    pillars = [d for d, _ in curve.pillars]
    for lo, hi in zip(pillars, pillars[1:]):
        expected = -(math.log(curve.df(hi)) - math.log(curve.df(lo))) / (_t(hi) - _t(lo))
        mid = lo + (hi - lo) / 2
        assert curve.instantaneous_forward(mid) == pytest.approx(expected, abs=1e-12)


def test_forward_integrates_to_minus_log_df():
    curve = _rising_curve()
    pillars = [d for d, _ in curve.pillars]
    integral = 0.0
    for lo, hi in zip(pillars, pillars[1:]):
        mid = lo + (hi - lo) / 2
        integral += curve.instantaneous_forward(mid) * (_t(hi) - _t(lo))
        assert integral == pytest.approx(-math.log(curve.df(hi)), abs=1e-10)


def test_zero_rate_is_minus_log_df_over_t():
    curve = _rising_curve()
    d = date(2028, 6, 1)
    assert curve.zero_rate(d) == pytest.approx(-math.log(curve.df(d)) / _t(d), abs=1e-12)
