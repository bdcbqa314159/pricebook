"""Day-count oracles (L0) — Topic 0 Slice 2.

Published references: ISDA 2006 §4.16 worked examples, ICMA Rule 251 (incl. the
UST-coupon-=-exactly-2.0000 regression that `strict_icma=False` used to break), the
30U/360 US-SIA February edge cases, and the three gap conventions (ACT/365L, 30E/360
ISDA, NL/365). Calendar is passed in — BUS/252 with no calendar raises (the quarry
silently defaulted to São Paulo); ICMA with no coupon anchors raises (no ACT/365F
fallback). Content mined from `core/day_count.py`; `strict_icma` is deleted.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.day_count import CouponPeriod
from pricebook_ng.foundation.day_count import DayCountConvention as DC
from pricebook_ng.foundation.day_count import year_fraction as yf
from pricebook_ng.foundation.market_calendars import get_calendar

APPROX = 1e-12


def _days(a, b):
    return (b - a).days


# ── actual/x ──
def test_act_360_and_365f():
    a, b = date(2024, 1, 1), date(2024, 7, 1)
    assert yf(a, b, DC.ACT_360) == pytest.approx(_days(a, b) / 360.0, abs=APPROX)
    assert yf(a, b, DC.ACT_365_FIXED) == pytest.approx(_days(a, b) / 365.0, abs=APPROX)


# ── 30U/360 (US SIA) — the February rules ──
def test_thirty_360_us_sia_end_of_feb():
    # start = last day of (non-leap) Feb → d1=30; end 08-31 → d2=30. 180/360 = 0.5.
    assert yf(date(2023, 2, 28), date(2023, 8, 31), DC.THIRTY_360) == pytest.approx(0.5, abs=APPROX)
    # start 01-31 → d1=30; end 07-31 → d2=30. 0.5.
    assert yf(date(2024, 1, 31), date(2024, 7, 31), DC.THIRTY_360) == pytest.approx(0.5, abs=APPROX)


def test_thirty_e_360_has_no_feb_rule():
    # 30E/360 does NOT apply the Feb rule → differs from 30U on a Feb-spanning start.
    assert yf(date(2023, 2, 28), date(2023, 8, 31), DC.THIRTY_E_360) == pytest.approx(182 / 360.0, abs=APPROX)
    assert yf(date(2024, 1, 31), date(2024, 7, 31), DC.THIRTY_E_360) == pytest.approx(0.5, abs=APPROX)


# ── ACT/ACT ISDA — the §4.16 worked example ──
def test_act_act_isda_worked_example():
    # ISDA 2006 §4.16: 2003-11-01 → 2004-05-01 = 61/365 + 121/366.
    expected = 61 / 365.0 + 121 / 366.0
    assert yf(date(2003, 11, 1), date(2004, 5, 1), DC.ACT_ACT_ISDA) == pytest.approx(expected, abs=APPROX)


# ── ACT/ACT ICMA — Rule 251, the UST 2.0000 regression ──
def test_act_act_icma_ust_coupon_is_exactly_half():
    # A semi-annual coupon period reprices to EXACTLY 0.5 (→ a 4% UST coupon = 2.0000,
    # not the 1.9836/2.0164 the silent ACT/365F fallback produced).
    rs, re = date(2024, 2, 15), date(2024, 8, 15)
    cp = CouponPeriod(reference_start=rs, reference_end=re, frequency=2)
    assert yf(rs, re, DC.ACT_ACT_ICMA, coupon_period=cp) == 0.5
    assert 100.0 * 0.04 * yf(rs, re, DC.ACT_ACT_ICMA, coupon_period=cp) == 2.0


def test_act_act_icma_requires_anchors():
    with pytest.raises(ValueError):
        yf(date(2024, 2, 15), date(2024, 8, 15), DC.ACT_ACT_ICMA)  # no coupon_period


# ── BUS/252 — calendar required, no São Paulo default ──
def test_bus_252_needs_a_calendar():
    sp = get_calendar("SAO_PAULO")
    a, b = date(2024, 1, 2), date(2024, 1, 31)
    val = yf(a, b, DC.BUS_252, calendar=sp)
    assert 0.0 < val < 0.15  # ~20 business days / 252
    with pytest.raises(ValueError):
        yf(a, b, DC.BUS_252)  # no calendar → raises (quarry silently used São Paulo)


# ── gap conventions ──
def test_act_365l_leap_denominator():
    # period containing 29 Feb → /366; otherwise /365.
    a, b = date(2024, 2, 1), date(2024, 3, 1)      # contains 2024-02-29
    assert yf(a, b, DC.ACT_365L) == pytest.approx(_days(a, b) / 366.0, abs=APPROX)
    a2, b2 = date(2023, 2, 1), date(2023, 3, 1)    # no leap day
    assert yf(a2, b2, DC.ACT_365L) == pytest.approx(_days(a2, b2) / 365.0, abs=APPROX)


def test_nl_365_excludes_leap_day():
    a, b = date(2024, 2, 1), date(2024, 3, 1)      # 29 actual days, minus 29-Feb
    assert yf(a, b, DC.NL_365) == pytest.approx(28 / 365.0, abs=APPROX)
    assert yf(a, b, DC.ACT_365_FIXED) == pytest.approx(29 / 365.0, abs=APPROX)  # contrast


def test_thirty_e_360_isda_last_day_and_termination():
    # d1 last day of Feb → 30; d2 = 31 (last day Aug) → 30. 0.5.
    assert yf(date(2023, 2, 28), date(2023, 8, 31), DC.THIRTY_E_360_ISDA) == pytest.approx(0.5, abs=APPROX)
    # termination exception: end = last day of Feb AND is_final → d2 NOT set to 30.
    rs, re = date(2023, 8, 31), date(2024, 2, 29)
    final = CouponPeriod(reference_start=rs, reference_end=re, frequency=2, is_final=True)
    non_final = yf(rs, re, DC.THIRTY_E_360_ISDA)                       # d2 30-31→30: 180/360
    term = yf(rs, re, DC.THIRTY_E_360_ISDA, coupon_period=final)       # d2 stays 29: 179/360
    assert non_final == pytest.approx(180 / 360.0, abs=APPROX)
    assert term == pytest.approx(179 / 360.0, abs=APPROX)


def test_act_365l_is_frequency_dependent():
    # ISDA §4.16(i): a semi-annual period ENDING in a leap year but NOT containing 29 Feb
    # uses 366; the annual rule (or default) would use 365 — the distinguishing case.
    a, b = date(2024, 3, 1), date(2024, 9, 1)      # 2024 leap; 29 Feb is before `a`
    semi = CouponPeriod(reference_start=a, reference_end=b, frequency=2)
    ann = CouponPeriod(reference_start=a, reference_end=b, frequency=1)
    assert yf(a, b, DC.ACT_365L, coupon_period=semi) == pytest.approx(_days(a, b) / 366.0, abs=APPROX)
    assert yf(a, b, DC.ACT_365L, coupon_period=ann) == pytest.approx(_days(a, b) / 365.0, abs=APPROX)


# ── S11: completeness vs ISDA 2006 — 1/1 and ACT/ACT AFB ──
def test_one_one_is_always_one():
    assert yf(date(2024, 3, 5), date(2027, 9, 20), DC.ONE_ONE) == 1.0   # 1/1 is always 1


def test_act_act_afb():
    # < 1 year, leap period (contains 29 Feb 2024) → actual/366
    a, b = date(2024, 1, 1), date(2024, 7, 1)
    assert yf(a, b, DC.ACT_ACT_AFB) == pytest.approx(_days(a, b) / 366.0, abs=1e-12)
    # 1.5 years → 1 whole year + a non-leap stub /365
    a2, b2 = date(2023, 1, 1), date(2024, 7, 1)
    assert yf(a2, b2, DC.ACT_ACT_AFB) == pytest.approx(1.0 + 181 / 365.0, abs=1e-12)
