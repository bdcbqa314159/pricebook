"""S02 oracle — business-day calendar, schedule generation, BUS/252.

Oracles are hand-computed reference dates/counts (self-consistency against the
convention definitions), exact — dates and integer counts, no tolerance needed.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.calendar import BusinessDayConvention as BDC
from pricebook_ng.foundation.calendar import Calendar
from pricebook_ng.foundation.schedule import Frequency, generate_schedule
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction

WEEKEND_ONLY = Calendar()
NY = Calendar(holidays=frozenset({date(2024, 1, 1)}))  # New Year's Day (a Monday)


# ---- is_business_day ----------------------------------------------------------
def test_is_business_day():
    assert WEEKEND_ONLY.is_business_day(date(2024, 1, 2))       # Tuesday
    assert not WEEKEND_ONLY.is_business_day(date(2024, 1, 6))   # Saturday
    assert not WEEKEND_ONLY.is_business_day(date(2024, 1, 7))   # Sunday
    assert not NY.is_business_day(date(2024, 1, 1))             # holiday
    assert NY.is_business_day(date(2024, 1, 1)) is False


# ---- adjust -------------------------------------------------------------------
def test_adjust_unadjusted_noop():
    d = date(2024, 1, 6)  # Saturday
    assert WEEKEND_ONLY.adjust(d, BDC.UNADJUSTED) == d


def test_adjust_following_and_preceding_over_weekend():
    sat = date(2024, 1, 6)
    assert WEEKEND_ONLY.adjust(sat, BDC.FOLLOWING) == date(2024, 1, 8)   # Monday
    assert WEEKEND_ONLY.adjust(sat, BDC.PRECEDING) == date(2024, 1, 5)   # Friday


def test_adjust_following_over_holiday():
    assert NY.adjust(date(2024, 1, 1), BDC.FOLLOWING) == date(2024, 1, 2)


def test_modified_following_rolls_back_at_month_end():
    # 2024-03-30 Sat, 03-31 Sun; FOLLOWING would cross into April -> roll back.
    assert WEEKEND_ONLY.adjust(date(2024, 3, 30), BDC.MODIFIED_FOLLOWING) == date(2024, 3, 29)


def test_modified_preceding_rolls_forward_at_month_start():
    # 2024-06-01 Sat, 06-02 Sun; PRECEDING would cross into May -> roll forward.
    assert WEEKEND_ONLY.adjust(date(2024, 6, 1), BDC.MODIFIED_PRECEDING) == date(2024, 6, 3)


# ---- business_days_between (start exclusive, end inclusive) --------------------
def test_business_days_between_weekend():
    # Mon 07-08 -> Fri 07-12: Tue,Wed,Thu,Fri = 4
    assert WEEKEND_ONLY.business_days_between(date(2024, 7, 8), date(2024, 7, 12)) == 4


def test_business_days_between_skips_holiday():
    cal = Calendar(holidays=frozenset({date(2024, 7, 10)}))  # Wednesday
    assert cal.business_days_between(date(2024, 7, 8), date(2024, 7, 12)) == 3


# ---- schedule generation ------------------------------------------------------
def test_regular_semi_annual_unadjusted():
    sched = generate_schedule(date(2024, 1, 15), date(2025, 1, 15), Frequency.SEMI_ANNUAL)
    assert sched == [date(2024, 1, 15), date(2024, 7, 15), date(2025, 1, 15)]


def test_regular_quarterly_unadjusted():
    sched = generate_schedule(date(2024, 1, 15), date(2025, 1, 15), Frequency.QUARTERLY)
    assert sched == [
        date(2024, 1, 15), date(2024, 4, 15), date(2024, 7, 15),
        date(2024, 10, 15), date(2025, 1, 15),
    ]


def test_end_of_month_roll():
    # start is EOM (leap Feb) -> rolls stay at EOM
    sched = generate_schedule(
        date(2024, 2, 29), date(2025, 2, 28), Frequency.SEMI_ANNUAL, eom=True
    )
    assert sched == [date(2024, 2, 29), date(2024, 8, 31), date(2025, 2, 28)]


def test_short_front_stub():
    # backward generation leaves a 5-month short first period
    sched = generate_schedule(date(2024, 2, 15), date(2025, 1, 15), Frequency.SEMI_ANNUAL)
    assert sched == [date(2024, 2, 15), date(2024, 7, 15), date(2025, 1, 15)]


def test_business_day_adjusted_schedule():
    # 2024-06-15 is a Saturday -> MODIFIED_FOLLOWING to Monday 06-17
    sched = generate_schedule(
        date(2024, 3, 15), date(2024, 9, 15), Frequency.QUARTERLY,
        calendar=WEEKEND_ONLY, convention=BDC.MODIFIED_FOLLOWING,
    )
    assert date(2024, 6, 17) in sched
    assert date(2024, 6, 15) not in sched


def test_schedule_start_after_end_raises():
    with pytest.raises(ValueError):
        generate_schedule(date(2025, 1, 1), date(2024, 1, 1), Frequency.ANNUAL)


# ---- BUS/252 (now that calendars exist) ---------------------------------------
def test_bus_252_year_fraction():
    cal = Calendar(holidays=frozenset({date(2024, 7, 10)}))  # Wednesday holiday
    yf = year_fraction(date(2024, 7, 8), date(2024, 7, 12), DC.BUS_252, calendar=cal)
    assert yf == pytest.approx(3 / 252, abs=1e-12)


def test_bus_252_requires_calendar():
    with pytest.raises(ValueError):
        year_fraction(date(2024, 7, 8), date(2024, 7, 12), DC.BUS_252)
