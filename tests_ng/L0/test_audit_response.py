"""L0 audit-response oracles (slice 6d) — findings #3 (ICMA frequency guard), #4/#5
(RegularPeriod anchors / short-stub), #6 (WeekendSchedule ordering). Each EXPOSES the finding.
"""

from datetime import date

import pytest

from pricebook_ng.foundation import (
    CouponPeriod,
    DayCountConvention,
    Frequency,
    RegularPeriod,
    RollRule,
    ScheduleTerms,
    build_schedule,
    year_fraction,
)

VAL = date(2026, 1, 15)
MID = date(2026, 7, 15)
ICMA = DayCountConvention.ACT_ACT_ICMA


def _terms(stub):
    return ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None), stub=stub)


def test_icma_rejects_frequency_above_12() -> None:
    # finding #3a — freq>12 ⇒ period_months = 12//freq = 0 ⇒ identity step ⇒ INFINITE LOOP (hang)
    cp = CouponPeriod(reference_start=VAL, reference_end=MID, frequency=13)
    with pytest.raises(ValueError):
        year_fraction(VAL, MID, ICMA, coupon_period=cp)


def test_icma_rejects_frequency_not_dividing_12() -> None:
    # finding #3b — freq∈{5,7,8,9,10,11} ⇒ misaligned notional grid ⇒ silently-wrong DCF
    cp = CouponPeriod(reference_start=VAL, reference_end=MID, frequency=5)
    with pytest.raises(ValueError):
        year_fraction(VAL, MID, ICMA, coupon_period=cp)


def test_regular_period_coincident_anchors_raises_up_front() -> None:
    # finding #4 — first_regular == last_regular builds a silent zero-length period today
    anchor = date(2027, 1, 15)
    with pytest.raises(ValueError, match="regular"):
        build_schedule(date(2026, 1, 15), date(2028, 1, 15), _terms(RegularPeriod(anchor, anchor)))


def test_regular_period_reversed_anchors_raises_up_front() -> None:
    # finding #4 — first_regular > last_regular builds a backwards span today
    rev = RegularPeriod(date(2027, 6, 15), date(2027, 1, 15))
    with pytest.raises(ValueError, match="regular"):
        build_schedule(date(2026, 1, 15), date(2028, 1, 15), _terms(rev))
