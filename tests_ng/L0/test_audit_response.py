"""L0 audit-response oracles (slice 6d) — findings #3 (ICMA frequency guard), #4/#5
(RegularPeriod anchors / short-stub), #6 (WeekendSchedule ordering). Each EXPOSES the finding.
"""

from datetime import date

import pytest

from pricebook_ng.foundation import (
    CouponPeriod,
    DayCountConvention,
    year_fraction,
)

VAL = date(2026, 1, 15)
MID = date(2026, 7, 15)
ICMA = DayCountConvention.ACT_ACT_ICMA


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
