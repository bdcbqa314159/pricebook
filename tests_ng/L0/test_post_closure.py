"""Post-closure seam oracles (L0) — the gaps BETWEEN closed findings.

From `redesign/independent_audits/POST_CLOSURE.md` / `POST_CLOSURE_FINDINGS.md`: each is a
wrong-answer edge or a serialization seam that falls between two findings each marked FIXED.
Red→green: the failing test is committed before its fix. Nothing here reopens the closure.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.calendars import BusinessDayConvention as BDC
from pricebook_ng.foundation.calendars import JointCalendar
from pricebook_ng.foundation.day_count import Accrual
from pricebook_ng.foundation.day_count import DayCountConvention as DC
from pricebook_ng.foundation.market_calendars import get_calendar
from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.rate_index import (
    AccrualConvention,
    AccrualMethod,
    FixingHistory,
    FixingRule,
    IndexId,
    ObservationStyle,
    RateIndex,
    RfrConvention,
    accrued_rate,
)
from pricebook_ng.foundation.results import PricingResult
from pricebook_ng.foundation.schedule import RollRule
from pricebook_ng.foundation.tenor import Tenor, TenorUnit

USD = Currency.USD


def _rfr(*, lockout=0):
    return RateIndex(
        IndexId("TEST_ON", USD, Tenor(1, TenorUnit.DAY)),
        AccrualConvention(
            DC.ACT_360,
            RollRule(get_calendar("US_GOVERNMENT_SECURITIES"), BDC.MODIFIED_FOLLOWING, eom=False),
        ),
        FixingRule(ObservationStyle.BACKWARD_LOOKING, AccrualMethod.COMPOUNDED, fixing_lag=0),
        RfrConvention(lockout=lockout),
    )


def _series(name, value, start, end):
    days, d = {}, date(start.year, 1, 1)
    while d <= end:
        days[d] = value
        d = date.fromordinal(d.toordinal() + 1)
    return FixingHistory({name: days})


# B1 — a lockout longer than the rate window underflowed `frozen` to a negative index, so Python
# negative-indexing SILENTLY froze rates to early dates (a short stub with a standard lockout).
def test_b1_lockout_longer_than_window_raises():
    idx = _rfr(lockout=5)  # 5 > the window length below
    acc = Accrual(date(2024, 6, 10), date(2024, 6, 12), DC.ACT_360)  # Mon→Wed: 2 overnight days
    fx = _series("TEST_ON", 0.05, date(2024, 6, 1), date(2024, 6, 30))
    with pytest.raises(ValueError, match="lockout"):
        accrued_rate(idx, acc, fx)


# B2 — `clean` subtracted raw amounts, so a cross-currency accrued produced a silently wrong clean.
def test_b2_clean_rejects_cross_currency_accrued():
    r = PricingResult(pv=Money(100.0, Currency.USD), accrued=Money(2.0, Currency.EUR))
    with pytest.raises(TypeError):
        _ = r.clean
    # the same-currency path still works
    ok = PricingResult(pv=Money(102.0, USD), accrued=Money(2.0, USD))
    assert ok.clean == Money(100.0, USD)


# A3 — a JointCalendar's "A+B" identity could not rehydrate: `get_calendar("A+B")` raised, so the
# first serialized XCCY trade (a JointCalendar on its RollRule) could not round-trip.
def test_a3_jointcalendar_round_trips_by_name():
    us, lon = get_calendar("US_GOVERNMENT_SECURITIES"), get_calendar("LONDON")
    jc = JointCalendar(us, lon)
    assert jc.to_dict() == {"calendar": "US_GOVERNMENT_SECURITIES+LONDON"}
    rt = get_calendar(jc.to_dict()["calendar"])
    assert rt == jc
    d = date(2024, 7, 4)  # US holiday, LON business day → not a joint business day
    assert rt.is_business_day(d) is False
    assert rt.is_business_day(d) == jc.is_business_day(d)
