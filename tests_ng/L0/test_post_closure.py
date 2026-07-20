"""Post-closure seam oracles (L0) — the gaps BETWEEN closed findings.

From `redesign/independent_audits/POST_CLOSURE.md` / `POST_CLOSURE_FINDINGS.md`: each is a
wrong-answer edge or a serialization seam that falls between two findings each marked FIXED.
Red→green: the failing test is committed before its fix. Nothing here reopens the closure.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.calendars import BusinessDayConvention as BDC
from pricebook_ng.foundation.calendars import (
    Calendar,
    JointCalendar,
    Observance,
    Weekend,
    fixed,
)
from pricebook_ng.foundation.day_count import Accrual
from pricebook_ng.foundation.day_count import DayCountConvention as DC
from pricebook_ng.foundation.market_calendars import get_calendar
from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.rate_basis import Compounding, convert_rate
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


# ── Ledger tightening (standing rule: no condition-driven deferrals before F1) ──
# Each of these was a Tier-4 deferral with a condition-driven trigger (no owning topic), so it
# fails the rule and is closed now with a test.


# AC-T4.7 — a December holiday observed FORWARD into January was missed: is_holiday checked
# year and year+1 but not year-1.
def test_t4_7_holiday_observed_from_prior_december_into_january():
    # fixed 31 Dec; 2023-12-31 is a Sunday → observed forward to Monday 2024-01-01
    cal = Calendar("YEAR_END_TEST", (fixed(12, 31),), observance=Observance.NEXT_WORKING_DAY)
    assert cal.is_holiday(date(2024, 1, 1))


# AC-T4.8 — observe() hardcoded Sat/Sun; for a FRI_SAT calendar a Friday holiday (a weekend day
# there) under a mondayising regime was left unshifted.
def test_t4_8_observe_parameterises_off_the_weekend_rule():
    cal = Calendar(
        "FRISAT_TEST", (fixed(1, 5),), weekend=Weekend.FRI_SAT,
        observance=Observance.NEXT_WORKING_DAY,
    )
    # 2024-01-05 is a Friday (FRI_SAT weekend) → next working day is Sunday 2024-01-07
    assert cal.observe(date(2024, 1, 5)) == date(2024, 1, 7)


# AC-T4.9 — NEAREST tie-break rolled backward; QuantLib / market practice roll forward.
def test_t4_9_nearest_rolls_forward_on_a_tie():
    from pricebook_ng.foundation.calendars import BusinessDayConvention as BDC
    ny = get_calendar("US_GOVERNMENT_SECURITIES")
    # 2024-07-04 (Thu holiday): prev bd Jul 3, next bd Jul 5, equidistant → roll FORWARD
    assert ny.adjust(date(2024, 7, 4), BDC.NEAREST) == date(2024, 7, 5)


# AC-T4.11 — convert_rate on a rate below -100% gave a bare "math domain error" (or a silent
# complex on a periodic basis); reject with a message naming the input.
def test_t4_11_convert_rate_rejects_a_rate_below_minus_100pct():
    with pytest.raises(ValueError, match="growth factor|per-period|100%"):
        convert_rate(-1.5, 1.0, Compounding.ANNUAL, Compounding.CONTINUOUS)


# AC-T4.17 — the weekend rule was time-invariant; add since= support so a change is expressible
# (shape only — no real market's future rule is guessed here).
def test_t4_17_weekend_rule_can_change_over_time():
    from pricebook_ng.foundation.calendars import WeekendSchedule
    sched = WeekendSchedule(((0, Weekend.FRI_SAT), (2020, Weekend.SAT_SUN)))
    cal = Calendar("WEEKEND_SWITCH_TEST", (), weekend=sched)
    # before 2020 the weekend is Fri/Sat
    assert cal.is_weekend(date(2019, 1, 4))       # Friday
    assert not cal.is_weekend(date(2019, 1, 6))   # Sunday — a business day under FRI_SAT
    # from 2020 the weekend is Sat/Sun
    assert cal.is_weekend(date(2020, 1, 5))       # Sunday
    assert not cal.is_weekend(date(2020, 1, 3))   # Friday — a business day under SAT_SUN
