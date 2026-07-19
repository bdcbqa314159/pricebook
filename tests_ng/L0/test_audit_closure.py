"""Foundation-audit-closure oracles (L0) — Tier 1 + Tier 2 findings.

Each counterexample from `redesign/independent_audits/AUDIT.md` is a failing test committed
BEFORE its fix (red→green). Expected values are the audit's hand calculations, verified
against ISDA/ANModel/ARRC where cited. Part A rulings apply: A1 (CONTINUE_SLOPE in the
interpolation's own space), A2 (one business-day primitive, `[start, end)`).
"""

from datetime import date

import pytest

from pricebook_ng.foundation.day_count import DayCountConvention as DC
from pricebook_ng.foundation.day_count import business_days_between, year_fraction
from pricebook_ng.foundation.interpolation import (
    Extrapolation,
    ExtrapolationEnds,
    Interpolation,
    interpolate,
)
from pricebook_ng.foundation.market_calendars import get_calendar
from pricebook_ng.foundation.rate_index import _overnight_days
from pricebook_ng.foundation.schedule import (
    Frequency,
    RollRule,
    ScheduleTerms,
    StubType,
    build_schedule,
)

NY = get_calendar("US_GOVERNMENT_SECURITIES")
TOKYO = get_calendar("TOKYO")


# ── 1.1 ACT/ACT AFB undercounts leap-to-leap (silent wrongness) ──
def test_1_1_afb_leap_to_leap_is_exactly_whole_years():
    assert year_fraction(date(2004, 2, 29), date(2008, 2, 29), DC.ACT_ACT_AFB) == 4.0
    assert year_fraction(date(2016, 2, 29), date(2020, 2, 29), DC.ACT_ACT_AFB) == 4.0


# ── 1.2 LOG_LINEAR + CONTINUE_SLOPE must not produce negative DFs (A1, silent wrongness) ──
def test_1_2_log_linear_continue_slope_stays_positive():
    xs, ys = [1.0, 2.0, 5.0], [0.97, 0.94, 0.80]
    ends = ExtrapolationEnds(right=Extrapolation.CONTINUE_SLOPE)
    df30 = interpolate(xs, ys, 30.0, Interpolation.LOG_LINEAR, ends)
    assert df30 > 0.0                              # bug returns −0.275
    assert df30 == pytest.approx(0.2086, abs=1e-3)  # log-space continuation


# ── 1.3 BUS/252 and CDI count the same window: one primitive, [start, end) (A2) ──
def test_1_3_business_days_between_is_start_inclusive_end_exclusive():
    # start is a business day, end is a holiday → [start,end) counts start, (start,end] would not
    start, end = date(2023, 12, 22), date(2023, 12, 25)  # Fri → Christmas Mon
    n = business_days_between(start, end, NY)
    assert n == 1                                  # {Dec 22}; the (start,end] bug gives 0
    # the CDI accrual window (`_overnight_days`, [start,end)) must agree exactly
    assert n == len(_overnight_days(start, end, NY))


# ── 1.4 Backward schedule: roll-day drift + EOM keyed on the wrong end (silent wrongness) ──
def test_1_4a_backward_rollday_does_not_drift():
    # end May 31, quarterly, backward, NO eom snap: interior = end − k·3M from the anchor,
    # so Nov 30 (May 31 − 6M, clamped) — not Nov 29 that iterative stepping drifts to via Feb 29.
    terms = ScheduleTerms(
        frequency=Frequency.QUARTERLY, roll=RollRule(eom=False), stub=StubType.SHORT_FRONT
    )
    s = build_schedule(date(2023, 5, 31), date(2024, 5, 31), terms)
    assert date(2023, 11, 30) in s.unadjusted      # bug drifts to Nov 29
    assert date(2023, 8, 31) in s.unadjusted        # bug drifts to Aug 29


def test_1_4b_backward_eom_keys_on_the_maturity_anchor():
    # end 2024-06-30 is a month-end; EOM must anchor on it (backward), interiors are month-ends
    terms = ScheduleTerms(frequency=Frequency.MONTHLY, roll=RollRule(eom=True), stub=StubType.SHORT_FRONT)
    s = build_schedule(date(2023, 1, 15), date(2024, 6, 30), terms)
    assert date(2024, 5, 31) in s.unadjusted       # bug (keyed on non-EOM start) gives May 30


# ── 1.5 Furikae substitution must be deterministic and walk forward correctly ──
def test_1_5_furikae_golden_week_substitute_deterministic():
    # 2020: Constitution Day May 3 (Sun) → substitute walks past May 4/5 holidays to May 6 (Wed)
    assert not TOKYO.is_business_day(date(2020, 5, 6))
    # determinism: the substitute set is independent of holiday iteration order
    from pricebook_ng.foundation.calendars import _furikae_substitutes
    hols = {date(2020, 5, 3), date(2020, 5, 4), date(2020, 5, 5)}
    assert _furikae_substitutes(hols) == _furikae_substitutes(set(reversed(sorted(hols))))
    assert date(2020, 5, 6) in _furikae_substitutes(hols)


LON = get_calendar("LONDON")


# ── 2.1 SOFR is plain compounded-in-arrears — no observation shift (silent wrongness) ──
def test_2_1_sofr_has_no_observation_shift():
    from pricebook_ng.foundation.rate_index import get_rate_index
    sofr = get_rate_index("SOFR")
    assert sofr.rfr.observation_shift == 0     # ISDA SOFR OIS: plain compounded, bug had shift=2
    assert sofr.rfr.payment_delay == 2
    # the LIBOR fallback correctly KEEPS shift=2 (Bloomberg/ISDA fallback)
    assert get_rate_index("USD_LIBOR_3M_FALLBACK").rfr.observation_shift == 2


# ── 2.2 SIFMA: Good Friday closed (no SOFR fixing); Saturday holidays NOT shifted to Friday ──
def test_2_2_sifma_good_friday_is_a_holiday():
    assert NY.is_holiday(date(2024, 3, 29))    # Good Friday 2024 — SOFR does not publish
    assert NY.is_holiday(date(2025, 4, 18))    # Good Friday 2025


def test_2_2_sifma_saturday_new_year_does_not_shift_to_friday():
    # New Year 2022-01-01 is a Saturday; SIFMA/Treasury/Fed were OPEN Fri 2021-12-31
    assert NY.is_business_day(date(2021, 12, 31))


# ── 2.3 LONDON 2022–23 one-off bank holidays (historical SONIA accuracy) ──
def test_2_3_london_one_off_holidays():
    assert LON.is_holiday(date(2022, 6, 2))    # Platinum Jubilee (moved Spring BH)
    assert LON.is_holiday(date(2022, 6, 3))    # Platinum Jubilee extra day
    assert LON.is_holiday(date(2022, 9, 19))   # State funeral of Elizabeth II
    assert LON.is_holiday(date(2023, 5, 8))    # Coronation of Charles III
    assert not LON.is_holiday(date(2022, 5, 30))  # the Spring BH was MOVED away from here


# ── 2.4 Tokyo equinoxes are astronomical, not hardcoded (TONA dates wrong this year) ──
def test_2_4_tokyo_equinoxes_are_astronomical():
    assert TOKYO.is_holiday(date(2024, 3, 20))      # Vernal Equinox 2024 (astronomical; was fixed 3/21)
    assert not TOKYO.is_holiday(date(2024, 3, 21))  # the old hardcoded date is no longer a holiday
    assert TOKYO.is_holiday(date(2024, 9, 22))      # Autumnal Equinox 2024 (was fixed 9/23)
    # 2024-09-22 is a Sunday, so 09-23 is its furikae substitute (also a holiday) — expected


# ── Phase 3 (structural) ──
from pricebook_ng.foundation.calendars import BusinessDayConvention as BDC


# 3.2 JointCalendar is a full calendar (adjust / add_business_days / identity via CalendarProtocol)
def test_3_2_joint_calendar_is_a_full_calendar():
    from pricebook_ng.foundation.calendars import CalendarProtocol, JointCalendar
    joint = JointCalendar(NY, LON)
    assert isinstance(joint, CalendarProtocol)          # runtime_checkable
    assert joint.identity                                # has an identity
    xmas = date(2024, 12, 25)                            # holiday in both
    assert joint.adjust(xmas, BDC.FOLLOWING) == date(2024, 12, 27)   # skip 25(both)+26(LON boxing)
    assert joint.add_business_days(date(2024, 12, 24), 1) == date(2024, 12, 27)
    # a cross-currency schedule keyed on a JointCalendar builds (was a runtime AttributeError)
    terms = ScheduleTerms(frequency=Frequency.QUARTERLY, roll=RollRule(calendar=joint))
    s = build_schedule(date(2024, 1, 15), date(2025, 1, 15), terms)
    assert len(s.adjusted) == len(s.unadjusted)


# 3.3 registries are open (public register_*) and raise on conflicting re-registration
def test_3_3_registries_open_and_conflict_raises():
    from pricebook_ng.foundation.calendars import Calendar, fixed
    from pricebook_ng.foundation.market_calendars import (
        get_calendar,
        register_calendar,
        temporary_calendar,
    )
    from pricebook_ng.foundation.money import register_currency
    from pricebook_ng.foundation.rate_index import get_rate_index, register_rate_index

    cal = Calendar("TEST_XYZ_CAL", (fixed(1, 1),))
    with temporary_calendar(cal):                   # public registration, isolated
        assert get_calendar("TEST_XYZ_CAL") is cal
        with pytest.raises(ValueError):             # conflicting identity raises
            register_calendar(Calendar("TEST_XYZ_CAL", ()))
    with pytest.raises(ValueError):                 # removed after the block
        get_calendar("TEST_XYZ_CAL")
    # every register_* refuses a conflicting re-registration (was a silent overwrite)
    with pytest.raises(ValueError):
        register_currency("USD")
    with pytest.raises(ValueError):
        register_rate_index(get_rate_index("SOFR"))


# Q1: the USD calendar is named for what it IS (government securities); SOFR binds explicitly
def test_3_q1_sofr_binds_to_named_government_securities_calendar():
    from pricebook_ng.foundation.market_calendars import get_calendar
    from pricebook_ng.foundation.rate_index import get_rate_index
    assert get_rate_index("SOFR").accrual.roll.calendar is get_calendar("US_GOVERNMENT_SECURITIES")
    with pytest.raises(ValueError):
        get_calendar("NEW_YORK_SIFMA")   # the generic name is gone — the "one USD calendar" trap disarmed


# 3.5 (discharged in 3b): ACT/ACT ICMA long first coupon computes Rule 251.2 (was NG-DEFER-1's raise)
def test_3_5_act_act_icma_long_first_coupon():
    # PUBLISHED oracle — ISDA 2006 §4.16 Actual/Actual (ICMA) long-first-coupon worked example:
    # a semi-annual bond (coupons 15 Jan / 15 Jul), interest period 15 Aug 2002 → 15 Jul 2003,
    # split into the notional periods [15 Jul 2002, 15 Jan 2003] (184d) and
    # [15 Jan 2003, 15 Jul 2003] (181d):  153/(2·184) + 181/(2·181) = 0.9157608695652174.
    from pricebook_ng.foundation.day_count import CouponPeriod
    cp = CouponPeriod(reference_start=date(2003, 1, 15), reference_end=date(2003, 7, 15), frequency=2)
    dcf = year_fraction(date(2002, 8, 15), date(2003, 7, 15), DC.ACT_ACT_ICMA, coupon_period=cp)
    assert dcf == pytest.approx(0.9157608695652174, abs=1e-12)


# 3.7 _denominator raises on unsupported day counts (no silent 360 fallback)
def test_3_7_denominator_raises_on_unsupported_day_count():
    from pricebook_ng.foundation.rate_index import _denominator
    assert _denominator(DC.ACT_360) == 360.0
    assert _denominator(DC.ACT_365_FIXED) == 365.0
    with pytest.raises(ValueError):
        _denominator(DC.THIRTY_360)


# 3.7 accrued_rate accepts any FixingSource (protocol), not just FixingHistory
def test_3_7_accrued_rate_accepts_a_fixing_source_protocol():
    from pricebook_ng.foundation.day_count import Accrual
    from pricebook_ng.foundation.rate_index import FixingSource, accrued_rate, get_rate_index

    class ConstFixings:  # a non-FixingHistory source (e.g. a future db/live feed)
        def rate(self, index_name: str, on: date) -> float:
            return 0.05

    assert isinstance(ConstFixings(), FixingSource)  # runtime_checkable
    sofr = get_rate_index("SOFR")
    acc = Accrual(date(2024, 4, 1), date(2024, 4, 8), DC.ACT_360)
    assert accrued_rate(sofr, acc, ConstFixings()) == pytest.approx(0.05, abs=1e-3)


# 3.6 FX spot date: T+2 joint-calendar, T+1 for USD/CAD, cross cannot settle on a US holiday
def test_3_6_fx_spot_date():
    from pricebook_ng.foundation.money import Currency, CurrencyPair
    from pricebook_ng.foundation.settlement import fx_spot_date, spot_lag
    EUR, USD, GBP, CAD = Currency.EUR, Currency.USD, Currency.GBP, Currency.CAD
    assert fx_spot_date(CurrencyPair(EUR, USD), date(2024, 1, 2)) == date(2024, 1, 4)   # T+2
    assert spot_lag(CurrencyPair(USD, CAD)) == 1
    assert fx_spot_date(CurrencyPair(USD, CAD), date(2024, 1, 2)) == date(2024, 1, 3)   # T+1
    # EUR/GBP cross: T+2 lands on MLK 2024-01-15 (US holiday) → pushes to 01-16
    assert fx_spot_date(CurrencyPair(EUR, GBP), date(2024, 1, 11)) == date(2024, 1, 16)


# ── Phase 3b (Schedule provenance) ──
def test_3b_schedule_emits_per_period_provenance():
    from pricebook_ng.foundation.schedule import SchedulePeriod
    # 7 months quarterly, short front → the first period is a 1-month stub
    terms = ScheduleTerms(frequency=Frequency.QUARTERLY, stub=StubType.SHORT_FRONT)
    s = build_schedule(date(2024, 1, 15), date(2024, 8, 15), terms)
    assert all(isinstance(p, SchedulePeriod) for p in s.periods)
    assert len(s.periods) == len(s.unadjusted) - 1
    assert s.periods[0].is_stub and not s.periods[-1].is_stub
    assert s.periods[0].accrual_start == date(2024, 1, 15)
    assert s.periods[0].accrual_end == date(2024, 2, 15)
    assert s.periods[-1].payment_date == s.adjusted[-1]     # payment at the adjusted period end
    # a regular schedule (exact quarters) has no stub period
    reg = build_schedule(date(2024, 1, 15), date(2024, 7, 15),
                         ScheduleTerms(frequency=Frequency.QUARTERLY))
    assert not any(p.is_stub for p in reg.periods)
