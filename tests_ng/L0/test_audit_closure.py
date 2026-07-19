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

NY = get_calendar("NEW_YORK_SIFMA")
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
    assert TOKYO.is_holiday(date(2024, 3, 20))     # Vernal Equinox 2024 (was hardcoded 3/21)
    assert not TOKYO.is_holiday(date(2024, 3, 21))
    assert TOKYO.is_holiday(date(2024, 9, 22))     # Autumnal Equinox 2024 (was hardcoded 9/23)
    assert not TOKYO.is_holiday(date(2024, 9, 23))
