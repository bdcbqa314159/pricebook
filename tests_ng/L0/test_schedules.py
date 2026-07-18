"""Schedule oracles (L0) — Topic 0 Slice 3.

Published references: ISDA 2006 §4.10 EOM anchoring (a month-end start rolls to
month-ends), all four stub types, adjusted ≠ unadjusted under a holiday (C2: accrual
uses unadjusted, payment adjusted), and the IMM (3rd Wednesday) and CDS (20th of
Mar/Jun/Sep/Dec) roll tables — neither of which existed in the quarry. The long-stub
`first_gap < months*30*0.5` merge heuristic is shed for an explicit stub spec.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.calendar import BusinessDayConvention as BDC
from pricebook_ng.foundation.market_calendars import get_calendar
from pricebook_ng.foundation.schedule import (
    Frequency,
    RollRule,
    Schedule,
    ScheduleTerms,
    StubType,
    build_schedule,
    cds_roll_date,
    imm_date,
    next_cds_roll,
    next_imm,
)

NY = get_calendar("NEW_YORK_SIFMA")


def _terms(freq, stub=StubType.SHORT_FRONT, roll=None):
    return ScheduleTerms(frequency=freq, roll=roll or RollRule(), stub=stub)


# ── IMM & CDS roll dates (new) ──
def test_imm_third_wednesday():
    assert imm_date(2024, 12) == date(2024, 12, 18)
    assert imm_date(2024, 6) == date(2024, 6, 19)
    assert next_imm(date(2024, 6, 1)) == date(2024, 6, 19)
    assert next_imm(date(2024, 6, 20)) == date(2024, 9, 18)  # rolled past June IMM


def test_cds_roll_twentieth():
    assert cds_roll_date(2024, 9) == date(2024, 9, 20)
    assert next_cds_roll(date(2024, 1, 1)) == date(2024, 3, 20)
    assert next_cds_roll(date(2024, 3, 20)) == date(2024, 3, 20)  # on the roll
    assert next_cds_roll(date(2024, 3, 21)) == date(2024, 6, 20)


# ── EOM anchoring (ISDA §4.10) ──
def test_eom_anchored_from_start():
    s = build_schedule(date(2024, 1, 31), date(2024, 4, 30), _terms(Frequency.MONTHLY))
    assert s.unadjusted == (date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31), date(2024, 4, 30))


def test_non_eom_start_preserves_day_with_clamp():
    # start 01-30 is not month-end → day preserved, clamped in short months (Feb → 29)
    s = build_schedule(date(2024, 1, 30), date(2024, 4, 30), _terms(Frequency.MONTHLY))
    assert s.unadjusted == (date(2024, 1, 30), date(2024, 2, 29), date(2024, 3, 30), date(2024, 4, 30))


# ── the four stubs (explicit, no merge heuristic) ──
def test_short_vs_long_front_stub():
    # 7 months, quarterly → 2 regular + a 1-month remainder at the front
    a, b = date(2024, 1, 15), date(2024, 8, 15)
    short = build_schedule(a, b, _terms(Frequency.QUARTERLY, StubType.SHORT_FRONT)).unadjusted
    long = build_schedule(a, b, _terms(Frequency.QUARTERLY, StubType.LONG_FRONT)).unadjusted
    assert short == (date(2024, 1, 15), date(2024, 2, 15), date(2024, 5, 15), date(2024, 8, 15))
    assert long == (date(2024, 1, 15), date(2024, 5, 15), date(2024, 8, 15))


def test_short_vs_long_back_stub():
    a, b = date(2024, 1, 15), date(2024, 8, 15)
    short = build_schedule(a, b, _terms(Frequency.QUARTERLY, StubType.SHORT_BACK)).unadjusted
    long = build_schedule(a, b, _terms(Frequency.QUARTERLY, StubType.LONG_BACK)).unadjusted
    assert short == (date(2024, 1, 15), date(2024, 4, 15), date(2024, 7, 15), date(2024, 8, 15))
    assert long == (date(2024, 1, 15), date(2024, 4, 15), date(2024, 8, 15))


def test_regular_period_no_stub():
    # start on a regular roll boundary → no stub to merge under either LONG type
    a, b = date(2024, 1, 15), date(2024, 7, 15)  # exactly 2 quarters
    reg = build_schedule(a, b, _terms(Frequency.QUARTERLY, StubType.LONG_FRONT)).unadjusted
    assert reg == (date(2024, 1, 15), date(2024, 4, 15), date(2024, 7, 15))


# ── adjusted ≠ unadjusted under a holiday (C2) ──
def test_adjusted_and_unadjusted_differ_under_holiday():
    # a quarterly schedule whose 06-30 roll is a Sunday; MODIFIED_FOLLOWING pulls it
    # back to Fri 06-28 (07-01 would spill into July)
    roll = RollRule(calendar=NY, convention=BDC.MODIFIED_FOLLOWING, eom=False)
    s = build_schedule(date(2023, 12, 30), date(2024, 6, 30), _terms(Frequency.QUARTERLY, roll=roll))
    assert date(2024, 6, 30) in s.unadjusted          # accrual boundary is the unadjusted date
    assert date(2024, 6, 30) not in s.adjusted        # payment moves off the weekend
    assert date(2024, 6, 28) in s.adjusted            # → Friday, same month
    assert len(s.unadjusted) == len(s.adjusted)


def test_no_calendar_leaves_dates_unadjusted():
    s = build_schedule(date(2024, 1, 15), date(2024, 7, 15), _terms(Frequency.QUARTERLY))
    assert s.adjusted == s.unadjusted   # RollRule with no calendar → no adjustment


def test_start_after_end_raises():
    with pytest.raises(ValueError):
        build_schedule(date(2024, 7, 1), date(2024, 1, 1), _terms(Frequency.MONTHLY))
