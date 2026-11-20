"""The index fixing series and lag resolution (L1). BERHAN-2012 §1.1.2, §2.5.

Resolution turns a *date* into the right published *print*: apply the index's publication lag to
get the reference month, then apply its interpolation convention within that month. Two real
conventions, and they disagree:

  FLAT   (UK index-linked gilts) -- the reference month's print, used unchanged
  LINEAR (US TIPS, most markets) -- interpolate between the reference month and the next, by
                                    (day - 1) / days_in_month

**These tests must be able to tell the two apart, which constrains the fixture twice.** A FLAT
series makes every interpolation scheme agree (the tranche 3 lesson), and -- less obvious -- so
does a reference date on the **1st of a month**, because the LINEAR weight is then exactly 0 and
LINEAR degenerates to FLAT. So the fixtures are a SLOPED series read at a MID-MONTH date, and the
degeneracy is asserted rather than assumed (`test_on_the_first_of_a_month_...`).
"""

import calendar
from datetime import date

import pytest

from pricebook.foundation import IndexInterpolation, InflationIndex
from pricebook.market import FixingSeries, reference_level

# A sloped series: +1.0 index point per month, so any two adjacent prints differ and an
# interpolation weight is visible in the answer.
PRINTS = {date(2025, m, 1): 100.0 + (m - 1) for m in range(1, 13)} | {
    date(2026, m, 1): 112.0 + (m - 1) for m in range(1, 13)
}
SERIES = FixingSeries(PRINTS)

TIPS = InflationIndex("USCPI", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
GILT = InflationIndex("UKRPI", publication_lag=8, interpolation=IndexInterpolation.FLAT)


# ---- oracle 3: the lag, pointwise, computed independently of the code under test ----

def test_a_flat_index_resolves_to_the_print_lag_months_earlier():
    """`GILT` has an 8-month lag: 2026-05-15 resolves to the 2025-09-01 print, which the test
    looks up directly in its own dict rather than asking the resolver."""
    assert reference_level(SERIES, date(2026, 5, 15), GILT) == PRINTS[date(2025, 9, 1)]


@pytest.mark.parametrize(
    ("d", "expected_month"),
    [
        (date(2026, 1, 1), date(2025, 5, 1)),    # January back 8 months crosses the year
        (date(2026, 8, 1), date(2025, 12, 1)),   # August back 8 lands on December
        (date(2026, 9, 1), date(2026, 1, 1)),    # September back 8 stays in 2026
    ],
)
def test_the_lag_arithmetic_crosses_year_boundaries(d, expected_month):
    """Month arithmetic with a lag is where an off-by-one hides, so the year-crossing cases are
    enumerated rather than trusted to a `month - lag` expression."""
    assert reference_level(SERIES, d, GILT) == PRINTS[expected_month]


# ---- oracle 4: interpolation has the POWER to fail ----

def test_linear_interpolates_between_the_reference_month_and_the_next():
    """TIPS, 3-month lag, 2026-04-15: the reference month is 2026-01 and the next is 2026-02;
    the weight is (15-1)/30 = 14/30 of the way into April. Computed here from the series and the
    convention, never read back from the resolver."""
    lo, hi = PRINTS[date(2026, 1, 1)], PRINTS[date(2026, 2, 1)]
    expected = lo + (15 - 1) / 30 * (hi - lo)
    assert reference_level(SERIES, date(2026, 4, 15), TIPS) == pytest.approx(expected, rel=1e-15)


def test_flat_and_linear_DISAGREE_on_a_sloped_series_mid_month():
    """The discriminating case. Same index name, same lag, same series, same date -- only the
    convention differs, and the two answers must not coincide, or neither test above has power."""
    flat = InflationIndex("X", publication_lag=3, interpolation=IndexInterpolation.FLAT)
    linear = InflationIndex("X", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
    d = date(2026, 4, 15)
    assert reference_level(SERIES, d, flat) != reference_level(SERIES, d, linear)


def test_on_the_first_of_a_month_linear_DEGENERATES_to_flat():
    """The fixture trap, asserted rather than left implicit: at day 1 the LINEAR weight is exactly
    zero, so a test that only ever reads the 1st of a month cannot distinguish the conventions."""
    flat = InflationIndex("X", publication_lag=3, interpolation=IndexInterpolation.FLAT)
    linear = InflationIndex("X", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
    d = date(2026, 4, 1)
    assert reference_level(SERIES, d, flat) == reference_level(SERIES, d, linear)


def test_a_flat_series_cannot_discriminate_the_conventions_either():
    """The other half of the fixture lesson: on a constant series both conventions agree at every
    date, so the slope is load-bearing and is asserted to be so."""
    flat_series = FixingSeries({date(2025, m, 1): 100.0 for m in range(1, 13)})
    f = InflationIndex("X", publication_lag=3, interpolation=IndexInterpolation.FLAT)
    ln = InflationIndex("X", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
    d = date(2025, 7, 15)
    assert reference_level(flat_series, d, f) == reference_level(flat_series, d, ln)


# ---- F-7.1: the month length, which survived fifteen tranches unpinned ----

def test_linear_interpolates_in_a_NON_THIRTY_DAY_month_with_a_DERIVED_divisor():
    """**Re-audit #7, F-7.1 — a blind oracle, and a new shape in the degenerate-fixture catalogue.**

    Hard-coding `days_in_month = 30.0` in `reference_level` survived all 2625 tests. The source was
    always right; the oracle was blind, for **two** reasons that had to be broken together:

    1. The only pointwise linear value test read **2026-04-15 — and April has exactly 30 days**, so
       the real divisor and the literal were the same number.
    2. **That test's own reference wrote `/ 30` as well.** So §4's independence rule was satisfied
       *formally* — the test did compute its own reference — while the reference was independent of
       the **code** and not of the **coincidence**.

    So this test does two things the old one could not: it reads **February (28 days)**, and it
    **derives** the month length with `calendar.monthrange` instead of writing a literal. Either
    alone would be insufficient — a non-30-day month with a hard-coded `30` in the reference still
    agrees with the mutant, and a derived divisor read in April still cannot see it.

    Measured: at 28 days the weight is `14/28 = 0.5000` against the mutant's `14/30 = 0.4667`, a
    **6.67%** relative error on the weight.
    """
    reference_date = date(2026, 2, 15)
    days_in_month = calendar.monthrange(reference_date.year, reference_date.month)[1]
    assert days_in_month == 28, "February must be the fixture, or the coincidence is not broken"

    # TIPS, 3-month lag: 2026-02 resolves to the 2025-11 print, the next month is 2025-12.
    lo, hi = PRINTS[date(2025, 11, 1)], PRINTS[date(2025, 12, 1)]
    expected = lo + (reference_date.day - 1) / days_in_month * (hi - lo)
    assert reference_level(SERIES, reference_date, TIPS) == pytest.approx(expected, rel=1e-15)


def test_the_MONTH_LENGTH_is_what_the_new_fixture_pins():
    """The companion that proves the test above is the one with power, rather than another reading
    of the same thing: on this date a hard-coded 30 gives a **measurably different** answer, and
    this states by how much. In April it would give the identical answer, which is the whole of
    F-7.1."""
    february, april = date(2026, 2, 15), date(2026, 4, 15)
    for reference_date, must_differ in ((february, True), (april, False)):
        days = calendar.monthrange(reference_date.year, reference_date.month)[1]
        correct = (reference_date.day - 1) / days
        hard_coded = (reference_date.day - 1) / 30.0
        assert (abs(correct - hard_coded) > 1e-9) is must_differ, (
            f"{reference_date}: {days} days, correct weight {correct}, at 30 {hard_coded}"
        )
    # And the size of it, so the finding's severity is in the suite and not only in the findings doc.
    assert abs(0.5 - 14.0 / 30.0) / 0.5 > 0.06


def test_a_THIRTY_ONE_DAY_month_interpolates_from_a_DERIVED_divisor_too():
    """The other side of 30, so the fixture is not merely non-30 but brackets it — a mutant rounding
    the month length in either direction is caught."""
    reference_date = date(2026, 1, 20)
    days_in_month = calendar.monthrange(reference_date.year, reference_date.month)[1]
    assert days_in_month == 31
    lo, hi = PRINTS[date(2025, 10, 1)], PRINTS[date(2025, 11, 1)]
    expected = lo + (reference_date.day - 1) / days_in_month * (hi - lo)
    assert reference_level(SERIES, reference_date, TIPS) == pytest.approx(expected, rel=1e-15)


# ---- oracle 7: failure is a value, never a guess ----

def test_a_date_before_the_series_begins_RAISES_rather_than_extrapolating():
    """Extending a series backwards is inventing market data. Same stance as `inflation_leg_value`
    refusing to assume the base level."""
    with pytest.raises(KeyError, match="no index print"):
        reference_level(SERIES, date(2025, 2, 15), GILT)   # back 8 months -> 2024-06, absent


def test_linear_needs_the_NEXT_month_too_and_refuses_without_it():
    """The subtle one: LINEAR reads two prints, so a series that covers the reference month but
    stops there must refuse -- not silently fall back to flat, which would return a plausible
    number under the wrong convention."""
    short = FixingSeries({date(2026, 1, 1): 112.0})
    idx = InflationIndex("X", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
    with pytest.raises(KeyError, match="no index print"):
        reference_level(short, date(2026, 4, 15), idx)


def test_a_one_point_series_resolves_FLAT_at_its_own_month():
    """The degenerate series that oracle 1's reduction leans on: one print, read flat, is exactly
    the old hand-fed scalar."""
    one = FixingSeries({date(2026, 1, 1): 112.0})
    idx = InflationIndex("X", publication_lag=0, interpolation=IndexInterpolation.FLAT)
    assert reference_level(one, date(2026, 1, 20), idx) == 112.0


def test_an_empty_series_is_refused_at_construction():
    with pytest.raises(ValueError, match="at least one print"):
        FixingSeries({})


def test_a_print_must_be_the_first_of_its_month():
    """A reference month is a month, not a day. Accepting `date(2026,1,17)` as a key would make
    two spellings of the same month two different keys."""
    with pytest.raises(ValueError, match="first of the month"):
        FixingSeries({date(2026, 1, 17): 112.0})


def test_a_non_positive_print_is_refused():
    with pytest.raises(ValueError, match="positive"):
        FixingSeries({date(2026, 1, 1): 0.0})
