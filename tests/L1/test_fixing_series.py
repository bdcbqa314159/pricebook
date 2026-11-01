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
