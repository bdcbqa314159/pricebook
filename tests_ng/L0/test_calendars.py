"""Calendar oracles (L0) — Topic 0 Slice 1.

Published reference values: known holiday dates, the three observance regimes
(US 5 U.S.C. §6103 vs Commonwealth next-working-day vs Johannesburg Sunday-only),
year-gated holidays (Juneteenth `since=2021`, Store Bededag `until=2023`), the
Christmas/Boxing collision cascade, Tokyo furikae, the Friday-Saturday weekend, and
the joint-calendar union. Structure is clean (identity-keyed, one `Calendar` type);
content is mined from the quarry `core/calendar.py`.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.calendar import BusinessDayConvention as BDC
from pricebook_ng.foundation.calendar import JointCalendar
from pricebook_ng.foundation.market_calendars import (
    calendar_for_currency,
    get_calendar,
    list_calendars,
)

NY = get_calendar("NEW_YORK_SIFMA")
LON = get_calendar("LONDON")
TGT = get_calendar("TARGET")
TOK = get_calendar("TOKYO")


# ── identity keying (C1): currency → calendar is a lookup, not the key ──
def test_all_37_markets_declared():
    assert len(list_calendars()) == 37


def test_currency_maps_to_identity():
    assert calendar_for_currency("USD") is get_calendar("NEW_YORK_SIFMA")
    assert calendar_for_currency("EUR") is get_calendar("TARGET")
    assert calendar_for_currency("GBP") is get_calendar("LONDON")


def test_unknown_identity_and_currency_raise():
    with pytest.raises(ValueError):
        get_calendar("ATLANTIS")
    with pytest.raises(ValueError):
        calendar_for_currency("XXX")


# ── known holidays ──
def test_us_fixed_and_floating_holidays():
    assert NY.is_holiday(date(2024, 1, 15))    # MLK — 3rd Mon Jan
    assert NY.is_holiday(date(2024, 11, 28))   # Thanksgiving — 4th Thu Nov
    assert not NY.is_business_day(date(2024, 1, 15))
    assert NY.is_business_day(date(2024, 1, 16))


def test_target_good_friday():
    assert TGT.is_holiday(date(2024, 3, 29))   # Good Friday 2024
    assert not TGT.is_holiday(date(2024, 1, 15))  # MLK is not a TARGET holiday


# ── observance: the three regimes ──
def test_us_sunday_holiday_observed_monday():
    # Independence Day 2021-07-04 is a Sunday → observed Monday 2021-07-05
    assert NY.is_holiday(date(2021, 7, 5))
    assert not NY.is_business_day(date(2021, 7, 5))


def test_us_vs_uk_saturday_divergence():
    # New Year 2022-01-01 is a Saturday. US → previous Friday; UK → next Monday.
    assert NY.is_holiday(date(2021, 12, 31))       # US: Friday before
    assert not NY.is_holiday(date(2022, 1, 3))
    assert LON.is_holiday(date(2022, 1, 3))        # UK: Monday after
    assert not LON.is_holiday(date(2021, 12, 31))


def test_johannesburg_sunday_only():
    jhb = get_calendar("JOHANNESBURG")
    # Human Rights Day 21 Mar 2021 is a Sunday → observed Monday (Sunday-only regime)
    assert jhb.is_holiday(date(2021, 3, 22))


# ── year-gated holidays ──
def test_juneteenth_since_2021():
    assert not NY.is_holiday(date(2020, 6, 19))    # before it existed
    assert NY.is_holiday(date(2021, 6, 18))        # 2021-06-19 is Sat → observed Fri


def test_store_bededag_until_2023():
    cph = get_calendar("COPENHAGEN")
    assert cph.is_holiday(date(2023, 5, 5))        # Great Prayer Day 2023 (easter+26)
    assert not cph.is_holiday(date(2024, 4, 26))   # abolished from 2024


# ── collision cascade + furikae ──
def test_christmas_boxing_cascade():
    # 2021: Christmas Sat, Boxing Sun. UK both observe onto Mon/Tue (27th, 28th).
    assert LON.is_holiday(date(2021, 12, 27))
    assert LON.is_holiday(date(2021, 12, 28))


def test_tokyo_furikae():
    # National Foundation Day 2024-02-11 is a Sunday → furikae substitute Mon 02-12
    assert TOK.is_holiday(date(2024, 2, 12))


# ── weekend rule ──
def test_tel_aviv_friday_saturday_weekend():
    tlv = get_calendar("TEL_AVIV")
    assert tlv.is_weekend(date(2024, 6, 7))    # Friday
    assert tlv.is_weekend(date(2024, 6, 8))    # Saturday
    assert not tlv.is_weekend(date(2024, 6, 9))  # Sunday is a business day in Israel


# ── business-day adjustment ──
def test_adjust_conventions():
    d = date(2022, 1, 1)  # Saturday
    assert NY.adjust(d, BDC.FOLLOWING) == date(2022, 1, 3)      # next business day (Mon)
    assert NY.adjust(d, BDC.PRECEDING) == date(2021, 12, 30)    # 12-31 is observed-holiday → 30th
    assert NY.adjust(date(2024, 1, 16), BDC.FOLLOWING) == date(2024, 1, 16)  # already good


def test_modified_following_stays_in_month():
    # 30 Apr 2022 is a Saturday; FOLLOWING would spill to May → MODIFIED_FOLLOWING pulls back
    d = date(2022, 4, 30)
    assert NY.adjust(d, BDC.MODIFIED_FOLLOWING) == date(2022, 4, 29)


# ── joint calendar ──
def test_joint_calendar_is_union():
    joint = JointCalendar(NY, TGT)
    assert joint.is_holiday(date(2024, 11, 28))   # US Thanksgiving (not TARGET)
    assert joint.is_holiday(date(2024, 3, 29))    # TARGET Good Friday (not US)
    assert joint.is_business_day(date(2024, 1, 16))  # neither
