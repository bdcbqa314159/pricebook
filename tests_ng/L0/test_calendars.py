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

from pricebook_ng.foundation.calendars import BusinessDayConvention as BDC
from pricebook_ng.foundation.calendars import JointCalendar
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


def test_sifma_vs_uk_saturday_divergence():
    # New Year 2022-01-01 is a Saturday. SIFMA does NOT shift to Friday (bond market open
    # 12-31, audit 2.2); UK shifts to the next Monday.
    assert NY.is_business_day(date(2021, 12, 31))  # SIFMA: Saturday holiday not shifted
    assert LON.is_holiday(date(2022, 1, 3))        # UK: Monday after
    assert not LON.is_holiday(date(2021, 12, 31))


def test_johannesburg_sunday_only():
    jhb = get_calendar("JOHANNESBURG")
    # Human Rights Day 21 Mar 2021 is a Sunday → observed Monday (Sunday-only regime)
    assert jhb.is_holiday(date(2021, 3, 22))


# ── year-gated holidays ──
def test_juneteenth_since_2021():
    assert not NY.is_holiday(date(2020, 6, 19))    # before it existed
    assert NY.is_holiday(date(2022, 6, 20))        # 2022-06-19 is Sun → observed Mon 06-20
    assert NY.is_business_day(date(2021, 6, 18))   # 2021-06-19 is Sat → NOT shifted (SIFMA open)


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
    assert NY.adjust(d, BDC.PRECEDING) == date(2021, 12, 31)    # SIFMA open 12-31 (Sat not shifted)
    assert NY.adjust(date(2024, 1, 16), BDC.FOLLOWING) == date(2024, 1, 16)  # already good


def test_modified_following_stays_in_month():
    # 30 Apr 2022 is a Saturday; FOLLOWING would spill to May → MODIFIED_FOLLOWING pulls back
    d = date(2022, 4, 30)
    assert NY.adjust(d, BDC.MODIFIED_FOLLOWING) == date(2022, 4, 29)


# ── joint calendar ──
def test_joint_calendar_is_union():
    joint = JointCalendar(NY, TGT)
    assert joint.is_holiday(date(2024, 11, 28))   # US Thanksgiving (not TARGET)
    assert joint.is_holiday(date(2024, 5, 1))     # TARGET May Day (not SIFMA)
    assert joint.is_business_day(date(2024, 1, 16))  # neither


# ── Topic 0 S3 checkpoint corrections ──


def test_anzac_day_not_mondayised_but_others_are():
    # ANZAC Day (25 Apr) is commemorated on the actual date in AU/NZ — NOT shifted —
    # even though New Year / Christmas in the SAME calendar are (per Cowork §3.1).
    aud = get_calendar("SYDNEY")
    assert aud.is_holiday(date(2020, 4, 25))        # Sat — stays on the actual date
    assert not aud.is_holiday(date(2020, 4, 27))    # Mon — NOT a substituted holiday
    assert aud.is_holiday(date(2022, 1, 3))         # New Year (Sat 01-01) IS mondayised
    nzd = get_calendar("WELLINGTON")
    assert nzd.is_holiday(date(2021, 4, 25))        # Sun — stays
    assert not nzd.is_holiday(date(2021, 4, 26))    # Mon — not substituted


def test_secular_only_calendars_are_marked():
    from pricebook_ng.foundation.calendars import Coverage
    for ident in ("RIYADH", "CAIRO", "ISTANBUL", "TEL_AVIV", "BEIJING", "SEOUL", "MUMBAI", "BANGKOK"):
        assert get_calendar(ident).coverage is Coverage.SECULAR_ONLY
    assert get_calendar("NEW_YORK_SIFMA").coverage is Coverage.COMPLETE
    assert get_calendar("TARGET").coverage is Coverage.COMPLETE


# ── S5: day_type classification (half-days) ──
def test_day_type_classifies_business_half_holiday_weekend():
    from pricebook_ng.foundation.calendars import DayType
    # Christmas Eve 2024-12-24 (Tue) is a US early close — a HALF day, still a business day
    assert NY.day_type(date(2024, 12, 24)) is DayType.HALF
    assert NY.is_business_day(date(2024, 12, 24))            # markets open (early close)
    assert NY.day_type(date(2024, 12, 25)) is DayType.HOLIDAY
    assert NY.day_type(date(2024, 6, 8)) is DayType.WEEKEND  # Saturday
    assert NY.day_type(date(2024, 6, 10)) is DayType.BUSINESS
    # a calendar with no half-days never returns HALF
    assert TGT.day_type(date(2024, 12, 24)) is DayType.BUSINESS
