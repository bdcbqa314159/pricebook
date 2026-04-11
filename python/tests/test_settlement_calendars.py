"""Tests for new calendars (CHF, AUD, CAD) and business-day-aware settlement."""

import pytest
from datetime import date, timedelta

from pricebook.calendar import (
    AUDCalendar,
    CADCalendar,
    CHFCalendar,
    JointCalendar,
    LondonCalendar,
    TARGETCalendar,
    USSettlementCalendar,
)
from pricebook.settlement import (
    add_business_days,
    bond_settlement_date,
    fx_spot_date,
)


# ---- Step 1: new calendars ----

class TestCHFCalendar:
    def test_new_year(self):
        cal = CHFCalendar()
        assert cal.is_holiday(date(2024, 1, 1))

    def test_berchtoldstag(self):
        cal = CHFCalendar()
        assert cal.is_holiday(date(2024, 1, 2))

    def test_swiss_national_day(self):
        cal = CHFCalendar()
        assert cal.is_holiday(date(2024, 8, 1))

    def test_good_friday(self):
        cal = CHFCalendar()
        # 2024 Easter = March 31 → Good Friday = March 29
        assert cal.is_holiday(date(2024, 3, 29))

    def test_ascension(self):
        cal = CHFCalendar()
        # 2024 Easter = March 31 → Ascension = May 9
        assert cal.is_holiday(date(2024, 5, 9))

    def test_whit_monday(self):
        cal = CHFCalendar()
        # 2024 Easter = March 31 → Whit Monday = May 20
        assert cal.is_holiday(date(2024, 5, 20))

    def test_normal_day_is_business(self):
        cal = CHFCalendar()
        # 2024-01-03 is a Wednesday, not a holiday
        assert cal.is_business_day(date(2024, 1, 3))

    def test_christmas(self):
        cal = CHFCalendar()
        assert cal.is_holiday(date(2024, 12, 25))
        assert cal.is_holiday(date(2024, 12, 26))


class TestAUDCalendar:
    def test_australia_day(self):
        cal = AUDCalendar()
        assert cal.is_holiday(date(2024, 1, 26))

    def test_anzac_day(self):
        cal = AUDCalendar()
        assert cal.is_holiday(date(2024, 4, 25))

    def test_good_friday(self):
        cal = AUDCalendar()
        assert cal.is_holiday(date(2024, 3, 29))

    def test_easter_saturday(self):
        cal = AUDCalendar()
        # 2024 Easter = March 31 → Saturday = March 30
        assert cal.is_holiday(date(2024, 3, 30))

    def test_queens_birthday(self):
        cal = AUDCalendar()
        # 2nd Monday in June 2024 = June 10
        assert cal.is_holiday(date(2024, 6, 10))

    def test_bank_holiday(self):
        cal = AUDCalendar()
        # 1st Monday in August 2024 = August 5
        assert cal.is_holiday(date(2024, 8, 5))

    def test_normal_day(self):
        cal = AUDCalendar()
        assert cal.is_business_day(date(2024, 3, 4))


class TestCADCalendar:
    def test_canada_day(self):
        cal = CADCalendar()
        assert cal.is_holiday(date(2024, 7, 1))

    def test_family_day(self):
        cal = CADCalendar()
        # 3rd Monday in February 2024 = Feb 19
        assert cal.is_holiday(date(2024, 2, 19))

    def test_victoria_day(self):
        cal = CADCalendar()
        # Monday before May 25, 2024 → May 25 is Saturday → Mon before = May 20
        assert cal.is_holiday(date(2024, 5, 20))

    def test_thanksgiving(self):
        cal = CADCalendar()
        # 2nd Monday in October 2024 = Oct 14
        assert cal.is_holiday(date(2024, 10, 14))

    def test_remembrance_day(self):
        cal = CADCalendar()
        assert cal.is_holiday(date(2024, 11, 11))

    def test_good_friday(self):
        cal = CADCalendar()
        assert cal.is_holiday(date(2024, 3, 29))

    def test_labour_day(self):
        cal = CADCalendar()
        # 1st Monday in September 2024 = Sep 2
        assert cal.is_holiday(date(2024, 9, 2))

    def test_normal_day(self):
        cal = CADCalendar()
        assert cal.is_business_day(date(2024, 3, 4))


class TestJointCalendarWithNew:
    def test_us_chf_joint(self):
        joint = JointCalendar(USSettlementCalendar(), CHFCalendar())
        # Swiss National Day (Aug 1) is holiday in CHF but not US
        assert joint.is_holiday(date(2024, 8, 1))
        # July 4 is US holiday but not Swiss
        assert joint.is_holiday(date(2024, 7, 4))


# ---- Step 2: business-day-aware settlement ----

class TestAddBusinessDays:
    def test_skip_weekend(self):
        # Friday + 1 business day = Monday
        friday = date(2024, 1, 12)  # Friday
        assert friday.weekday() == 4
        result = add_business_days(friday, 1)
        assert result == date(2024, 1, 15)  # Monday

    def test_skip_holiday_with_calendar(self):
        cal = USSettlementCalendar()
        # MLK Day 2024 = Jan 15 (Monday)
        friday = date(2024, 1, 12)
        result = add_business_days(friday, 1, cal)
        # Monday Jan 15 is MLK → next business day is Tue Jan 16
        assert result == date(2024, 1, 16)

    def test_zero_days(self):
        d = date(2024, 1, 15)
        assert add_business_days(d, 0) == d

    def test_negative_days(self):
        # Monday - 1 business day = Friday
        monday = date(2024, 1, 15)
        assert add_business_days(monday, -1) == date(2024, 1, 12)

    def test_multiple_days(self):
        # Wednesday + 5 business days = next Wednesday
        wed = date(2024, 1, 10)
        assert add_business_days(wed, 5) == date(2024, 1, 17)

    def test_no_calendar_only_weekends(self):
        # No calendar → only weekends skipped
        friday = date(2024, 1, 12)
        assert add_business_days(friday, 2) == date(2024, 1, 16)  # Tue


class TestFXSpotDate:
    def test_t_plus_2_default(self):
        """Most FX pairs settle T+2."""
        trade = date(2024, 1, 10)  # Wednesday
        spot = fx_spot_date(trade, "EUR", "USD")
        assert spot == date(2024, 1, 12)  # Friday

    def test_t_plus_1_usd_cad(self):
        """USD/CAD settles T+1."""
        trade = date(2024, 1, 10)
        spot = fx_spot_date(trade, "USD", "CAD")
        assert spot == date(2024, 1, 11)

    def test_skip_weekend(self):
        # Thursday T+2 → skip weekend → Monday
        trade = date(2024, 1, 11)  # Thursday
        spot = fx_spot_date(trade, "EUR", "USD")
        assert spot == date(2024, 1, 15)  # Monday (T+2 skipping weekend)

    def test_skip_holiday_with_calendar(self):
        cal = USSettlementCalendar()
        # Trade on Fri Jan 12. T+2 = Tue Jan 16 (Mon Jan 15 = MLK)
        trade = date(2024, 1, 12)
        spot = fx_spot_date(trade, "EUR", "USD", cal)
        assert spot == date(2024, 1, 17)  # Wed (Mon=MLK, Tue=T+1, Wed=T+2)

    def test_case_insensitive(self):
        trade = date(2024, 1, 10)
        assert fx_spot_date(trade, "usd", "cad") == date(2024, 1, 11)

    def test_spot_lands_on_business_day(self):
        """Result is always a business day."""
        cal = USSettlementCalendar()
        for d in range(1, 32):
            try:
                trade = date(2024, 1, d)
            except ValueError:
                continue
            spot = fx_spot_date(trade, "EUR", "USD", cal)
            assert spot.weekday() < 5  # not weekend
            assert not cal.is_holiday(spot)


class TestBondSettlementDate:
    def test_us_t_plus_1(self):
        trade = date(2024, 1, 10)  # Wednesday
        settle = bond_settlement_date(trade, "US")
        assert settle == date(2024, 1, 11)  # Thursday

    def test_eu_t_plus_2(self):
        trade = date(2024, 1, 10)
        settle = bond_settlement_date(trade, "EUR")
        assert settle == date(2024, 1, 12)

    def test_uk_t_plus_1(self):
        trade = date(2024, 1, 10)
        settle = bond_settlement_date(trade, "GBP")
        assert settle == date(2024, 1, 11)

    def test_jp_t_plus_2(self):
        trade = date(2024, 1, 10)
        settle = bond_settlement_date(trade, "JPY")
        assert settle == date(2024, 1, 12)

    def test_unknown_market_defaults_t_plus_2(self):
        trade = date(2024, 1, 10)
        settle = bond_settlement_date(trade, "BRL")
        assert settle == date(2024, 1, 12)

    def test_skip_weekend(self):
        trade = date(2024, 1, 11)  # Thursday
        settle = bond_settlement_date(trade, "EUR")
        # T+2 from Thursday → skip weekend → Monday
        assert settle == date(2024, 1, 15)

    def test_skip_holiday_with_calendar(self):
        cal = USSettlementCalendar()
        # Trade Fri Jan 12. US T+1 → Mon Jan 15 = MLK → Tue Jan 16
        trade = date(2024, 1, 12)
        settle = bond_settlement_date(trade, "US", cal)
        assert settle == date(2024, 1, 16)

    def test_case_insensitive(self):
        assert bond_settlement_date(date(2024, 1, 10), "us") == date(2024, 1, 11)
