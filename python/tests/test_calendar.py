"""Tests for business day calendars and date adjustment."""

import pytest
from datetime import date

from pricebook.calendar import (
    BusinessDayConvention,
    USSettlementCalendar,
)


@pytest.fixture
def nyc():
    return USSettlementCalendar()


class TestUSSettlementHolidays:
    """Verify known US holidays."""

    def test_new_years_2024(self, nyc):
        # Jan 1 2024 is a Monday
        assert nyc.is_holiday(date(2024, 1, 1))

    def test_new_years_on_saturday_observed_friday(self, nyc):
        # Jan 1 2028 is a Saturday -> observed Friday Dec 31 2027
        assert nyc.is_holiday(date(2027, 12, 31))

    def test_new_years_on_sunday_observed_monday(self, nyc):
        # Jan 1 2023 is a Sunday -> observed Monday Jan 2
        assert nyc.is_holiday(date(2023, 1, 2))

    def test_mlk_day_2024(self, nyc):
        # 3rd Monday of January 2024 = Jan 15
        assert nyc.is_holiday(date(2024, 1, 15))

    def test_presidents_day_2024(self, nyc):
        # 3rd Monday of February 2024 = Feb 19
        assert nyc.is_holiday(date(2024, 2, 19))

    def test_memorial_day_2024(self, nyc):
        # Last Monday of May 2024 = May 27
        assert nyc.is_holiday(date(2024, 5, 27))

    def test_juneteenth_2024(self, nyc):
        # June 19 2024 is a Wednesday
        assert nyc.is_holiday(date(2024, 6, 19))

    def test_juneteenth_not_before_2021(self, nyc):
        assert not nyc.is_holiday(date(2020, 6, 19))

    def test_independence_day_2024(self, nyc):
        # July 4 2024 is a Thursday
        assert nyc.is_holiday(date(2024, 7, 4))

    def test_labor_day_2024(self, nyc):
        # 1st Monday of September 2024 = Sep 2
        assert nyc.is_holiday(date(2024, 9, 2))

    def test_thanksgiving_2024(self, nyc):
        # 4th Thursday of November 2024 = Nov 28
        assert nyc.is_holiday(date(2024, 11, 28))

    def test_christmas_2024(self, nyc):
        # Dec 25 2024 is a Wednesday
        assert nyc.is_holiday(date(2024, 12, 25))

    def test_regular_day_is_not_holiday(self, nyc):
        assert not nyc.is_holiday(date(2024, 3, 12))


class TestBusinessDay:
    """Test weekend and business day detection."""

    def test_weekday_is_business_day(self, nyc):
        # Wednesday March 13 2024
        assert nyc.is_business_day(date(2024, 3, 13))

    def test_saturday_is_not_business_day(self, nyc):
        assert not nyc.is_business_day(date(2024, 3, 16))

    def test_sunday_is_not_business_day(self, nyc):
        assert not nyc.is_business_day(date(2024, 3, 17))

    def test_holiday_is_not_business_day(self, nyc):
        assert not nyc.is_business_day(date(2024, 12, 25))


class TestAdjust:
    """Test business day adjustment conventions."""

    def test_business_day_unchanged(self, nyc):
        # Wednesday stays Wednesday for all conventions
        d = date(2024, 3, 13)
        for conv in BusinessDayConvention:
            if conv == BusinessDayConvention.UNADJUSTED:
                continue
            assert nyc.adjust(d, conv) == d

    def test_unadjusted_returns_same_date(self, nyc):
        # Saturday stays Saturday
        d = date(2024, 3, 16)
        assert nyc.adjust(d, BusinessDayConvention.UNADJUSTED) == d

    def test_following(self, nyc):
        # Saturday Mar 16 -> Monday Mar 18
        assert nyc.adjust(date(2024, 3, 16), BusinessDayConvention.FOLLOWING) == date(2024, 3, 18)

    def test_preceding(self, nyc):
        # Saturday Mar 16 -> Friday Mar 15
        assert nyc.adjust(date(2024, 3, 16), BusinessDayConvention.PRECEDING) == date(2024, 3, 15)

    def test_modified_following_same_month(self, nyc):
        # Saturday Mar 16 -> Monday Mar 18 (same month, no modification)
        assert nyc.adjust(date(2024, 3, 16), BusinessDayConvention.MODIFIED_FOLLOWING) == date(2024, 3, 18)

    def test_modified_following_crosses_month(self, nyc):
        # Saturday Mar 30 2024 -> following would be Mon Apr 1
        # But that crosses month -> preceding to Fri Mar 29
        assert nyc.adjust(date(2024, 3, 30), BusinessDayConvention.MODIFIED_FOLLOWING) == date(2024, 3, 29)

    def test_modified_preceding_same_month(self, nyc):
        # Sunday Mar 17 -> preceding is Fri Mar 15 (same month)
        assert nyc.adjust(date(2024, 3, 17), BusinessDayConvention.MODIFIED_PRECEDING) == date(2024, 3, 15)

    def test_modified_preceding_crosses_month(self, nyc):
        # Sunday Sep 1 2024 -> preceding would be Sat Aug 31 -> Fri Aug 30
        # But that crosses month -> following to Mon Sep 2
        # Sep 2 is Labor Day -> Tue Sep 3
        assert nyc.adjust(date(2024, 9, 1), BusinessDayConvention.MODIFIED_PRECEDING) == date(2024, 9, 3)

    def test_holiday_adjusted_following(self, nyc):
        # Christmas 2024 is Wednesday -> following is Thursday Dec 26
        assert nyc.adjust(date(2024, 12, 25), BusinessDayConvention.FOLLOWING) == date(2024, 12, 26)
