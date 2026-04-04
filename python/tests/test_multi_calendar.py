"""Tests for multi-calendar, settlement, and ACT/ACT ISDA."""

import pytest
from datetime import date, timedelta

from pricebook.calendar import (
    TARGETCalendar,
    LondonCalendar,
    TokyoCalendar,
    JointCalendar,
    USSettlementCalendar,
    BusinessDayConvention,
)
from pricebook.day_count import DayCountConvention, year_fraction


class TestTARGETCalendar:
    def test_new_year(self):
        cal = TARGETCalendar()
        assert not cal.is_business_day(date(2024, 1, 1))

    def test_labour_day(self):
        cal = TARGETCalendar()
        assert not cal.is_business_day(date(2024, 5, 1))

    def test_christmas(self):
        cal = TARGETCalendar()
        assert not cal.is_business_day(date(2024, 12, 25))
        assert not cal.is_business_day(date(2024, 12, 26))

    def test_good_friday_2024(self):
        cal = TARGETCalendar()
        # Good Friday 2024 = March 29
        assert not cal.is_business_day(date(2024, 3, 29))

    def test_easter_monday_2024(self):
        cal = TARGETCalendar()
        # Easter Monday 2024 = April 1
        assert not cal.is_business_day(date(2024, 4, 1))

    def test_normal_day(self):
        cal = TARGETCalendar()
        assert cal.is_business_day(date(2024, 3, 15))  # Friday

    def test_adjust(self):
        cal = TARGETCalendar()
        adjusted = cal.adjust(date(2024, 1, 1), BusinessDayConvention.FOLLOWING)
        assert adjusted == date(2024, 1, 2)


class TestLondonCalendar:
    def test_new_year(self):
        cal = LondonCalendar()
        assert not cal.is_business_day(date(2024, 1, 1))

    def test_good_friday(self):
        cal = LondonCalendar()
        assert not cal.is_business_day(date(2024, 3, 29))

    def test_early_may(self):
        cal = LondonCalendar()
        # First Monday in May 2024 = May 6
        assert not cal.is_business_day(date(2024, 5, 6))

    def test_christmas(self):
        cal = LondonCalendar()
        assert not cal.is_business_day(date(2024, 12, 25))

    def test_normal_day(self):
        cal = LondonCalendar()
        assert cal.is_business_day(date(2024, 6, 3))


class TestTokyoCalendar:
    def test_new_year(self):
        cal = TokyoCalendar()
        assert not cal.is_business_day(date(2024, 1, 1))
        assert not cal.is_business_day(date(2024, 1, 2))
        assert not cal.is_business_day(date(2024, 1, 3))

    def test_national_foundation(self):
        cal = TokyoCalendar()
        # Feb 11 2024 is Sunday, so the holiday itself is on Sunday
        # But we're checking if it's a non-business day
        assert not cal.is_business_day(date(2024, 2, 11))

    def test_normal_day(self):
        cal = TokyoCalendar()
        assert cal.is_business_day(date(2024, 6, 3))


class TestJointCalendar:
    def test_union_of_holidays(self):
        us = USSettlementCalendar()
        target = TARGETCalendar()
        joint = JointCalendar(us, target)

        # US holiday but not TARGET
        mlk_2024 = date(2024, 1, 15)  # MLK Day
        assert not joint.is_business_day(mlk_2024)

        # TARGET holiday but not US
        may_1_2024 = date(2024, 5, 1)  # Labour Day
        assert not joint.is_business_day(may_1_2024)

    def test_common_holiday(self):
        us = USSettlementCalendar()
        target = TARGETCalendar()
        joint = JointCalendar(us, target)
        # Jan 1 = holiday in both
        assert not joint.is_business_day(date(2024, 1, 1))

    def test_adjust(self):
        us = USSettlementCalendar()
        target = TARGETCalendar()
        joint = JointCalendar(us, target)
        adjusted = joint.adjust(date(2024, 1, 1), BusinessDayConvention.FOLLOWING)
        assert joint.is_business_day(adjusted)


class TestACTACTISDA:
    def test_same_year(self):
        yf = year_fraction(date(2024, 1, 1), date(2024, 7, 1), DayCountConvention.ACT_ACT_ISDA)
        # 182 days / 366 (2024 is leap year)
        assert yf == pytest.approx(182 / 366, rel=1e-10)

    def test_full_year(self):
        yf = year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCountConvention.ACT_ACT_ISDA)
        assert yf == pytest.approx(1.0, rel=1e-10)

    def test_cross_year(self):
        yf = year_fraction(date(2023, 7, 1), date(2024, 7, 1), DayCountConvention.ACT_ACT_ISDA)
        # 184 days in 2023 (365) + 182 days in 2024 (366)
        expected = 184 / 365 + 182 / 366
        assert yf == pytest.approx(expected, rel=1e-10)

    def test_non_leap_year(self):
        yf = year_fraction(date(2023, 1, 1), date(2023, 7, 1), DayCountConvention.ACT_ACT_ISDA)
        assert yf == pytest.approx(181 / 365, rel=1e-10)

    def test_multi_year(self):
        yf = year_fraction(date(2022, 1, 1), date(2025, 1, 1), DayCountConvention.ACT_ACT_ISDA)
        assert yf == pytest.approx(3.0, rel=1e-10)
