"""Tests for day count conventions."""

import pytest
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction


class TestACT360:
    """ACT/360: standard USD money market convention."""

    def test_six_months(self):
        # Jan 15 to Jul 15 = 182 actual days
        yf = year_fraction(date(2024, 1, 15), date(2024, 7, 15), DayCountConvention.ACT_360)
        assert yf == pytest.approx(182 / 360.0)

    def test_one_year(self):
        # 2024 is a leap year: 366 days
        yf = year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCountConvention.ACT_360)
        assert yf == pytest.approx(366 / 360.0)

    def test_one_month(self):
        # Jan 1 to Feb 1 = 31 days
        yf = year_fraction(date(2024, 1, 1), date(2024, 2, 1), DayCountConvention.ACT_360)
        assert yf == pytest.approx(31 / 360.0)


class TestACT365Fixed:
    """ACT/365F: standard GBP convention."""

    def test_six_months(self):
        yf = year_fraction(date(2024, 1, 15), date(2024, 7, 15), DayCountConvention.ACT_365_FIXED)
        assert yf == pytest.approx(182 / 365.0)

    def test_one_year_leap(self):
        yf = year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCountConvention.ACT_365_FIXED)
        assert yf == pytest.approx(366 / 365.0)

    def test_one_year_non_leap(self):
        yf = year_fraction(date(2023, 1, 1), date(2024, 1, 1), DayCountConvention.ACT_365_FIXED)
        assert yf == pytest.approx(365 / 365.0)


class TestThirty360:
    """30/360 US (Bond Basis)."""

    def test_regular_period(self):
        # 30/360: Jan 15 to Jul 15 = 6 * 30 = 180 days
        yf = year_fraction(date(2024, 1, 15), date(2024, 7, 15), DayCountConvention.THIRTY_360)
        assert yf == pytest.approx(180 / 360.0)

    def test_one_year(self):
        yf = year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCountConvention.THIRTY_360)
        assert yf == pytest.approx(1.0)

    def test_day_31_adjustment(self):
        # Jan 31 to Mar 31: d1=31->30, d2=31->30 (since d1=30)
        # 2*30 + (30-30) = 60 days
        yf = year_fraction(date(2024, 1, 31), date(2024, 3, 31), DayCountConvention.THIRTY_360)
        assert yf == pytest.approx(60 / 360.0)

    def test_day_31_no_adjustment(self):
        # Jan 15 to Mar 31: d1=15 (no adj), d2=31 stays (since d1!=30)
        # 2*30 + (31-15) = 76 days
        yf = year_fraction(date(2024, 1, 15), date(2024, 3, 31), DayCountConvention.THIRTY_360)
        assert yf == pytest.approx(76 / 360.0)


class TestEdgeCases:
    """Edge cases common to all conventions."""

    def test_same_date_returns_zero(self):
        d = date(2024, 6, 15)
        for conv in DayCountConvention:
            assert year_fraction(d, d, conv) == 0.0

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            year_fraction(date(2024, 7, 1), date(2024, 1, 1), DayCountConvention.ACT_360)
