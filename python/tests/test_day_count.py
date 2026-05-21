"""Tests for day count conventions."""

import pytest
from datetime import date

from pricebook.core.day_count import (
    DayCountConvention, year_fraction, business_days_between,
)
from pricebook.core.calendar import SaoPauloCalendar, USSettlementCalendar


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


class TestBUS252:
    """BUS/252: Brazilian business day convention."""

    @pytest.fixture
    def brl_cal(self):
        return SaoPauloCalendar()

    def test_one_week(self, brl_cal):
        """Mon to Fri = 4 business days (start exclusive, end inclusive)."""
        start = date(2024, 7, 8)   # Monday
        end = date(2024, 7, 12)    # Friday
        yf = year_fraction(start, end, DayCountConvention.BUS_252, calendar=brl_cal)
        assert yf == pytest.approx(4 / 252.0)

    def test_one_year_approx(self, brl_cal):
        """Roughly 252 business days in a year."""
        yf = year_fraction(
            date(2024, 1, 2), date(2025, 1, 2),
            DayCountConvention.BUS_252, calendar=brl_cal,
        )
        assert 0.95 < yf < 1.05  # close to 1.0

    def test_carnival_skip(self, brl_cal):
        """BUS/252 should skip carnival holidays."""
        # 2024 carnival Mon = Feb 12, Tue = Feb 13
        # Feb 9 (Fri) to Feb 14 (Wed) = 1 business day (only Feb 14)
        bd = business_days_between(date(2024, 2, 9), date(2024, 2, 14), brl_cal)
        assert bd == 1

    def test_weekend_skip(self, brl_cal):
        """BUS/252 should skip weekends."""
        # Fri Jul 5 to Mon Jul 8 = 1 business day (Mon only)
        bd = business_days_between(date(2024, 7, 5), date(2024, 7, 8), brl_cal)
        assert bd == 1

    def test_defaults_to_brl_calendar(self):
        """When no calendar passed, BUS/252 defaults to São Paulo."""
        yf = year_fraction(
            date(2024, 7, 8), date(2024, 7, 12), DayCountConvention.BUS_252,
        )
        assert yf == pytest.approx(4 / 252.0)

    def test_with_us_calendar(self):
        """BUS/252 can use a different calendar."""
        cal = USSettlementCalendar()
        # Jul 4 is US holiday. Jul 1 (Mon) to Jul 5 (Fri) = 3 bd (Jul 2, 3, 5)
        bd = business_days_between(date(2024, 7, 1), date(2024, 7, 5), cal)
        assert bd == 3

    def test_business_days_independence_day(self):
        """Sep 7 (BRL Independence Day) is not a business day."""
        cal = SaoPauloCalendar()
        # Sep 6 (Fri) to Sep 9 (Mon) = 1 bd (only Mon Sep 9, since Sep 7 is Sat anyway)
        # Actually Sep 7, 2024 is Saturday. Let's use 2023 where Sep 7 is Thursday.
        # Sep 6, 2023 (Wed) to Sep 8, 2023 (Fri) = 1 bd (Sep 8, since Sep 7 is holiday)
        bd = business_days_between(date(2023, 9, 6), date(2023, 9, 8), cal)
        assert bd == 1


class TestEdgeCases:
    """Edge cases common to all conventions."""

    def test_same_date_returns_zero(self):
        d = date(2024, 6, 15)
        for conv in DayCountConvention:
            if conv == DayCountConvention.BUS_252:
                assert year_fraction(d, d, conv, calendar=SaoPauloCalendar()) == 0.0
            else:
                assert year_fraction(d, d, conv) == 0.0

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            year_fraction(date(2024, 7, 1), date(2024, 1, 1), DayCountConvention.ACT_360)
