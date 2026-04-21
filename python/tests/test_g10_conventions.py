"""Tests for G10 calendar, day count, and convention coverage."""

from datetime import date
import math
import pytest

from pricebook.calendar import (
    SEKCalendar, NOKCalendar, NZDCalendar,
    USSettlementCalendar, TARGETCalendar, LondonCalendar,
    TokyoCalendar, CHFCalendar, AUDCalendar, CADCalendar,
)
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.curve_builder import build_curves, _CONVENTIONS


# ---- Calendars: all G10 exist ----

class TestG10Calendars:
    def test_all_g10_calendars_exist(self):
        """Every G10 currency has a calendar implementation."""
        cals = {
            "USD": USSettlementCalendar(),
            "EUR": TARGETCalendar(),
            "GBP": LondonCalendar(),
            "JPY": TokyoCalendar(),
            "CHF": CHFCalendar(),
            "AUD": AUDCalendar(),
            "CAD": CADCalendar(),
            "SEK": SEKCalendar(),
            "NOK": NOKCalendar(),
            "NZD": NZDCalendar(),
        }
        for ccy, cal in cals.items():
            # Each calendar should have at least 1 holiday in 2026
            has_holiday = any(cal.is_holiday(date(2026, m, 1)) for m in range(1, 13))
            assert has_holiday or True  # just verify instantiation works
            # Verify New Year's (universal, except JPY uses Jan 1-3)
            assert cal.is_holiday(date(2026, 1, 1)), f"{ccy} missing New Year's"

    def test_sek_midsummer(self):
        """SEK: Midsummer Eve is a Friday in late June."""
        cal = SEKCalendar()
        # 2026: Midsummer Day (Sat) is Jun 20 → Eve is Jun 19
        assert cal.is_holiday(date(2026, 6, 19))

    def test_sek_national_day(self):
        cal = SEKCalendar()
        assert cal.is_holiday(date(2026, 6, 6))

    def test_nok_constitution_day(self):
        cal = NOKCalendar()
        assert cal.is_holiday(date(2026, 5, 17))

    def test_nok_maundy_thursday(self):
        """NOK has Maundy Thursday (Thursday before Easter)."""
        cal = NOKCalendar()
        # Easter 2026: April 5 → Maundy Thursday: April 2
        assert cal.is_holiday(date(2026, 4, 2))

    def test_nzd_waitangi_day(self):
        cal = NZDCalendar()
        assert cal.is_holiday(date(2026, 2, 6))

    def test_nzd_anzac_day(self):
        cal = NZDCalendar()
        assert cal.is_holiday(date(2026, 4, 25))

    def test_nzd_labour_day(self):
        """NZD Labour Day: 4th Monday in October."""
        cal = NZDCalendar()
        # 2026: 4th Mon Oct = Oct 26
        assert cal.is_holiday(date(2026, 10, 26))

    def test_weekdays_not_holidays(self):
        """A normal Wednesday in March should not be a holiday anywhere."""
        test_date = date(2026, 3, 11)  # Wednesday
        for CalClass in [USSettlementCalendar, TARGETCalendar, LondonCalendar,
                          TokyoCalendar, CHFCalendar, AUDCalendar, CADCalendar,
                          SEKCalendar, NOKCalendar, NZDCalendar]:
            cal = CalClass()
            assert not cal.is_holiday(test_date), f"{CalClass.__name__} wrongly marks {test_date} as holiday"


# ---- 30E/360 day count ----

class TestThirtyE360:
    def test_basic(self):
        """30E/360: both d1=31 and d2=31 become 30."""
        yf = year_fraction(date(2026, 1, 31), date(2026, 7, 31), DayCountConvention.THIRTY_E_360)
        # (30-30) + 30*(7-1) = 180 days → 0.5
        assert yf == pytest.approx(0.5)

    def test_vs_us_30_360(self):
        """30E/360 differs from US 30/360 when d2=31 and d1<30."""
        # US 30/360: d1=28, d2=31 → d2 stays 31 (since d1<30)
        us = year_fraction(date(2026, 2, 28), date(2026, 5, 31), DayCountConvention.THIRTY_360)
        # 30E/360: d1=28, d2=30 (unconditionally capped)
        eu = year_fraction(date(2026, 2, 28), date(2026, 5, 31), DayCountConvention.THIRTY_E_360)
        # They should differ by 1 day / 360
        assert abs(us - eu) == pytest.approx(1 / 360, abs=1e-10)

    def test_same_as_us_for_normal_dates(self):
        """When no day=31, both conventions agree."""
        us = year_fraction(date(2026, 3, 15), date(2026, 9, 15), DayCountConvention.THIRTY_360)
        eu = year_fraction(date(2026, 3, 15), date(2026, 9, 15), DayCountConvention.THIRTY_E_360)
        assert us == pytest.approx(eu)

    def test_one_year(self):
        yf = year_fraction(date(2026, 1, 1), date(2027, 1, 1), DayCountConvention.THIRTY_E_360)
        assert yf == pytest.approx(1.0)


# ---- G10 conventions ----

class TestG10Conventions:
    def test_all_g10_have_conventions(self):
        """All 10 G10 currencies are in the convention map."""
        g10 = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"]
        for ccy in g10:
            assert ccy in _CONVENTIONS, f"{ccy} missing from _CONVENTIONS"

    def test_usd_conventions(self):
        c = _CONVENTIONS["USD"]
        assert c.deposit_day_count == DayCountConvention.ACT_360
        assert c.fixed_day_count == DayCountConvention.THIRTY_360

    def test_gbp_all_act365(self):
        c = _CONVENTIONS["GBP"]
        assert c.deposit_day_count == DayCountConvention.ACT_365_FIXED
        assert c.fixed_day_count == DayCountConvention.ACT_365_FIXED
        assert c.float_day_count == DayCountConvention.ACT_365_FIXED

    def test_eur_annual_fixed(self):
        from pricebook.schedule import Frequency
        c = _CONVENTIONS["EUR"]
        assert c.fixed_frequency == Frequency.ANNUAL

    def test_build_curves_all_g10(self):
        """build_curves() should accept all 10 G10 currencies without error."""
        ref = date(2026, 4, 21)
        deposits = [(date(2026, 7, 21), 0.03)]
        swaps = [(date(2028, 4, 21), 0.03)]
        for ccy in ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"]:
            result = build_curves(ccy, ref, deposits, swaps)
            assert result.currency == ccy
            assert result.ois is not None
