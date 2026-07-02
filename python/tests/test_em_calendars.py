"""Tests for EM calendars and calendar registry.

Covers: 24 EM calendars (CEE, MENA, Africa, LatAm, Asia, DKK),
calendar registry (get_calendar, list_calendars), Orthodox Easter,
business day conventions, and cross-calendar consistency.
"""

import pytest
from datetime import date, timedelta

from pricebook.core.calendar import (
    # CEE
    WarsawCalendar, PragueCalendar, BudapestCalendar, BucharestCalendar,
    # Turkey & MENA
    IstanbulCalendar, RiyadhCalendar, TelAvivCalendar, CairoCalendar,
    # Africa
    JohannesburgCalendar, NairobiCalendar, LagosCalendar,
    # LatAm
    SaoPauloCalendar, MexicoCityCalendar, SantiagoCalendar, BogotaCalendar,
    # Asia
    BeijingCalendar, SeoulCalendar, MumbaiCalendar, SingaporeCalendar,
    HongKongCalendar, JakartaCalendar, KualaLumpurCalendar, BangkokCalendar,
    ManilaCalendar,
    # Other
    DenmarkCalendar,
    # Registry
    get_calendar, list_calendars,
    # Easter algorithms
    _orthodox_easter,
    _gregorian_easter,
    # Base + convention
    BusinessDayConvention, JointCalendar,
)


# ═══════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════


class TestCalendarRegistry:
    def test_get_g10(self):
        for code in ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "SEK", "NOK", "NZD"]:
            cal = get_calendar(code)
            assert cal.is_business_day(date(2024, 7, 8))  # a Monday, no G10 holiday

    def test_get_em(self):
        em_codes = [
            "PLN", "CZK", "HUF", "RON", "TRY", "SAR", "ILS", "EGP",
            "ZAR", "KES", "NGN", "BRL", "MXN", "CLP", "COP",
            "CNY", "KRW", "INR", "SGD", "HKD", "IDR", "MYR", "THB", "PHP",
            "DKK",
        ]
        for code in em_codes:
            cal = get_calendar(code)
            assert cal is not None

    def test_case_insensitive(self):
        cal1 = get_calendar("usd")
        cal2 = get_calendar("USD")
        # Both should work (same type)
        assert type(cal1) == type(cal2)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="No calendar for"):
            get_calendar("XYZ")

    def test_list_calendars(self):
        codes = list_calendars()
        assert "USD" in codes
        assert "BRL" in codes
        assert "INR" in codes
        assert len(codes) >= 35  # 11 G10 + 24 EM

    def test_registry_count(self):
        codes = list_calendars()
        assert len(codes) == 37


# ═══════════════════════════════════════════════════════════════
# Orthodox Easter
# ═══════════════════════════════════════════════════════════════


class TestOrthodoxEaster:
    def test_known_dates(self):
        # Well-known Orthodox Easter dates (Gregorian)
        assert _orthodox_easter(2024) == date(2024, 5, 5)
        assert _orthodox_easter(2025) == date(2025, 4, 20)
        assert _orthodox_easter(2023) == date(2023, 4, 16)

    def test_differs_from_western(self):
        """Orthodox and Western Easter often differ."""
        # 2024: Western = Mar 31, Orthodox = May 5
        western = _gregorian_easter(2024)
        orthodox = _orthodox_easter(2024)
        assert western != orthodox


# ═══════════════════════════════════════════════════════════════
# CEE Calendars
# ═══════════════════════════════════════════════════════════════


class TestCEECalendars:
    def test_warsaw_constitution_day(self):
        cal = WarsawCalendar()
        assert cal.is_holiday(date(2024, 5, 3))  # May 3rd Constitution Day

    def test_warsaw_corpus_christi(self):
        cal = WarsawCalendar()
        # 2024 Easter = Mar 31, Corpus Christi = Easter + 60 = May 30
        assert cal.is_holiday(date(2024, 5, 30))

    def test_prague_jan_hus(self):
        cal = PragueCalendar()
        assert cal.is_holiday(date(2024, 7, 6))  # Jan Hus Day

    def test_prague_christmas_eve(self):
        cal = PragueCalendar()
        assert cal.is_holiday(date(2024, 12, 24))

    def test_budapest_revolution(self):
        cal = BudapestCalendar()
        assert cal.is_holiday(date(2024, 3, 15))  # 1848 Revolution
        assert cal.is_holiday(date(2024, 10, 23))  # Republic Day

    def test_budapest_whit_monday(self):
        cal = BudapestCalendar()
        # 2024 Easter = Mar 31, Whit Monday = Easter + 50 = May 20
        assert cal.is_holiday(date(2024, 5, 20))

    def test_bucharest_orthodox_easter(self):
        cal = BucharestCalendar()
        # 2024 Orthodox Easter = May 5
        assert cal.is_holiday(date(2024, 5, 3))  # Orthodox Good Friday
        assert cal.is_holiday(date(2024, 5, 6))  # Orthodox Easter Monday

    def test_bucharest_national_day(self):
        cal = BucharestCalendar()
        assert cal.is_holiday(date(2024, 12, 1))  # National Day


# ═══════════════════════════════════════════════════════════════
# Turkey & MENA
# ═══════════════════════════════════════════════════════════════


class TestTurkeyMENA:
    def test_istanbul_republic_day(self):
        cal = IstanbulCalendar()
        assert cal.is_holiday(date(2024, 10, 29))

    def test_istanbul_victory_day(self):
        cal = IstanbulCalendar()
        assert cal.is_holiday(date(2024, 8, 30))

    def test_riyadh_national_day(self):
        cal = RiyadhCalendar()
        assert cal.is_holiday(date(2024, 9, 23))

    def test_riyadh_founding_day(self):
        cal = RiyadhCalendar()
        assert cal.is_holiday(date(2024, 2, 22))
        assert not cal.is_holiday(date(2021, 2, 22))  # Before 2022

    def test_tel_aviv_weekend(self):
        """Israel has Friday-Saturday weekend."""
        cal = TelAvivCalendar()
        friday = date(2024, 7, 5)  # a Friday
        saturday = date(2024, 7, 6)
        sunday = date(2024, 7, 7)
        assert cal.is_weekend(friday)
        assert cal.is_weekend(saturday)
        assert not cal.is_weekend(sunday)

    def test_cairo_revolution_day(self):
        cal = CairoCalendar()
        assert cal.is_holiday(date(2024, 7, 23))


# ═══════════════════════════════════════════════════════════════
# Africa
# ═══════════════════════════════════════════════════════════════


class TestAfricaCalendars:
    def test_johannesburg_freedom_day(self):
        cal = JohannesburgCalendar()
        assert cal.is_holiday(date(2024, 4, 27))  # Saturday → no observe needed

    def test_johannesburg_sunday_observe(self):
        """South Africa moves Sunday holidays to Monday."""
        cal = JohannesburgCalendar()
        # Human Rights Day 2024: Mar 21 is Thursday → stays
        assert cal.is_holiday(date(2024, 3, 21))

    def test_johannesburg_good_friday(self):
        cal = JohannesburgCalendar()
        assert cal.is_holiday(date(2024, 3, 29))  # Good Friday 2024

    def test_nairobi_madaraka(self):
        cal = NairobiCalendar()
        assert cal.is_holiday(date(2024, 6, 1))

    def test_lagos_democracy_day(self):
        cal = LagosCalendar()
        assert cal.is_holiday(date(2024, 6, 12))


# ═══════════════════════════════════════════════════════════════
# LatAm
# ═══════════════════════════════════════════════════════════════


class TestLatAmCalendars:
    def test_sao_paulo_carnival(self):
        cal = SaoPauloCalendar()
        # 2024 Easter = Mar 31, Carnival Mon = Easter-48 = Feb 12, Tue = Easter-47 = Feb 13
        assert cal.is_holiday(date(2024, 2, 12))  # Carnival Monday
        assert cal.is_holiday(date(2024, 2, 13))  # Carnival Tuesday

    def test_sao_paulo_tiradentes(self):
        cal = SaoPauloCalendar()
        assert cal.is_holiday(date(2024, 4, 21))

    def test_sao_paulo_independence(self):
        cal = SaoPauloCalendar()
        assert cal.is_holiday(date(2024, 9, 7))

    def test_mexico_constitution(self):
        cal = MexicoCityCalendar()
        # Constitution Day: 1st Mon Feb 2024 = Feb 5
        assert cal.is_holiday(date(2024, 2, 5))

    def test_mexico_benito_juarez(self):
        cal = MexicoCityCalendar()
        # 3rd Mon Mar 2024 = Mar 18
        assert cal.is_holiday(date(2024, 3, 18))

    def test_mexico_maundy_thursday(self):
        cal = MexicoCityCalendar()
        # 2024 Easter = Mar 31, Maundy Thu = Mar 28
        assert cal.is_holiday(date(2024, 3, 28))

    def test_santiago_independence(self):
        cal = SantiagoCalendar()
        assert cal.is_holiday(date(2024, 9, 18))
        assert cal.is_holiday(date(2024, 9, 19))

    def test_bogota_emiliani(self):
        """Colombia moves many holidays to Monday (emiliani law)."""
        cal = BogotaCalendar()
        # Epiphany 2024: Jan 6 is Saturday → next Monday = Jan 8
        assert cal.is_holiday(date(2024, 1, 8))

    def test_bogota_independence(self):
        cal = BogotaCalendar()
        assert cal.is_holiday(date(2024, 7, 20))


# ═══════════════════════════════════════════════════════════════
# Asia
# ═══════════════════════════════════════════════════════════════


class TestAsiaCalendars:
    def test_beijing_national_day(self):
        cal = BeijingCalendar()
        assert cal.is_holiday(date(2024, 10, 1))
        assert cal.is_holiday(date(2024, 10, 2))
        assert cal.is_holiday(date(2024, 10, 3))

    def test_seoul_liberation(self):
        cal = SeoulCalendar()
        assert cal.is_holiday(date(2024, 8, 15))

    def test_seoul_hangeul(self):
        cal = SeoulCalendar()
        assert cal.is_holiday(date(2024, 10, 9))

    def test_mumbai_republic_day(self):
        cal = MumbaiCalendar()
        assert cal.is_holiday(date(2024, 1, 26))

    def test_mumbai_independence(self):
        cal = MumbaiCalendar()
        assert cal.is_holiday(date(2024, 8, 15))

    def test_singapore_national_day(self):
        cal = SingaporeCalendar()
        assert cal.is_holiday(date(2024, 8, 9))

    def test_singapore_good_friday(self):
        cal = SingaporeCalendar()
        assert cal.is_holiday(date(2024, 3, 29))

    def test_hong_kong_hksar(self):
        cal = HongKongCalendar()
        assert cal.is_holiday(date(2024, 7, 1))

    def test_jakarta_independence(self):
        cal = JakartaCalendar()
        assert cal.is_holiday(date(2024, 8, 17))

    def test_kuala_lumpur_merdeka(self):
        cal = KualaLumpurCalendar()
        assert cal.is_holiday(date(2024, 8, 31))

    def test_bangkok_songkran(self):
        cal = BangkokCalendar()
        assert cal.is_holiday(date(2024, 4, 13))
        assert cal.is_holiday(date(2024, 4, 14))
        assert cal.is_holiday(date(2024, 4, 15))

    def test_manila_independence(self):
        cal = ManilaCalendar()
        assert cal.is_holiday(date(2024, 6, 12))

    def test_manila_rizal_day(self):
        cal = ManilaCalendar()
        assert cal.is_holiday(date(2024, 12, 30))


# ═══════════════════════════════════════════════════════════════
# Denmark
# ═══════════════════════════════════════════════════════════════


class TestDenmarkCalendar:
    def test_constitution_day(self):
        cal = DenmarkCalendar()
        assert cal.is_holiday(date(2024, 6, 5))

    def test_great_prayer_day_removed(self):
        """Store Bededag abolished after 2023."""
        cal = DenmarkCalendar()
        # 2023 Easter = Apr 9, Great Prayer Day = Easter + 26 = May 5
        assert cal.is_holiday(date(2023, 5, 5))
        # 2024: no longer a holiday
        # 2024 Easter = Mar 31, Easter + 26 = Apr 26
        assert not cal.is_holiday(date(2024, 4, 26))


# ═══════════════════════════════════════════════════════════════
# Business Day Conventions (cross-calendar)
# ═══════════════════════════════════════════════════════════════


class TestBusinessDayConventions:
    def test_add_business_days_brl(self):
        """BRL calendar skips Carnival."""
        cal = SaoPauloCalendar()
        # Feb 9, 2024 (Fri) + 1 bd → Feb 12 is Carnival Mon, Feb 13 is Carnival Tue
        # Next business day = Wed Feb 14
        result = cal.add_business_days(date(2024, 2, 9), 1)
        assert result == date(2024, 2, 14)
        # Verify the carnival days are indeed holidays
        assert cal.is_holiday(date(2024, 2, 12))
        assert cal.is_holiday(date(2024, 2, 13))

    def test_modified_following_zar(self):
        cal = JohannesburgCalendar()
        # If a holiday falls on last day of month, MOD_FOLLOWING rolls back
        d = date(2024, 3, 31)  # Sunday
        adj = cal.adjust(d, BusinessDayConvention.MODIFIED_FOLLOWING)
        assert adj.month == 3  # stays in March
        assert cal.is_business_day(adj)

    def test_preceding_mxn(self):
        cal = MexicoCityCalendar()
        d = date(2024, 9, 16)  # Independence Day (Monday)
        adj = cal.adjust(d, BusinessDayConvention.PRECEDING)
        assert adj < d
        assert cal.is_business_day(adj)

    def test_all_calendars_have_business_days(self):
        """Every calendar should have at least 200 business days in 2024."""
        for code in list_calendars():
            cal = get_calendar(code)
            bd_count = sum(
                1 for i in range(366)
                if cal.is_business_day(date(2024, 1, 1) + timedelta(days=i))
            )
            assert bd_count >= 200, f"{code} has only {bd_count} business days in 2024"

    def test_joint_calendar_em(self):
        """Joint calendar of BRL + MXN has holidays from both."""
        joint = JointCalendar(SaoPauloCalendar(), MexicoCityCalendar())
        # BRL: Sep 7 (Independence), MXN: Sep 16 (Independence)
        assert joint.is_holiday(date(2024, 9, 7))
        assert joint.is_holiday(date(2024, 9, 16))
